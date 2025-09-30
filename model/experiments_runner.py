# experiments_runner.py
from __future__ import annotations
import os, time, math, json, random, argparse
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# === Módulos del proyecto ===
from data_loading import build_train_dataset, build_val_dataset
from loss_functions import build_losses
from model_creation import build_model
from metrics import iou_per_class
from samplers import MinorityOversampler, build_aggressive_weighted_sampler

# ------------------------------
# Configuración global
# ------------------------------
@dataclass
class GlobalCfg:
    seed: int = 1337
    in_channels: int = 25
    num_classes: int = 7
    tile_size: int = 512
    train_batch: int = 12
    val_batch: int = 12
    num_workers: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    amp: bool = True
    grad_clip: float = 1.0
    max_epochs: int = 80
    warmup_epochs: int = 5
    tta_scales: Tuple[float, ...] = (1.0,)

GC = GlobalCfg()

RESULTS_DIR = "results"
CKPT_DIR = os.path.join(RESULTS_DIR, "ckpt")
os.makedirs(CKPT_DIR, exist_ok=True)

# ------------------------------
# Utilidades
# ------------------------------
def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_band_stats(path: str, expect_c: Optional[int] = None) -> Tuple[List[float], List[float]]:
    d = np.load(path)
    mean = d["mean"].astype(float)
    std  = d["std"].astype(float)
    if expect_c is not None and len(mean) != expect_c:
        raise ValueError(f"band_stats ({len(mean)}) != in_channels ({expect_c})")
    return mean.tolist(), std.tolist()

def load_class_weights(path: str, num_classes: int) -> torch.Tensor:
    if not os.path.exists(path):
        # Fallback: pesos uniformes
        print(f"[WARN] {path} no encontrado. Usando pesos uniformes.")
        return torch.ones(num_classes, dtype=torch.float32)
    obj = json.load(open(path, "r"))
    w = obj.get("weights", None)
    if not isinstance(w, list) or len(w) != num_classes:
        raise ValueError(f"class_weights en {path} no coincide con num_classes={num_classes}")
    return torch.tensor(w, dtype=torch.float32)

def build_manual_multipliers(num_classes: int, boosts: Dict[int, float]) -> torch.Tensor:
    v = np.ones(num_classes, dtype=np.float32)
    for c, m in boosts.items():
        if 0 <= c < num_classes:
            v[c] = float(m)
    return torch.tensor(v, dtype=torch.float32)

def combine_weights(base: torch.Tensor, mult: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    w = base * mult
    if normalize:
        w = w * (len(w) / w.sum().clamp(min=1e-8))
    return w

# ------------------------------
# Config por ejecución
# ------------------------------
@dataclass
class RunConfig:
    run_name: str
    seed: int
    model_variant: str                     # 'inflated' | 'projection'
    loss_name: str = 'ce'                  # 'ce'|'focal'|'tversky'|'cb_focal_tversky_lovasz'
    drw: bool = False                      # deferred re-weighting (warmup sin pesos)
    ohem_topk: Optional[float] = 0.3
    sampler: str = 'minority_oversampler'  # 'random'|'weighted'|'minority_oversampler'
    minority_center_prob: float = 0.95     # prob. de crops centrados en minoritarias (train)
    use_manual_boosts: bool = True         # aplicar boosts manuales a clases minoritarias

# ------------------------------
# Entrenamiento / Validación
# ------------------------------
def train_validate(
    cfg: RunConfig,
    device: torch.device,
    band_mean: List[float],
    band_std: List[float],
    img_dir: str,
    mask_dir: str,
    val_ratio: float,
    seed_split: int,
    minority_classes: List[int],
    class_w_base: torch.Tensor,
    manual_mult: torch.Tensor
) -> Dict[str, Any]:

    set_seed(cfg.seed)

    # Datasets
    train_ds = build_train_dataset(
        band_mean=band_mean, band_std=band_std,
        tile_size=GC.tile_size, minority_classes=minority_classes,
        minority_center_prob=cfg.minority_center_prob,
        img_dir=img_dir, mask_dir=mask_dir,
        val_ratio=val_ratio, seed=seed_split, stratify=True,
        min_minor_pixels=256, max_center_tries=8,
        cutmix_p=0.30, cutmix_lam=0.5
    )
    val_ds = build_val_dataset(
        band_mean=band_mean, band_std=band_std,
        tile_size=GC.tile_size, minority_classes=minority_classes,
        img_dir=img_dir, mask_dir=mask_dir,
        val_ratio=val_ratio, seed=seed_split, stratify=True
    )

    # Sampler
    if cfg.sampler == 'weighted':
        sampler = build_aggressive_weighted_sampler(train_ds, alpha=0.85, gamma=2.0)
        train_loader = DataLoader(train_ds, batch_size=GC.train_batch, sampler=sampler,
                                  num_workers=GC.num_workers, pin_memory=True)
    elif cfg.sampler == 'minority_oversampler':
        sampler = MinorityOversampler(train_ds, minority_boost=6.0, epoch_mult=1.25)
        train_loader = DataLoader(train_ds, batch_size=GC.train_batch, sampler=sampler,
                                  num_workers=GC.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=GC.train_batch, shuffle=True,
                                  num_workers=GC.num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=GC.val_batch, shuffle=False,
                            num_workers=GC.num_workers, pin_memory=True)

    # Modelo
    model = build_model(variant=cfg.model_variant, in_channels=GC.in_channels, num_classes=GC.num_classes).to(device)

    # Pérdidas: pesos combinados
    class_weights = combine_weights(
        base=class_w_base.to(device),
        mult=manual_mult.to(device) if cfg.use_manual_boosts else torch.ones_like(class_w_base).to(device),
        normalize=False
    )

    warm_loss = build_losses('focal' if cfg.drw else cfg.loss_name, class_weights=None, ohem_topk=None)
    full_loss = build_losses(cfg.loss_name, class_weights=class_weights, ohem_topk=cfg.ohem_topk)

    # Optimizador + LR schedule
    opt = torch.optim.AdamW(model.parameters(), lr=GC.lr, weight_decay=GC.weight_decay)

    def lr_lambda(step):
        total_steps = GC.max_epochs * max(1, len(train_loader))
        warmup_steps = GC.warmup_epochs * max(1, len(train_loader))
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=GC.amp)

    # Loop
    history = []
    best_miou = -1.0
    best_epoch = -1
    best_path = os.path.join(CKPT_DIR, f"{cfg.run_name}_best.pth")

    for epoch in range(GC.max_epochs):
        model.train(); run_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=GC.amp):
                logits = model(imgs)
                loss = warm_loss(logits, masks) if (cfg.drw and epoch < GC.warmup_epochs) else full_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if GC.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GC.grad_clip)
            scaler.step(opt); scaler.update()
            run_loss += loss.item() * imgs.size(0)
        sched.step()
        train_loss = run_loss / len(train_loader.dataset)

        # Validación
        model.eval(); iou_sum = torch.zeros(GC.num_classes, dtype=torch.float32); n = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1)
                ious = iou_per_class(preds.cpu(), masks.cpu(), GC.num_classes)
                ious[torch.isnan(ious)] = 0.0
                iou_sum += ious; n += 1
        per_class = (iou_sum / max(1, n)).tolist()
        miou = float(np.mean(per_class))

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'mIoU': miou,
            **{f'IoU_{c}': per_class[c] for c in range(GC.num_classes)}
        })

        if miou > best_miou:
            best_miou = miou; best_epoch = epoch + 1
            torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)}, best_path)

        print(f"[{cfg.run_name}] {epoch+1}/{GC.max_epochs} loss={train_loss:.4f} mIoU={miou:.4f}")

    return {
        'cfg': asdict(cfg),
        'history': history,
        'best_miou': best_miou,
        'best_epoch': best_epoch,
        'best_ckpt': best_path,
        'per_class_best': {k: history[best_epoch-1][k] for k in history[best_epoch-1] if k.startswith('IoU_')}
    }

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True, help="Carpeta con imágenes .npy (C,H,W)")
    parser.add_argument("--mask_dir", required=True, help="Carpeta con máscaras .npy (H,W)")
    parser.add_argument("--band_stats", default="band_stats.npz")
    parser.add_argument("--class_weights", default="class_weights.json")
    parser.add_argument("--minority", default="4,5,6", help="Clases minoritarias separadas por coma")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed_split", type=int, default=1337, help="Semilla para el split train/val")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sampler", choices=["minority_oversampler","weighted","random"], default="minority_oversampler")
    parser.add_argument("--no_manual_boosts", action="store_true", help="Desactiva multiplicadores manuales a minoritarias")
    parser.add_argument("--runs_quick", action="store_true", help="Modo rápido: solo 1 semilla y 1 variante")
    args = parser.parse_args()

    # Config dinámico desde CLI
    GC.max_epochs = args.epochs
    GC.train_batch = GC.val_batch = args.batch
    GC.lr = args.lr

    minority_classes = [int(x) for x in args.minority.split(",") if x.strip() != ""]
    print(f"Minority classes: {minority_classes}")

    # Cargar stats y pesos
    band_mean, band_std = load_band_stats(args.band_stats, expect_c=GC.in_channels)
    base_w = load_class_weights(args.class_weights, num_classes=GC.num_classes)
    manual_mult = build_manual_multipliers(GC.num_classes, boosts={4:6.0, 5:7.0, 6:8.0})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Definir runs
    runs: List[RunConfig] = []
    seeds = [1337] #, 2025] if not args.runs_quick else [1337]
    variants = ['inflated', 'projection'] if not args.runs_quick else ['inflated']

    for seed in seeds:
        for variant in variants:
            runs += [
                RunConfig(run_name=f"{variant}-ce-manual-{args.sampler}-mcp95-s{seed}", seed=seed,
                          model_variant=variant, loss_name='ce', drw=False, ohem_topk=0.3,
                          sampler=args.sampler, minority_center_prob=0.95,
                          use_manual_boosts=(not args.no_manual_boosts)),
                RunConfig(run_name=f"{variant}-focal-manual-{args.sampler}-mcp95-s{seed}", seed=seed,
                          model_variant=variant, loss_name='focal', drw=False, ohem_topk=0.3,
                          sampler=args.sampler, minority_center_prob=0.95,
                          use_manual_boosts=(not args.no_manual_boosts)),
                RunConfig(run_name=f"{variant}-cb_focal_tversky_lovasz-{args.sampler}-mcp95-s{seed}", seed=seed,
                          model_variant=variant, loss_name='cb_focal_tversky_lovasz', drw=True, ohem_topk=None,
                          sampler=args.sampler, minority_center_prob=0.95,
                          use_manual_boosts=False),
            ]

    # Ejecutar runs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []; runs_meta = []
    t0 = time.time()

    for rcfg in runs:
        res = train_validate(
            cfg=rcfg, device=device, band_mean=band_mean, band_std=band_std,
            img_dir=args.img_dir, mask_dir=args.mask_dir,
            val_ratio=args.val_ratio, seed_split=args.seed_split,
            minority_classes=minority_classes,
            class_w_base=base_w, manual_mult=manual_mult
        )

        # Log por época
        for row in res['history']:
            all_rows.append({'run_name': rcfg.run_name, **res['cfg'], **row})

        # Resumen best
        runs_meta.append({
            'run_name': rcfg.run_name,
            **res['cfg'],
            'best_epoch': res['best_epoch'],
            'best_mIoU': res['best_miou'],
            **res['per_class_best'],
            'best_ckpt': res['best_ckpt']
        })

    # Guardar resultados
    hist_df = pd.DataFrame(all_rows)
    best_df = pd.DataFrame(runs_meta).sort_values(by='best_mIoU', ascending=False)

    csv_path = os.path.join(RESULTS_DIR, 'metrics_log.csv')
    xlsx_path = os.path.join(RESULTS_DIR, 'metrics_log.xlsx')

    hist_df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as w:
        hist_df.to_excel(w, sheet_name='history', index=False)
        best_df.to_excel(w, sheet_name='best_by_run', index=False)
        grp = best_df.groupby(['model_variant', 'loss_name', 'sampler']).agg({'best_mIoU':'mean'}).reset_index().sort_values('best_mIoU', ascending=False)
        grp.to_excel(w, sheet_name='group_summary', index=False)

    dt = time.time() - t0
    print(f"Hecho en {dt/60:.1f} min. Resultados: {xlsx_path} | {csv_path}")

if __name__ == "__main__":
    main()
