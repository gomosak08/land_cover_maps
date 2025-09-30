# experiments_runner.py
from __future__ import annotations
import os, time, math, json, random
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path

from loss_functions import build_losses
from samplers import MinorityOversampler, build_aggressive_weighted_sampler
from metrics import iou_per_class
from model_creation import build_model
from data_loading import build_train_dataset, build_val_dataset  # usa tus helpers si los tienes

RESULTS_DIR = "results"; os.makedirs(RESULTS_DIR, exist_ok=True)
CKPT_DIR = os.path.join(RESULTS_DIR, "ckpt"); os.makedirs(CKPT_DIR, exist_ok=True)

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
    tta_scales: tuple = (1.0,)

GC = GlobalCfg()

MINORITY_CLASSES = [4,5,6]

def load_band_stats(path="band_stats.npz"):
    d = np.load(path)
    return d["mean"].astype(float).tolist(), d["std"].astype(float).tolist()

def load_class_weights(path="class_weights.json", num_classes=7):
    obj = json.load(open(path, "r"))
    w = obj["weights"]
    assert len(w) == num_classes, "class_weights.json no coincide con num_classes"
    return torch.tensor(w, dtype=torch.float32)

def build_manual_multipliers(num_classes=7, boosts: Dict[int,float] | None = None):
    if boosts is None:
        boosts = {4:6.0, 5:7.0, 6:8.0}
    v = np.ones(num_classes, dtype=np.float32)
    for c, m in boosts.items():
        if 0 <= c < num_classes:
            v[c] = float(m)
    return torch.tensor(v, dtype=torch.float32)

def combine_weights(base: torch.Tensor, mult: torch.Tensor, normalize=False):
    w = base * mult
    if normalize:
        w = w * (len(w) / w.sum().clamp(min=1e-8))
    return w

@dataclass
class RunConfig:
    run_name: str
    seed: int
    model_variant: str              # 'inflated' | 'projection'
    loss_name: str = 'ce'           # 'ce'|'focal'|'tversky'|'cb_focal_tversky_lovasz'
    drw: bool = False
    ohem_topk: Optional[float] = 0.3
    sampler: str = 'minority_oversampler'   # 'random'|'weighted'|'minority_oversampler'
    minority_center_prob: float = 0.95
    use_manual_boosts: bool = True

def set_seed(s:int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def train_validate(cfg: RunConfig, device, band_mean, band_std, class_w_base: torch.Tensor, manual_mult: torch.Tensor):
    set_seed(cfg.seed)

    # crea datasets (usa tus funciones; si no existen, crea DataLoaders directamente)
    train_ds = build_train_dataset(band_mean, band_std, tile_size=GC.tile_size,
                                   minority_classes=MINORITY_CLASSES,
                                   minority_center_prob=cfg.minority_center_prob)
    val_ds   = build_val_dataset(band_mean, band_std, tile_size=GC.tile_size,
                                 minority_classes=MINORITY_CLASSES)

    # sampler
    if cfg.sampler == 'weighted':
        sampler = build_aggressive_weighted_sampler(train_ds, alpha=0.85, gamma=2.0)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=GC.train_batch,
                        sampler=sampler, num_workers=GC.num_workers, pin_memory=True)
    elif cfg.sampler == 'minority_oversampler':
        sampler = MinorityOversampler(train_ds, minority_boost=6.0, epoch_mult=1.25)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=GC.train_batch,
                        sampler=sampler, num_workers=GC.num_workers, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=GC.train_batch,
                        shuffle=True, num_workers=GC.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=GC.val_batch,
                    shuffle=False, num_workers=GC.num_workers, pin_memory=True)

    # modelo
    model = build_model(variant=cfg.model_variant, in_channels=GC.in_channels, num_classes=GC.num_classes).to(device)

    # pesos
    class_weights = combine_weights(class_w_base.to(device),
                                    manual_mult.to(device) if cfg.use_manual_boosts else torch.ones_like(class_w_base),
                                    normalize=False)
    warm_loss = build_losses('focal' if cfg.drw else cfg.loss_name, class_weights=None, ohem_topk=None)
    full_loss = build_losses(cfg.loss_name, class_weights=class_weights, ohem_topk=cfg.ohem_topk)

    opt = torch.optim.AdamW(model.parameters(), lr=GC.lr, weight_decay=GC.weight_decay)
    def lr_lambda(step):
        total = GC.max_epochs * max(1, len(train_loader))
        warm  = GC.warmup_epochs * max(1, len(train_loader))
        if step < warm: return float(step+1)/float(max(1,warm))
        prog = (step - warm)/float(max(1, total - warm))
        return 0.5*(1.0 + math.cos(math.pi*prog))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=GC.amp)

    history=[]; best_miou=-1.0; best_epoch=-1
    best_path = os.path.join(CKPT_DIR, f"{cfg.run_name}_best.pth")

    for epoch in range(GC.max_epochs):
        model.train(); run_loss=0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=GC.amp):
                logits = model(x)
                loss = warm_loss(logits,y) if (cfg.drw and epoch < GC.warmup_epochs) else full_loss(logits,y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if GC.grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), GC.grad_clip)
            scaler.step(opt); scaler.update()
            run_loss += loss.item()*x.size(0)
        sch.step()
        train_loss = run_loss / len(train_loader.dataset)

        # val
        model.eval(); iou_sum = torch.zeros(GC.num_classes, dtype=torch.float32); n=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                ious = iou_per_class(preds.cpu(), y.cpu(), GC.num_classes)
                ious[torch.isnan(ious)] = 0.0
                iou_sum += ious; n += 1
        per_class = (iou_sum / max(1,n)).tolist()
        miou = float(np.mean(per_class))
        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'mIoU': miou, **{f'IoU_{c}': per_class[c] for c in range(GC.num_classes)}})
        if miou > best_miou:
            best_miou = miou; best_epoch = epoch+1
            torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)}, best_path)
        print(f"[{cfg.run_name}] {epoch+1}/{GC.max_epochs} loss={train_loss:.4f} mIoU={miou:.4f}")

    return {'cfg': asdict(cfg), 'history': history, 'best_miou': best_miou, 'best_epoch': best_epoch,
            'best_ckpt': best_path, 'per_class_best': {k: history[best_epoch-1][k] for k in history[best_epoch-1] if k.startswith('IoU_')}}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    band_mean, band_std = load_band_stats("band_stats.npz")
    base_w = load_class_weights("class_weights.json", num_classes=GC.num_classes)
    manual_mult = build_manual_multipliers(num_classes=GC.num_classes, boosts={4:6.0,5:7.0,6:8.0})

    runs: List[RunConfig] = []
    for seed in [1337, 2025]:
        for variant in ['inflated','projection']:
            runs += [
                RunConfig(run_name=f"{variant}-ce-manual-minOver-mcp95-s{seed}", seed=seed, model_variant=variant,
                          loss_name='ce', drw=False, ohem_topk=0.3, sampler='minority_oversampler',
                          minority_center_prob=0.95, use_manual_boosts=True),
                RunConfig(run_name=f"{variant}-focal-manual-minOver-mcp95-s{seed}", seed=seed, model_variant=variant,
                          loss_name='focal', drw=False, ohem_topk=0.3, sampler='minority_oversampler',
                          minority_center_prob=0.95, use_manual_boosts=True),
                RunConfig(run_name=f"{variant}-cbfvtl-minOver-mcp95-s{seed}", seed=seed, model_variant=variant,
                          loss_name='cb_focal_tversky_lovasz', drw=True, ohem_topk=None, sampler='minority_oversampler',
                          minority_center_prob=0.95, use_manual_boosts=False),
            ]

    all_rows=[]; runs_meta=[]
    t0=time.time()
    for cfg in runs:
        res = train_validate(cfg, device, band_mean, band_std, base_w, manual_mult)
        for row in res['history']:
            all_rows.append({'run_name': cfg.run_name, **res['cfg'], **row})
        runs_meta.append({'run_name': cfg.run_name, **res['cfg'], 'best_epoch': res['best_epoch'],
                          'best_mIoU': res['best_miou'], **res['per_class_best'], 'best_ckpt': res['best_ckpt']})

    hist_df = pd.DataFrame(all_rows)
    best_df = pd.DataFrame(runs_meta).sort_values('best_mIoU', ascending=False)
    csv_path = os.path.join(RESULTS_DIR, 'metrics_log.csv')
    xlsx_path = os.path.join(RESULTS_DIR, 'metrics_log.xlsx')
    hist_df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as w:
        hist_df.to_excel(w, sheet_name='history', index=False)
        best_df.to_excel(w, sheet_name='best_by_run', index=False)
        grp = best_df.groupby(['model_variant','loss_name','sampler']).agg({'best_mIoU':'mean'}).reset_index().sort_values('best_mIoU', ascending=False)
        grp.to_excel(w, sheet_name='group_summary', index=False)
    print(f"Done in {time.time()-t0:.1f}s -> {xlsx_path} / {csv_path}")

if __name__ == "__main__":
    main()
