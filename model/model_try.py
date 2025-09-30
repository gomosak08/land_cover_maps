"""
Script de experimentos para segmentación multiclase multiespectral (25 bandas)
=============================================================================

Qué hace:
- Compara **dos variantes de entrada**: (A) Proyección 1×1 (25→3) con encoder preentrenado, (B) Inflar primera conv a 25 canales.
- Permite **probar múltiples funciones de pérdida** (CE, Focal, CB‑Focal, Tversky, Focal‑Tversky, Lovasz y combinadas), con **DRW** y **OHEM** opcionales.
- Permite **sampler ponderado** por rareza y/o **centrado en minoritarias** en el crop.
- Registra por ejecución: métricas por época y los **mejores resultados** (mIoU y por‑clase) y los guarda en **Excel** (y CSV) para comparar.

Cómo usar:
- Ajusta rutas de datos (lista de (img.npy, mask.npy)).
- Ajusta medias/desvs por banda y recuentos globales por clase.
- Define grid de experimentos en `EXPERIMENTS`.
- Ejecuta: `python experiments_runner.py`.

Salida:
- `results/metrics_log.csv` y `results/metrics_log.xlsx` con todos los runs y su mejor época.
- Checkpoints `results/ckpt/<run_id>_best.pth` del mejor modelo por mIoU macro.

Requisitos:
  pip install torch torchvision timm segmentation-models-pytorch albumentations pandas openpyxl numpy opencv-python pytorch-toolbelt

Nota: El código usa un bucle corto de épocas por defecto para exploración. Sube `max_epochs` para pruebas largas.
"""

from __future__ import annotations
import os
import math
import random
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ============================
# Config global por defecto
# ============================
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
    max_epochs: int = 25   # súbelo para entrenos reales (ej. 80)
    warmup_epochs: int = 5 # DRW warmup
    tta_scales: tuple = (1.0,)  # puedes usar (0.75,1.0,1.25)

GC = GlobalCfg()

# ============================
# Utilidades reproducibilidad
# ============================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================
# Métricas
# ============================

def iou_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    ious = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(inter / (union + 1e-7))
    return torch.tensor(ious, dtype=torch.float32)

# ============================
# Pérdidas y componentes
# ============================
class FocalLossMulti(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 1.5, smooth: float = 1.0):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth
    def forward(self, logits, target):
        B,C,H,W = logits.shape
        probs = F.softmax(logits, dim=1)
        target_1h = F.one_hot(target, num_classes=C).permute(0,3,1,2).float()
        dims = (0,2,3)
        TP = (probs * target_1h).sum(dims)
        FP = (probs * (1 - target_1h)).sum(dims)
        FN = ((1 - probs) * target_1h).sum(dims)
        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        loss = torch.pow(1 - tversky, self.gamma)
        return loss.mean()

def topk_ce_loss(logits, target, k: float = 0.3):
    ce = F.cross_entropy(logits, target, reduction='none')
    ce_flat = ce.view(-1)
    k_ = max(1, int(k * ce_flat.numel()))
    vals, _ = torch.topk(ce_flat, k_)
    return vals.mean()

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, per_image: bool = False):
        super().__init__()
        try:
            from pytorch_toolbelt import losses as L
            self.loss = L.LovaszLoss(mode='multiclass', per_image=per_image)
        except Exception:
            self.loss = None
    def forward(self, logits, target):
        if self.loss is None:
            return torch.tensor(0., device=logits.device, dtype=logits.dtype)
        return self.loss(logits, target)

# Class‑Balanced weights (Cui et al.)

def class_balanced_weights(counts: np.ndarray, beta: float = 0.999) -> torch.Tensor:
    counts = counts.astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    eff_num = 1. - np.power(beta, counts)
    weights = (1. - beta) / eff_num
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)

# ============================
# Aumentaciones / Dataset
# ============================
class NChannelChannelDropout:
    def __init__(self, p: float = 0.1, max_drop: int = 3):
        self.p, self.max_drop = p, max_drop
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        if random.random() < self.p:
            C = image.shape[2]
            k = random.randint(1, min(self.max_drop, C-1))
            idx = np.random.choice(C, size=k, replace=False)
            image = image.copy(); image[..., idx] = 0.0
        return image, mask

def build_transforms(mean: List[float], std: List[float]):
    aug_geo = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(0.05,0.25,30,border_mode=0,p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
    ], p=1.0)
    aug_noise = A.Compose([A.GaussNoise(var_limit=(5.,20.), p=0.3)], p=1.0)
    norm = A.Compose([A.Normalize(mean=mean, std=std, max_pixel_value=1.0), ToTensorV2(transpose_mask=True)])
    return aug_geo, aug_noise, norm

class MultiBandSegDataset(Dataset):
    def __init__(self, samples: List[Tuple[str,str]], minority_classes: List[int], mean, std, crop_size=512, minority_center_prob=0.0):
        self.samples = samples
        self.minority = set(minority_classes)
        self.crop = crop_size
        self.p_center = minority_center_prob
        self.aug_geo, self.aug_noise, self.norm = build_transforms(mean, std)
        self.ch_dropout = NChannelChannelDropout(p=0.1, max_drop=3)
        # rareza por muestra
        self.rare_score = []
        for _, mpath in samples:
            m = np.load(mpath)
            flat = m.reshape(-1)
            if flat.size == 0:
                self.rare_score.append(0.0)
            else:
                self.rare_score.append(np.isin(flat, list(self.minority)).sum() / float(flat.size))
        self.rare_score = np.array(self.rare_score, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def _random_crop(self, img: np.ndarray, mask: np.ndarray, yx: Optional[Tuple[int,int]] = None):
        C,H,W = img.shape
        ch = cw = self.crop
        if H < ch or W < cw:
            img = np.pad(img, ((0,0),(0,max(0,ch-H)),(0,max(0,cw-W))), mode='reflect')
            mask = np.pad(mask, ((0,max(0,ch-H)),(0,max(0,cw-W))), mode='reflect')
            H,W = img.shape[1:]
        if yx is None:
            y = random.randint(0, H - ch); x = random.randint(0, W - cw)
        else:
            yc,xc = yx; y = np.clip(yc - ch//2, 0, H-ch); x = np.clip(xc - cw//2, 0, W-cw)
        return img[:, y:y+ch, x:x+cw], mask[y:y+ch, x:x+cw]

    def __getitem__(self, idx):
        ipath, mpath = self.samples[idx]
        img = np.load(ipath).astype(np.float32) # [C,H,W] en [0,1]
        mask = np.load(mpath).astype(np.int64)  # [H,W]
        if random.random() < self.p_center:
            coords = np.argwhere(np.isin(mask, list(self.minority)))
            if len(coords) > 0:
                yc,xc = coords[np.random.randint(len(coords))]
                img, mask = self._random_crop(img, mask, (int(yc), int(xc)))
            else:
                img, mask = self._random_crop(img, mask)
        else:
            img, mask = self._random_crop(img, mask)
        img = np.transpose(img, (1,2,0))
        img, mask = self.ch_dropout(img, mask)
        data = self.aug_geo(image=img, mask=mask)
        data = self.aug_noise(**data)
        data = self.norm(**data)
        return data['image'], data['mask'].long()

# Sampler ponderado por rareza

def build_weighted_sampler(dataset: MultiBandSegDataset, alpha: float = 0.7) -> WeightedRandomSampler:
    rs = dataset.rare_score
    if rs.max() > 0:
        rs = rs / (rs.max() + 1e-8)
    base = np.full_like(rs, 1.0/len(rs))
    weights = alpha*rs + (1-alpha)*base
    weights = weights / (weights.sum() + 1e-12)
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(dataset), replacement=True)

# ============================
# Modelos: Proyección vs Inflado
# ============================
class ProjectionUNetPP(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, encoder_name: str = 'resnet34', encoder_weights: str = 'imagenet'):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
    def forward(self, x):
        return self.model(self.proj(x))

class InflatedUNetPP(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, encoder_name: str = 'resnet34', encoder_weights: str = 'imagenet'):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,  # temporal; inflaremos conv1
            classes=num_classes,
            activation=None,
        )
        self.inflate_first_conv_to_n_channels(in_channels)
    def inflate_first_conv_to_n_channels(self, new_in_channels: int):
        conv1 = None
        for name in ['conv1', 'stem.conv1']:
            conv1 = getattr(self.model.encoder, name, None)
            if conv1 is not None:
                break
        if conv1 is None:
            raise ValueError('No conv1 found in encoder')
        W = conv1.weight.data
        new_conv = nn.Conv2d(new_in_channels, W.shape[0], kernel_size=conv1.kernel_size, stride=conv1.stride, padding=conv1.padding, bias=False)
        with torch.no_grad():
            meanW = W.mean(dim=1, keepdim=True)
            new_conv.weight[:] = meanW.repeat(1, new_in_channels, 1, 1)
        if hasattr(self.model.encoder, 'conv1'):
            self.model.encoder.conv1 = new_conv
        elif hasattr(self.model.encoder, 'stem') and hasattr(self.model.encoder.stem, 'conv1'):
            self.model.encoder.stem.conv1 = new_conv
    def forward(self, x):
        return self.model(x)

# ============================
# Entrenamiento / Validación / TTA simple
# ============================

def build_losses(name: str, class_weights: Optional[torch.Tensor], ohem_topk: Optional[float]):
    name = name.lower()
    ce = lambda logits, y: F.cross_entropy(logits, y, weight=class_weights)
    if name == 'ce':
        def loss_fn(logits, y):
            l = ce(logits, y)
            if ohem_topk: l = l + 0.2*topk_ce_loss(logits, y, k=ohem_topk)
            return l
        return loss_fn
    elif name == 'focal':
        fl = FocalLossMulti(alpha=class_weights, gamma=2.0)
        def loss_fn(logits, y):
            l = fl(logits, y)
            if ohem_topk: l = l + 0.2*topk_ce_loss(logits, y, k=ohem_topk)
            return l
        return loss_fn
    elif name == 'cb_focal_tversky_lovasz':
        fl = FocalLossMulti(alpha=class_weights, gamma=2.0)
        tv = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.5)
        lv = LovaszSoftmaxLoss(per_image=False)
        def loss_fn(logits, y):
            return 0.4*fl(logits, y) + 0.4*tv(logits, y) + 0.2*lv(logits, y)
        return loss_fn
    elif name == 'tversky':
        tv = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.5)
        return lambda logits, y: tv(logits, y)
    elif name == 'lovasz':
        lv = LovaszSoftmaxLoss(per_image=False)
        return lambda logits, y: lv(logits, y)
    else:
        raise ValueError(f"Loss desconocida: {name}")


def tta_predict(model: nn.Module, image: torch.Tensor, scales=(1.0,)) -> torch.Tensor:
    model.eval(); device = next(model.parameters()).device
    with torch.no_grad():
        H,W = image.shape[2:]
        logits_acc = torch.zeros((1, GC.num_classes, H, W), device=device)
        for s in scales:
            if abs(s-1.0) < 1e-6:
                img_s = image
            else:
                Hs, Ws = int(H*s), int(W*s)
                img_s = F.interpolate(image, size=(Hs,Ws), mode='bilinear', align_corners=False)
            for hflip in [False, True]:
                for vflip in [False, True]:
                    img_aug = img_s
                    if hflip: img_aug = torch.flip(img_aug, dims=[3])
                    if vflip: img_aug = torch.flip(img_aug, dims=[2])
                    logits = model(img_aug.to(device))
                    if vflip: logits = torch.flip(logits, dims=[2])
                    if hflip: logits = torch.flip(logits, dims=[3])
                    if img_aug.shape[2:] != (H,W):
                        logits = F.interpolate(logits, size=(H,W), mode='bilinear', align_corners=False)
                    logits_acc += logits
        logits_acc /= (len(scales) * 4)
    return logits_acc

# ============================
# Runner
# ============================
@dataclass
class RunConfig:
    run_name: str
    seed: int
    model_variant: str         # 'projection' | 'inflated'
    encoder_name: str = 'resnet34'
    encoder_weights: str = 'imagenet'
    loss_name: str = 'cb_focal_tversky_lovasz'
    drw: bool = True           # deferred re-weighting (usar pesos tras warmup)
    ohem_topk: Optional[float] = 0.3
    sampler: str = 'weighted'  # 'weighted' | 'random'
    minority_center_prob: float = 0.7

# === Ajusta tus datos aquí ===
# Tus imágenes/máscaras NO son consecutivas ni están divididas en train/val.
# Vamos a escanear carpetas y emparejar por nombre de archivo (sin extensión),
# luego hacemos split reproducible y opcionalmente estratificado por presencia de clases minoritarias.

from pathlib import Path

def find_pairs(img_dir: str, mask_dir: str, require_mask: bool = True) -> List[Tuple[str,str]]:
    img_dir = Path(img_dir); mask_dir = Path(mask_dir)
    img_paths = {p.stem: p for p in img_dir.glob('*.npy')}
    mask_paths = {p.stem: p for p in mask_dir.glob('*.npy')}
    keys = sorted(set(img_paths.keys()) & set(mask_paths.keys())) if require_mask else sorted(img_paths.keys())
    pairs = []
    for k in keys:
        ip = img_paths[k]
        mp = mask_paths.get(k, None)
        if require_mask and mp is None:
            continue
        pairs.append((str(ip), str(mp)))
    return pairs


def stratify_key(mask_path: str, minority_set: set) -> int:
    # 0: sin minoritarias, 1: con minoritarias
    try:
        m = np.load(mask_path)
        return int(np.isin(m, list(minority_set)).any())
    except Exception:
        return 0


def train_val_split(pairs: List[Tuple[str,str]], val_ratio: float = 0.2, seed: int = 1337,
                    minority_classes: Optional[List[int]] = None, stratify: bool = True) -> Tuple[List[Tuple[str,str]], List[Tuple[str,str]]]:
    rng = random.Random(seed)
    if stratify and minority_classes is not None:
        minority_set = set(minority_classes)
        group0, group1 = [], []
        for ip, mp in pairs:
            (group1 if stratify_key(mp, minority_set)==1 else group0).append((ip,mp))
        def split_group(group):
            rng.shuffle(group)
            n_val = int(round(len(group) * val_ratio))
            return group[n_val:], group[:n_val]
        tr0, va0 = split_group(group0)
        tr1, va1 = split_group(group1)
        train = tr0 + tr1; val = va0 + va1
    else:
        pairs = pairs.copy(); rng.shuffle(pairs)
        n_val = int(round(len(pairs) * val_ratio))
        val = pairs[:n_val]; train = pairs[n_val:]
    rng.shuffle(train); rng.shuffle(val)
    return train, val

# Directorios REALES (edita estas rutas)
IMG_DIR = "/home/gomosak/conafor_archivo/segmentacion/cnn/img_data"
MASK_DIR = "/home/gomosak/conafor_archivo/segmentacion/cnn/img_mask"

# Encuentra y empareja archivos .npy
ALL_PAIRS = find_pairs(IMG_DIR, MASK_DIR, require_mask=True)
print(f"Encontrados {len(ALL_PAIRS)} pares img/mask")

# Configura clases minoritarias para estratificar
MINORITY_CLASSES = [3,5,6]  # ajusta a tu caso

# Haz el split reproducible (80/20 por defecto)
TRAIN_SAMPLES, VAL_SAMPLES = train_val_split(ALL_PAIRS, val_ratio=0.2, seed=GC.seed,
                                            minority_classes=MINORITY_CLASSES, stratify=True)
print(f"Train={len(TRAIN_SAMPLES)}  Val={len(VAL_SAMPLES)}")

# Estadísticos de normalización por banda (rellena con tus valores reales)
BAND_MEAN = [0.1]*GC.in_channels
BAND_STD  = [0.2]*GC.in_channels

# Recuentos globales por clase (para Class‑Balanced). Pon tus números reales
CLASS_COUNTS = np.array([10_000_000, 8_000_000, 5_000_000, 400_000, 300_000, 150_000, 100_000])
CB_WEIGHTS = class_balanced_weights(CLASS_COUNTS, beta=0.999)

# Grid de experimentos (edita libremente)
EXPERIMENTS: List[RunConfig] = []
seeds = [1337]#, 2025]
for seed in seeds:
    for model_variant in ['projection' , 'inflated']:
        for loss_name in ['ce', 'focal', 'tversky', 'cb_focal_tversky_lovasz']:
            for sampler in ['random', 'weighted']:
                EXPERIMENTS.append(RunConfig(
                    run_name=f"{model_variant}-{loss_name}-{sampler}-s{seed}",
                    seed=seed,
                    model_variant=model_variant,
                    loss_name=loss_name,
                    sampler=sampler,
                    drw=True if loss_name in ['focal','cb_focal_tversky_lovasz'] else False,
                    ohem_topk=0.3 if loss_name in ['ce','focal','cb_focal_tversky_lovasz'] else None,
                    minority_center_prob=0.7 if sampler=='weighted' else 0.5,
                ))


# Salidas
RESULTS_DIR = 'results'
CKPT_DIR = os.path.join(RESULTS_DIR, 'ckpt')
os.makedirs(CKPT_DIR, exist_ok=True)

# ============================
# Loop de entrenamiento/validación por run
# ============================

def build_model(cfg: RunConfig) -> nn.Module:
    if cfg.model_variant == 'projection':
        return ProjectionUNetPP(GC.in_channels, GC.num_classes, encoder_name=cfg.encoder_name, encoder_weights=cfg.encoder_weights)
    elif cfg.model_variant == 'inflated':
        return InflatedUNetPP(GC.in_channels, GC.num_classes, encoder_name=cfg.encoder_name, encoder_weights=cfg.encoder_weights)
    else:
        raise ValueError('model_variant')


def train_validate(cfg: RunConfig, device: torch.device) -> Dict[str, Any]:
    set_seed(cfg.seed)

    # Dataset y DataLoaders
    train_ds = MultiBandSegDataset(TRAIN_SAMPLES, MINORITY_CLASSES, BAND_MEAN, BAND_STD, crop_size=GC.tile_size, minority_center_prob=cfg.minority_center_prob)
    val_ds   = MultiBandSegDataset(VAL_SAMPLES,   MINORITY_CLASSES, BAND_MEAN, BAND_STD, crop_size=GC.tile_size, minority_center_prob=0.0)

    if cfg.sampler == 'weighted':
        sampler = build_weighted_sampler(train_ds, alpha=0.7)
        train_loader = DataLoader(train_ds, batch_size=GC.train_batch, sampler=sampler, num_workers=GC.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=GC.train_batch, shuffle=True, num_workers=GC.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=GC.val_batch, shuffle=False, num_workers=GC.num_workers, pin_memory=True)

    # Modelo
    model = build_model(cfg).to(device)

    # Pérdidas + DRW
    cb_weights = CB_WEIGHTS.to(device)
    # Warm: sin pesos
    warm_loss = build_losses('focal' if cfg.drw else cfg.loss_name, class_weights=None, ohem_topk=None)
    # Full: según cfg
    full_loss = build_losses(cfg.loss_name, class_weights=cb_weights if ('cb' in cfg.loss_name or cfg.loss_name=='focal') else None, ohem_topk=cfg.ohem_topk)

    optimizer = torch.optim.AdamW(model.parameters(), lr=GC.lr, weight_decay=GC.weight_decay)
    # Cosine LR con warmup lineal
    def lr_lambda(step):
        total_steps = GC.max_epochs * max(1, len(train_loader))
        warmup_steps = GC.warmup_epochs * max(1, len(train_loader))
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=GC.amp)

    # Log por época
    history = []
    best_miou = -1.0
    best_epoch = -1
    best_path = os.path.join(CKPT_DIR, f"{cfg.run_name}_best.pth")

    for epoch in range(GC.max_epochs):
        model.train(); train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=GC.amp):
                logits = model(imgs)
                loss = warm_loss(logits, masks) if (cfg.drw and epoch < GC.warmup_epochs) else full_loss(logits, masks)
            scaler.scale(loss).backward()
            if GC.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GC.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * imgs.size(0)
        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Val
        model.eval(); iou_sum = torch.zeros(GC.num_classes, dtype=torch.float32)
        n_batches = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                if len(GC.tta_scales) == 1 and GC.tta_scales[0] == 1.0:
                    logits = model(imgs)
                else:
                    logits_list = []
                    for b in range(imgs.shape[0]):
                        logits_list.append(tta_predict(model, imgs[b:b+1], scales=GC.tta_scales))
                    logits = torch.cat(logits_list, dim=0)
                preds = torch.argmax(logits, dim=1)
                ious = iou_per_class(preds.cpu(), masks.cpu(), GC.num_classes)
                ious[torch.isnan(ious)] = 0.0
                iou_sum += ious
                n_batches += 1
        per_class = (iou_sum / max(1,n_batches)).tolist()
        miou = float(np.mean(per_class))

        history.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'mIoU': miou,
            **{f'IoU_{c}': per_class[c] for c in range(GC.num_classes)}
        })

        if miou > best_miou:
            best_miou = miou; best_epoch = epoch+1
            torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)}, best_path)

        print(f"[{cfg.run_name}] epoch {epoch+1}/{GC.max_epochs} loss={train_loss:.4f} mIoU={miou:.4f}")

    return {
        'cfg': asdict(cfg),
        'history': history,
        'best_miou': best_miou,
        'best_epoch': best_epoch,
        'best_ckpt': best_path,
        'per_class_best': {k: history[best_epoch-1][k] for k in history[best_epoch-1] if k.startswith('IoU_')}
    }

# ============================
# Main: ejecuta grid y guarda Excel/CSV
# ============================
if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_rows = []
    runs_meta = []
    t0 = time.time()

    for cfg in EXPERIMENTS:
        res = train_validate(cfg, device)
        # historial a filas
        for row in res['history']:
            all_rows.append({
                'run_name': cfg.run_name,
                **res['cfg'],
                **row
            })
        # resumen best
        runs_meta.append({
            'run_name': cfg.run_name,
            **res['cfg'],
            'best_epoch': res['best_epoch'],
            'best_mIoU': res['best_miou'],
            **res['per_class_best'],
            'best_ckpt': res['best_ckpt']
        })

    # DataFrames
    hist_df = pd.DataFrame(all_rows)
    best_df = pd.DataFrame(runs_meta)

    # Orden útil
    sort_cols = ['best_mIoU','model_variant','loss_name','sampler','seed']
    best_df = best_df.sort_values(by=['best_mIoU'], ascending=False)

    # Guardar
    csv_path = os.path.join(RESULTS_DIR, 'metrics_log.csv')
    xlsx_path = os.path.join(RESULTS_DIR, 'metrics_log.xlsx')
    hist_df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        hist_df.to_excel(writer, sheet_name='history', index=False)
        best_df.to_excel(writer, sheet_name='best_by_run', index=False)

    # Además, guarda un resumen por grupo (modelo×loss×sampler), promediando seeds
    group_cols = ['model_variant','loss_name','sampler']
    agg = best_df.groupby(group_cols).agg({'best_mIoU':'mean'}).reset_index().sort_values('best_mIoU', ascending=False)
    with pd.ExcelWriter(xlsx_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        agg.to_excel(writer, sheet_name='group_summary', index=False)

    print(f"Hecho en {time.time()-t0:.1f}s. Resultados en: {xlsx_path} y {csv_path}")
