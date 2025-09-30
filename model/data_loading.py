# data_loading.py
from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

# Albumentations (renombrado a 'alb' para evitar colisiones)
import albumentations as alb
from albumentations.pytorch import ToTensorV2

# ==== Compat fastai (opcional, solo para load_data legacy) ====
try:
    import fastai.vision.all as fv
    from functools import partial
    try:
        from image_processing import MSTensorImage, open_npy_mask
    except Exception:
        MSTensorImage, open_npy_mask = None, None
except Exception:
    fv = None
    partial = None
    MSTensorImage, open_npy_mask = None, None

# ======================================================
# Utils de normalización y augmentations (Albumentations)
# ======================================================

class NChannelChannelDropout:
    """Apaga aleatoriamente k canales (k<=max_drop) con probabilidad p."""
    def __init__(self, p: float = 0.1, max_drop: int = 3):
        self.p, self.max_drop = p, max_drop
    def __call__(self, image: np.ndarray, mask: Optional[np.ndarray] = None):
        if random.random() < self.p:
            C = image.shape[2]
            k = random.randint(1, min(self.max_drop, C-1))
            idx = np.random.choice(C, size=k, replace=False)
            image = image.copy(); image[..., idx] = 0.0
        return image, mask

class GaussianNoiseManual(alb.ImageOnlyTransform):
    """
    Ruido gaussiano para imágenes en escala [0,1], compatible con multiprocessing.
    std ~ U(std_min, std_max)
    """
    def __init__(self, std_min: float = 0.01, std_max: float = 0.05, always_apply=False, p: float = 0.3):
        super().__init__(always_apply, p)
        self.std_min = float(std_min)
        self.std_max = float(std_max)

    def apply(self, img, **params):
        std = float(np.random.uniform(self.std_min, self.std_max))
        noise = np.random.normal(0.0, std, size=img.shape).astype(img.dtype)
        out = img + noise
        return np.clip(out, 0.0, 1.0)

def build_transforms(mean, std):
    """
    (aug_geo, aug_noise, norm) con Albumentations, sin warning y 100% multiprocess-safe.
    """
    aug_geo = alb.Compose([
        alb.RandomRotate90(p=0.5),
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.Affine(
            scale=(0.75, 1.25),
            translate_percent=(0.0, 0.0),
            rotate=(-30, 30),
            shear=(0.0, 0.0),
            p=0.5
        ),
        alb.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
    ], p=1.0)

    # Preferir GaussianNoise si está disponible en tu versión; si no, usar el transform manual
    try:
        noise_tf = alb.GaussianNoise(sigma_limit=(0.01, 0.05), mean=0, p=0.3)
    except Exception:
        noise_tf = GaussianNoiseManual(std_min=0.01, std_max=0.05, p=0.3)

    aug_noise = alb.Compose([noise_tf], p=1.0)

    norm = alb.Compose([
        alb.Normalize(mean=mean, std=std, max_pixel_value=1.0),
        ToTensorV2(transpose_mask=True)
    ])
    return aug_geo, aug_noise, norm





# ======================================================
# Helpers de datos: emparejar y dividir
# ======================================================

def find_pairs(img_dir: str, mask_dir: str, require_mask: bool = True) -> List[Tuple[str, str]]:
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
    """0: sin minoritarias; 1: contiene alguna minoritaria."""
    try:
        m = np.load(mask_path)
        return int(np.isin(m, list(minority_set)).any())
    except Exception:
        return 0

def train_val_split(
    pairs: List[Tuple[str, str]],
    val_ratio: float = 0.2,
    seed: int = 1337,
    minority_classes: Optional[List[int]] = None,
    stratify: bool = True
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    rng = random.Random(seed)
    if stratify and minority_classes is not None:
        minority_set = set(minority_classes)
        group0, group1 = [], []
        for ip, mp in pairs:
            (group1 if stratify_key(mp, minority_set) == 1 else group0).append((ip, mp))
        def split_group(group):
            rng.shuffle(group); n_val = int(round(len(group) * val_ratio))
            return group[n_val:], group[:n_val]
        tr0, va0 = split_group(group0); tr1, va1 = split_group(group1)
        train = tr0 + tr1; val = va0 + va1
    else:
        pairs = pairs.copy(); rng.shuffle(pairs)
        n_val = int(round(len(pairs) * val_ratio))
        val = pairs[:n_val]; train = pairs[n_val:]
    rng.shuffle(train); rng.shuffle(val)
    return train, val

# Cache del split para consistencia entre train y val llamados por separado
_SPLIT_CACHE: Dict[Tuple[str, str, float, int, Tuple[int, ...], bool], Tuple[List, List]] = {}

def _get_split_cached(img_dir: str, mask_dir: str, val_ratio: float, seed: int,
                      minority_classes: List[int], stratify: bool) -> Tuple[List, List]:
    key = (str(Path(img_dir).resolve()), str(Path(mask_dir).resolve()), val_ratio, seed, tuple(minority_classes), stratify)
    if key not in _SPLIT_CACHE:
        pairs = find_pairs(img_dir, mask_dir, require_mask=True)
        _SPLIT_CACHE[key] = train_val_split(pairs, val_ratio=val_ratio, seed=seed,
                                            minority_classes=minority_classes, stratify=stratify)
    return _SPLIT_CACHE[key]

# ======================================================
# Dataset con crops garantizados + CutMix condicionado
# ======================================================

@dataclass
class DLConfig:
    crop_size: int = 512
    minority_center_prob: float = 0.0   # probabilidad de forzar crop centrado en minoritarias (train)
    min_minor_pixels: int = 256         # píxeles mínimos de minoritarias dentro del crop
    max_center_tries: int = 8           # reintentos para alcanzar min_minor_pixels
    cutmix_p: float = 0.30              # probabilidad de aplicar cutmix condicionado
    cutmix_lam: float = 0.5             # tamaño del parche ~raíz(1-lam)

class MultiBandSegDataset(Dataset):
    """
    Dataset de segmentación multibanda (C,H,W) + mask (H,W) con:
      - Crops aleatorios o centrados en minoritarias (con reintentos).
      - CutMix condicionado para inyectar minoritarias.
      - Channel-dropout y A.Compose de Albumentations.
    """
    def __init__(
        self,
        samples: List[Tuple[str, str]],
        minority_classes: List[int],
        band_mean: List[float],
        band_std: List[float],
        cfg: DLConfig,
        is_train: bool = True
    ):
        self.samples = samples
        self.minority = set(minority_classes)
        self.cfg = cfg
        self.is_train = is_train

        # Transforms
        self.aug_geo, self.aug_noise, self.norm = build_transforms(band_mean, band_std)
        self.ch_dropout = NChannelChannelDropout(p=0.1, max_drop=3)

        # rare_score por muestra (proporción de píxeles minoritarios)
        self.rare_score = []
        for _, mpath in samples:
            try:
                m = np.load(mpath)
                flat = m.reshape(-1)
                self.rare_score.append(np.isin(flat, list(self.minority)).sum() / float(max(1, flat.size)))
            except Exception:
                self.rare_score.append(0.0)
        self.rare_score = np.array(self.rare_score, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def _random_crop(self, img: np.ndarray, mask: np.ndarray, yx: Optional[Tuple[int, int]] = None):
        C, H, W = img.shape
        ch = cw = self.cfg.crop_size
        if H < ch or W < cw:
            img = np.pad(img, ((0, 0), (0, max(0, ch - H)), (0, max(0, cw - W))), mode='reflect')
            mask = np.pad(mask, ((0, max(0, ch - H)), (0, max(0, cw - W))), mode='reflect')
            H, W = img.shape[1:]
        if yx is None:
            y = random.randint(0, H - ch); x = random.randint(0, W - cw)
        else:
            yc, xc = yx; y = np.clip(int(yc) - ch // 2, 0, H - ch); x = np.clip(int(xc) - cw // 2, 0, W - cw)
        return img[:, y:y + ch, x:x + cw], mask[y:y + ch, x:x + cw]

    def _random_box(self, lam: float):
        cut_ratio = np.sqrt(1. - lam)
        h = int(self.cfg.crop_size * cut_ratio)
        w = int(self.cfg.crop_size * cut_ratio)
        y = np.random.randint(0, self.cfg.crop_size - h + 1)
        x = np.random.randint(0, self.cfg.crop_size - w + 1)
        return y, x, h, w

    def _sample_index_with_minor(self):
        cand = np.where(self.rare_score > 0)[0]
        if len(cand) == 0:
            return np.random.randint(len(self.samples))
        return int(np.random.choice(cand))

    def __getitem__(self, idx):
        ipath, mpath = self.samples[idx]
        img = np.load(ipath).astype(np.float32)  # (C,H,W), en [0,1] si ya normalizaste upstream
        mask = np.load(mpath).astype(np.int64)   # (H,W)

        # Validación: un solo crop aleatorio sin forzar ni cutmix
        if not self.is_train:
            img_c, mask_c = self._random_crop(img, mask)
            img_c = np.transpose(img_c, (1, 2, 0))
            data = self.norm(image=img_c, mask=mask_c)  # solo normaliza en val
            return data['image'], data['mask'].long()

        # ---- TRAIN ----
        want_minor = (random.random() < self.cfg.minority_center_prob)
        tries = 0
        while True:
            if want_minor:
                coords = np.argwhere(np.isin(mask, list(self.minority)))
                if len(coords) > 0:
                    yc, xc = coords[np.random.randint(len(coords))]
                    img_c, mask_c = self._random_crop(img, mask, (int(yc), int(xc)))
                else:
                    img_c, mask_c = self._random_crop(img, mask)
            else:
                img_c, mask_c = self._random_crop(img, mask)

            if want_minor:
                if np.isin(mask_c, list(self.minority)).sum() >= self.cfg.min_minor_pixels:
                    break
                tries += 1
                if tries < self.cfg.max_center_tries:
                    continue
            break

        # CutMix condicionado (inyectar desde muestra con minoritarias)
        if random.random() < self.cfg.cutmix_p:
            j = self._sample_index_with_minor()
            ip2, mp2 = self.samples[j]
            img2 = np.load(ip2).astype(np.float32)
            mask2 = np.load(mp2).astype(np.int64)

            coords2 = np.argwhere(np.isin(mask2, list(self.minority)))
            if len(coords2) > 0:
                y2, x2 = coords2[np.random.randint(len(coords2))]
                img2c, mask2c = self._random_crop(img2, mask2, (int(y2), int(x2)))
            else:
                img2c, mask2c = self._random_crop(img2, mask2)

            y, x, h, w = self._random_box(self.cfg.cutmix_lam)
            img_c[:, y:y + h, x:x + w] = img2c[:, y:y + h, x:x + w]
            mask_c[y:y + h, x:x + w] = mask2c[y:y + h, x:x + w]

        # Albumentations: channel-dropout -> geo/noise -> normalize+ToTensor
        img_c = np.transpose(img_c, (1, 2, 0))  # CHW -> HWC
        img_c, mask_c = self.ch_dropout(img_c, mask_c)
        data = self.aug_geo(image=img_c, mask=mask_c)
        data = self.aug_noise(**data)
        data = self.norm(**data)
        return data['image'], data['mask'].long()

# ======================================================
# Builders de Dataset (para usar en tu runner PyTorch)
# ======================================================

def build_train_dataset(
    band_mean: List[float],
    band_std: List[float],
    tile_size: int,
    minority_classes: List[int],
    minority_center_prob: float,
    img_dir: str,
    mask_dir: str,
    val_ratio: float = 0.2,
    seed: int = 1337,
    stratify: bool = True,
    min_minor_pixels: int = 256,
    max_center_tries: int = 8,
    cutmix_p: float = 0.30,
    cutmix_lam: float = 0.5
) -> MultiBandSegDataset:
    train_pairs, _ = _get_split_cached(img_dir, mask_dir, val_ratio, seed, minority_classes, stratify)
    cfg = DLConfig(
        crop_size=tile_size,
        minority_center_prob=minority_center_prob,
        min_minor_pixels=min_minor_pixels,
        max_center_tries=max_center_tries,
        cutmix_p=cutmix_p,
        cutmix_lam=cutmix_lam
    )
    return MultiBandSegDataset(train_pairs, minority_classes, band_mean, band_std, cfg, is_train=True)

def build_val_dataset(
    band_mean: List[float],
    band_std: List[float],
    tile_size: int,
    minority_classes: List[int],
    img_dir: str,
    mask_dir: str,
    val_ratio: float = 0.2,
    seed: int = 1337,
    stratify: bool = True
) -> MultiBandSegDataset:
    _, val_pairs = _get_split_cached(img_dir, mask_dir, val_ratio, seed, minority_classes, stratify)
    cfg = DLConfig(
        crop_size=tile_size,
        minority_center_prob=0.0,
        min_minor_pixels=0,
        max_center_tries=1,
        cutmix_p=0.0,
        cutmix_lam=0.0
    )
    return MultiBandSegDataset(val_pairs, minority_classes, band_mean, band_std, cfg, is_train=False)

# ======================================================
# Compatibilidad: load_data estilo fastai (legacy)
# ======================================================

def _normalize_ms_fastai(t: "fv.TensorImage", mean: np.ndarray, std: np.ndarray):
    mean_t = torch.tensor(mean, dtype=t.dtype, device=t.device)[:, None, None]
    std_t  = torch.tensor(std,  dtype=t.dtype, device=t.device)[:, None, None]
    return (t - mean_t) / (std_t + 1e-6)

def _open_ms_fastai(fn, mean, std):
    if MSTensorImage is None:
        raise RuntimeError("MSTensorImage no está disponible; instala fastai o usa los builders PyTorch.")
    im: "fv.TensorImage" = MSTensorImage.create(Path(fn), chnls=None, chnls_first=True)
    return _normalize_ms_fastai(im, mean, std)

def load_data(path, mask_path, img_size: (int, int), batch_size: int,
              mean: np.ndarray, std: np.ndarray, valid_pct: float = 0.2, seed: int = 42):
    """
    Compatibilidad con tu función previa que devolvía DataLoaders (fastai).
    Usa resize + flips básicos. Para oversampling/crops garantizados usa
    build_train_dataset/build_val_dataset con PyTorch.
    """
    if fv is None or MSTensorImage is None or open_npy_mask is None or partial is None:
        raise RuntimeError("Fastai o image_processing no disponibles. "
                           "Usa los builders PyTorch build_train_dataset/build_val_dataset.")
    img_path = Path(path); msk_path = Path(mask_path)
    files = sorted(img_path.glob('*.npy'), key=lambda p: int(p.stem))
    def get_mask(p): return (msk_path/f'{p.stem}.npy')

    item_tfms = [fv.Resize(img_size, method='pad', pad_mode='zeros')]
    batch_tfms = [fv.FlipItem(p=0.5), fv.DihedralItem(p=0.0)]

    dblock = fv.DataBlock(
        blocks=(fv.TransformBlock(type_tfms=partial(_open_ms_fastai, mean=mean, std=std)),
                fv.TransformBlock(type_tfms=partial(open_npy_mask, cls=fv.TensorMask, path=str(msk_path) + '/'))),
        get_items=lambda _: files,
        get_y=get_mask,
        splitter=fv.RandomSplitter(valid_pct=valid_pct, seed=seed),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )
    return dblock.dataloaders(img_path, bs=batch_size, num_workers=4, pin_memory=True)
