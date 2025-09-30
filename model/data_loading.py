from fastai.vision.all import *
from functools import partial
from image_processing import MSTensorImage, open_npy_mask
import numpy as np
from pathlib import Path

def _normalize_ms(t:TensorImage, mean:np.ndarray, std:np.ndarray):
    # t: (C,H,W) float32; mean/std: shape (C,)
    mean_t = torch.tensor(mean, dtype=t.dtype, device=t.device)[:, None, None]
    std_t  = torch.tensor(std,  dtype=t.dtype, device=t.device)[:, None, None]
    return (t - mean_t) / (std_t + 1e-6)

def _open_ms(fn, mean, std):
    # Use MSTensorImage.create (channels-first) then normalize
    im:TensorImage = MSTensorImage.create(Path(fn), chnls=None, chnls_first=True)
    return _normalize_ms(im, mean, std)

def load_data(path, mask_path, img_size:(int,int), batch_size:int,
              mean:np.ndarray, std:np.ndarray, valid_pct:float=0.2, seed:int=42):
    """
    Build DataLoaders for multi-spectral npy tiles and integer masks.
    - path: folder with *.npy images (C,H,W)
    - mask_path: sibling folder with same-named *.npy masks (H,W)
    - img_size: (H, W) — will be identity for (544,480); kept for safety.
    - mean/std: per-band stats, shape (C,)
    """

    img_path = Path(path)
    msk_path = Path(mask_path)
    files = sorted(img_path.glob('*.npy'), key=lambda p: int(p.stem))

    def get_mask(p): return (msk_path/f'{p.stem}.npy')

    # basic, geometry-safe augments; avoid strong color ops (25 bands)
    item_tfms = [Resize(img_size, method='pad', pad_mode='zeros')]
    batch_tfms = [FlipItem(p=0.5), DihedralItem(p=0.0)]  # light geo aug

    dblock = DataBlock(
        blocks=(TransformBlock(type_tfms=partial(_open_ms, mean=mean, std=std)),
                TransformBlock(type_tfms=partial(open_npy_mask, cls=TensorMask, path=str(msk_path) + '/'))),
        get_items=lambda _: files,
        get_y=get_mask,
        splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )

    return dblock.dataloaders(img_path, bs=batch_size, num_workers=4, pin_memory=True)
