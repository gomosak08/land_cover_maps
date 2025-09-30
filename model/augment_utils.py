# augment_utils.py
from __future__ import annotations
import random
import numpy as np

class NChannelChannelDropout:
    def __init__(self, p: float = 0.1, max_drop: int = 3):
        self.p, self.max_drop = p, max_drop
    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None):
        if random.random() < self.p:
            C = image.shape[2]
            k = random.randint(1, min(self.max_drop, C-1))
            idx = np.random.choice(C, size=k, replace=False)
            image = image.copy(); image[..., idx] = 0.0
        return image, mask

def random_box(crop_size: int, lam: float = 0.5):
    cut_ratio = np.sqrt(1. - lam)
    h = int(crop_size * cut_ratio)
    w = int(crop_size * cut_ratio)
    y = np.random.randint(0, crop_size - h + 1)
    x = np.random.randint(0, crop_size - w + 1)
    return y, x, h, w
