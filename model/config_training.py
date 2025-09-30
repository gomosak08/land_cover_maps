# config_training.py
from __future__ import annotations
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict

BAND_STATS_PATH = Path("band_stats.npz")
CLASS_WEIGHTS_PATH = Path("class_weights.json")

# Ajusta si tus minoritarias son otras
MINORITY_CLASSES: List[int] = [4, 5, 6]

def load_band_stats():
    data = np.load(BAND_STATS_PATH)
    return data["mean"].astype(float).tolist(), data["std"].astype(float).tolist()

def load_class_weights(num_classes: int) -> torch.Tensor:
    obj = json.loads(Path(CLASS_WEIGHTS_PATH).read_text())
    w = obj.get("weights", None)
    if not isinstance(w, list) or len(w) != num_classes:
        raise ValueError("class_weights.json no tiene 'weights' válidos.")
    return torch.tensor(w, dtype=torch.float32)

def build_manual_multipliers(num_classes: int, minority: List[int],
                             base: float = 1.0,
                             minor_boosts: Dict[int, float] | None = None) -> torch.Tensor:
    if minor_boosts is None:
        minor_boosts = {4: 6.0, 5: 7.0, 6: 8.0}
    m = np.full(shape=(num_classes,), fill_value=base, dtype=np.float32)
    for c, v in minor_boosts.items():
        if 0 <= c < num_classes:
            m[c] = float(v)
    return torch.tensor(m, dtype=torch.float32)

def combine_class_weights(basic: torch.Tensor, manual_mult: torch.Tensor,
                          normalize: bool = False) -> torch.Tensor:
    w = basic * manual_mult
    if normalize:
        w = w * (len(w) / w.sum().clamp(min=1e-8))
    return w
