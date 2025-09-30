# samplers.py
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler

class MinorityOversampler(Sampler[int]):
    """
    Oversampler agresivo: más probabilidad a samples con clases minoritarias.
    Requiere dataset.rare_score en [0..1].
    """
    def __init__(self, dataset, minority_boost: float = 6.0, epoch_mult: float = 1.25):
        self.dataset = dataset
        self.N = len(dataset)
        self.epoch_len = int(round(self.N * epoch_mult))
        base = np.ones(self.N, dtype=np.float64)
        boost = (dataset.rare_score > 0).astype(np.float64) * (minority_boost - 1.0) + 1.0
        weights = base * boost
        self.weights = torch.tensor(weights / weights.sum(), dtype=torch.double)

    def __len__(self):
        return self.epoch_len

    def __iter__(self):
        idxs = torch.multinomial(self.weights, num_samples=self.epoch_len, replacement=True)
        return iter(idxs.tolist())

def build_aggressive_weighted_sampler(dataset, alpha: float = 0.85, gamma: float = 2.0) -> WeightedRandomSampler:
    rs = dataset.rare_score.astype(np.float64)
    if rs.max() > 0:
        rs = rs / (rs.max() + 1e-8)
    rs = np.power(rs + 1e-6, gamma)
    base = np.full_like(rs, 1.0 / len(rs))
    weights = alpha * rs + (1 - alpha) * base
    weights = weights / (weights.sum() + 1e-12)
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double),
                                 num_samples=len(dataset), replacement=True)
