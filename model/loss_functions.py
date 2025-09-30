# loss_functions.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossMulti(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.register_buffer('alpha', alpha if alpha is not None else None)
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
        probs = torch.softmax(logits, dim=1)
        tgt = torch.nn.functional.one_hot(target, num_classes=C).permute(0,3,1,2).float()
        dims = (0,2,3)
        TP = (probs * tgt).sum(dims)
        FP = (probs * (1 - tgt)).sum(dims)
        FN = ((1 - probs) * tgt).sum(dims)
        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return torch.pow(1 - tversky, self.gamma).mean()

try:
    from pytorch_toolbelt import losses as L
    class LovaszSoftmaxLoss(nn.Module):
        def __init__(self, per_image: bool = False): super().__init__(); self.loss = L.LovaszLoss(mode='multiclass', per_image=per_image)
        def forward(self, logits, target): return self.loss(logits, target)
except Exception:
    class LovaszSoftmaxLoss(nn.Module):
        def __init__(self, per_image: bool = False): super().__init__()
        def forward(self, logits, target): return torch.tensor(0., device=logits.device, dtype=logits.dtype)

def topk_ce_loss(logits, target, k: float = 0.3):
    ce = F.cross_entropy(logits, target, reduction='none')
    ce_flat = ce.view(-1)
    k_ = max(1, int(k * ce_flat.numel()))
    vals, _ = torch.topk(ce_flat, k_)
    return vals.mean()

def build_losses(name: str, class_weights: torch.Tensor | None, ohem_topk: float | None):
    """
    class_weights: pesos ya combinados (p.ej. derivados de class_weights.json * multiplicadores manuales).
    """
    name = name.lower()
    if name == 'ce':
        def loss_fn(logits, y):
            l = F.cross_entropy(logits, y, weight=class_weights)
            if ohem_topk: l = l + 0.2 * topk_ce_loss(logits, y, k=ohem_topk)
            return l
        return loss_fn
    elif name == 'focal':
        fl = FocalLossMulti(alpha=class_weights, gamma=2.0)
        def loss_fn(logits, y):
            l = fl(logits, y)
            if ohem_topk: l = l + 0.2 * topk_ce_loss(logits, y, k=ohem_topk)
            return l
        return loss_fn
    elif name == 'tversky':
        tv = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.5)
        return lambda logits, y: tv(logits, y)
    elif name in ['cb_focal_tversky_lovasz', 'cbfvtl']:
        fl = FocalLossMulti(alpha=class_weights, gamma=2.0)
        tv = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.5)
        lv = LovaszSoftmaxLoss(per_image=False)
        return lambda logits, y: 0.4*fl(logits, y) + 0.4*tv(logits, y) + 0.2*lv(logits, y)
    else:
        raise ValueError(f"Loss desconocida: {name}")
