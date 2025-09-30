# metrics.py
import fastai.vision.all as fv
import torch
from fastai.learner import Metric

# =======================
# Métricas fastai (tuyas)
# =======================

class MeanIoU(Metric):
    def __init__(self, num_classes:int):
        self.k = num_classes

    def reset(self): self.ious = []

    def accumulate(self, learn):
        pred = learn.pred.argmax(dim=1)
        y = learn.y
        ious = []
        for c in range(self.k):
            p = (pred==c); t = (y==c)
            inter = (p & t).sum().float()
            union = (p | t).sum().float().clamp_min(1)
            ious.append(inter/union)
        self.ious.append(torch.stack(ious).mean())

    @property
    def value(self):
        return torch.stack(self.ious).mean()

    @property
    def name(self):
        return "mIoU"

class PerClassIoU(Metric):
    """
    Métrica que calcula IoU (Intersection over Union) por clase.
    Devuelve un tensor de tamaño [num_classes] con el IoU de cada clase.
    """
    def __init__(self, num_classes:int, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def reset(self):
        self.intersections = torch.zeros(self.num_classes, dtype=torch.float64)
        self.unions = torch.zeros(self.num_classes, dtype=torch.float64)

    def accumulate(self, learn):
        preds = learn.pred
        targs = learn.y

        # Asegurar dimensiones (N, H, W)
        if preds.ndim == 4:  # [N, C, H, W]
            preds = torch.argmax(preds, dim=1)

        preds = preds.view(-1)
        targs = targs.view(-1)

        if self.ignore_index is not None:
            mask = targs != self.ignore_index
            preds, targs = preds[mask], targs[mask]

        for c in range(self.num_classes):
            pred_c = preds == c
            targ_c = targs == c
            inter = (pred_c & targ_c).sum().item()
            union = (pred_c | targ_c).sum().item()
            self.intersections[c] += inter
            self.unions[c] += union

    @property
    def value(self):
        # IoU por clase
        ious = torch.zeros(self.num_classes, dtype=torch.float64)
        for c in range(self.num_classes):
            if self.unions[c] > 0:
                ious[c] = self.intersections[c] / self.unions[c]
        return ious

    @property
    def name(self):
        return "PerClassIoU"

def seg_accuracy(yp, y):
    """
    Calculates segmentation accuracy.
    """
    return fv.accuracy(yp, y, axis=1)

# =========================================
# Funciones PyTorch puras para el runner 🤝
# =========================================

def iou_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                  ignore_index: int | None = None, eps: float = 1e-7) -> torch.Tensor:
    """
    Calcula IoU por clase en tensores (PyTorch puro).
    - pred: logits [N,C,H,W] o etiquetas predichas [N,H,W]
    - target: etiquetas verdaderas [N,H,W]
    Devuelve: tensor [num_classes] con IoU por clase (NaN si no hay píxeles de esa clase).
    """
    # Si vienen logits, convertir a etiquetas
    if pred.ndim == 4:  # [N, C, H, W]
        pred = pred.argmax(dim=1)

    # Aplanar
    pred = pred.view(-1)
    target = target.view(-1)

    if ignore_index is not None:
        valid = target != ignore_index
        pred = pred[valid]
        target = target[valid]

    ious = torch.empty(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            ious[c] = torch.nan
            continue
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        ious[c] = inter / (union + eps) if union > 0 else torch.nan
    return ious

def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
             ignore_index: int | None = None) -> torch.Tensor:
    """
    mIoU = media de iou_per_class ignorando NaNs.
    """
    per_cls = iou_per_class(pred, target, num_classes, ignore_index)
    return torch.nanmean(per_cls)
