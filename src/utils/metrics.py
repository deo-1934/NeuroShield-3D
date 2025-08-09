# English: Core segmentation metrics (Dice, IoU, Sensitivity/Specificity) with safe shapes.
# فارسی: متریک‌های اصلی سگمنتیشن (دایس، IoU، حساسیت/ویژگی) با بررسی امن شکل‌ها.

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F

# -------------------------- Helpers / توابع کمکی --------------------------

def _check_shapes(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, Tuple[int, ...]]:
    """
    English: Validate tensor shapes and return (num_classes, spatial_shape).
    فارسی: شکل تنسورها را بررسی کرده و (تعداد کلاس‌ها، شکل فضایی) را برمی‌گرداند.
    """
    if logits.ndim != 5:
        raise ValueError(f"[metrics] logits must be [B,C,D,H,W], got {tuple(logits.shape)}")
    if targets.ndim != 4:
        raise ValueError(f"[metrics] targets must be [B,D,H,W], got {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("[metrics] batch size mismatch between logits and targets.")
    if logits.shape[2:] != targets.shape[1:]:
        raise ValueError("[metrics] spatial shape mismatch between logits and targets.")
    return logits.shape[1], logits.shape[2:]

def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    English: Convert labels [B,D,H,W] -> one-hot [B,C,D,H,W].
    فارسی: تبدیل لیبل‌ها از [B,D,H,W] به وان‌هات [B,C,D,H,W].
    """
    return F.one_hot(labels.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

# -------------------------- Dice / دایس --------------------------

def dice_per_class(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6, ignore_bg: bool = True
) -> torch.Tensor:
    """
    English:
        Computes Dice score per class. Returns tensor [C] (or [C-1] if ignore_bg).
    فارسی:
        دایس برای هر کلاس را محاسبه می‌کند. خروجی تنسوری با طول C (یا C-1 در صورت نادیده‌گرفتن پس‌زمینه) است.
    """
    C, _ = _check_shapes(logits, targets)
    probs = torch.softmax(logits, dim=1)
    targ = one_hot(targets, C)
    dims = (0, 2, 3, 4)  # sum over batch & spatial dims / جمع روی بچ و ابعاد فضایی
    inter = torch.sum(probs * targ, dim=dims)
    union = torch.sum(probs, dim=dims) + torch.sum(targ, dim=dims)
    dice = (2 * inter + eps) / (union + eps)
    if ignore_bg and C > 1:
        dice = dice[1:]
    return dice

def dice_mean(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6, ignore_bg: bool = True
) -> torch.Tensor:
    """
    English: Mean Dice across classes (optionally skipping background).
    فارسی: میانگین دایس روی کلاس‌ها (با قابلیت نادیده‌گرفتن پس‌زمینه).
    """
    return dice_per_class(logits, targets, eps=eps, ignore_bg=ignore_bg).mean()

# -------------------------- IoU / Jaccard --------------------------

def iou_per_class(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6, ignore_bg: bool = True
) -> torch.Tensor:
    """
    English: Intersection-over-Union (Jaccard) per class.
    فارسی: اشتراک بر اتحاد (ژاکارد) برای هر کلاس.
    """
    C, _ = _check_shapes(logits, targets)
    preds_oh = one_hot(torch.argmax(logits, dim=1), C)
    targ = one_hot(targets, C)
    dims = (0, 2, 3, 4)
    inter = torch.sum(preds_oh * targ, dim=dims)
    union = torch.sum(preds_oh, dim=dims) + torch.sum(targ, dim=dims) - inter
    iou = (inter + eps) / (union + eps)
    if ignore_bg and C > 1:
        iou = iou[1:]
    return iou

# -------------------------- Sensitivity / Specificity --------------------------

def confusion_counts(
    logits: torch.Tensor, targets: torch.Tensor, cls: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    English: Returns (TP, FP, TN, FN) for a given class based on argmax predictions.
    فارسی: مقادیر (TP, FP, TN, FN) را برای یک کلاس بر اساس پیش‌بینی argmax برمی‌گرداند.
    """
    _check_shapes(logits, targets)
    preds = torch.argmax(logits, dim=1)
    pos_pred = preds == cls
    pos_true = targets == cls
    tp = (pos_pred & pos_true).sum().float()
    fp = (pos_pred & ~pos_true).sum().float()
    tn = (~pos_pred & ~pos_true).sum().float()
    fn = (~pos_pred & pos_true).sum().float()
    return tp, fp, tn, fn

def sensitivity_specificity(
    logits: torch.Tensor, targets: torch.Tensor, cls: int, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    English:
        Voxel-wise Sensitivity (Recall) and Specificity for a given class.
    فارسی:
        حساسیت (Recall) و ویژگی (Specificity) وکسل‌محور برای یک کلاس مشخص.
    """
    tp, fp, tn, fn = confusion_counts(logits, targets, cls)
    sens = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    return sens, spec
