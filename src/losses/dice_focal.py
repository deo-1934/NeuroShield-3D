# English: Multi-class 3D segmentation losses: Dice + (CrossEntropy or Focal).
# فارسی: ضررهای سگمنتیشن چندکلاسه 3بعدی: دایس + (کراس‌انتروپی یا فوکال).

from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F

# ----------------------- helpers / توابع کمکی -----------------------

def _one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    English: [B,D,H,W] -> [B,C,D,H,W] one-hot float.
    فارسی: لیبل از [B,D,H,W] به وان‌هات [B,C,D,H,W] تبدیل می‌شود.
    """
    return F.one_hot(y.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

def _dice_loss_mc(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
    ignore_bg: bool = True,
) -> torch.Tensor:
    """
    English: Multi-class soft Dice loss (optionally ignore background class=0).
    فارسی: دایس نرم چندکلاسه (با امکان نادیده‌گرفتن پس‌زمینه کلاس صفر).
    """
    if logits.ndim != 5 or targets.ndim != 4:
        raise ValueError("dice_loss expects logits[B,C,D,H,W] and targets[B,D,H,W].")
    B, C, _, _, _ = logits.shape

    probs = torch.softmax(logits, dim=1)                   # [B,C,D,H,W]
    targ_oh = _one_hot(targets, C)                         # [B,C,D,H,W]

    # Optionally drop background channel for Dice only
    if ignore_bg and C > 1:
        probs_d = probs[:, 1:, ...]
        targ_d = targ_oh[:, 1:, ...]
    else:
        probs_d, targ_d = probs, targ_oh

    dims = (0, 2, 3, 4)                                    # sum over batch + space
    inter = torch.sum(probs_d * targ_d, dim=dims)
    denom = torch.sum(probs_d, dim=dims) + torch.sum(targ_d, dim=dims)
    dice = (2.0 * inter + smooth) / (denom + smooth)       # [C' ]
    loss = 1.0 - dice.mean()
    return loss

def _focal_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    English: Multi-class focal cross-entropy.
    فارسی: کراس‌انتروپی فوکال چندکلاسه.
    """
    B, C, _, _, _ = logits.shape
    log_p = torch.log_softmax(logits, dim=1)               # [B,C,D,H,W]
    targ_oh = _one_hot(targets, C)                         # [B,C,D,H,W]
    ce_map = -(targ_oh * log_p)                            # [B,C,D,H,W]
    p_t = torch.exp(-ce_map)                               # equals prob of true class
    focal = (1.0 - p_t) ** gamma * ce_map                  # [B,C,D,H,W]
    if alpha is not None:
        # alpha per class, shape [C] -> [1,C,1,1,1]
        focal = focal * alpha.view(1, -1, 1, 1, 1)
    return focal.sum(dim=1).mean()                         # mean over voxels & batch

# ----------------------- main composite loss / ضرر ترکیبی -----------------------

def dice_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
    ce_weight: float = 0.5,
    ignore_bg_in_dice: bool = True,
    focal_gamma: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    English:
        Composite loss = ce_weight * CE (or Focal CE) + (1-ce_weight) * Dice.
        - logits: [B,C,D,H,W], targets: [B,D,H,W]
        - if focal_gamma > 0 -> use Focal CE; otherwise standard CrossEntropy.
        - class_weights: optional per-class weights for CE/Focal, shape [C].
    فارسی:
        ضرر ترکیبی = (وزن CE) × CE (یا فوکال CE) + (۱−وزن CE) × دایس.
        - logits: [B,C,D,H,W] ، targets: [B,D,H,W]
        - اگر focal_gamma>0 باشد از فوکال CE استفاده می‌شود؛ وگرنه CE معمولی.
        - class_weights: وزن کلاس‌ها برای CE/فوکال با شکل [C] (اختیاری).
    """
    # Dice part
    dice = _dice_loss_mc(logits, targets, smooth=smooth, ignore_bg=ignore_bg_in_dice)

    # CE / Focal part
    if focal_gamma and focal_gamma > 0.0:
        alpha = class_weights.to(logits.device) if class_weights is not None else None
        ce = _focal_ce(logits, targets, alpha=alpha, gamma=float(focal_gamma))
    else:
        ce = F.cross_entropy(
            logits, targets.long(),
            weight=class_weights.to(logits.device) if class_weights is not None else None
        )

    return ce_weight * ce + (1.0 - ce_weight) * dice

__all__ = ["dice_ce_loss"]
