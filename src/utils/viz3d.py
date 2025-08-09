# English: Lightweight 3D visualization helpers (no heavy deps).
# فارسی: توابع سبکِ کمکی برای نمایش و کار با حجم‌های سه‌بعدی (بدون وابستگی سنگین).

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Core: take_slice
# ----------------------------------------------------------------------

def take_slice(
    volume: np.ndarray,
    axis: int = 2,
    index: Optional[int] = None,
) -> np.ndarray:
    """
    English:
        Return a 2D slice from a 3D or 4D volume.
        - If volume is 3D: [D, H, W]
        - If volume is 4D: [C, D, H, W] (first dimension is channels); the function
          will slice over the last three dims, ignoring channel dim.
        axis: which spatial axis to slice over (0->D, 1->H, 2->W)
        index: which slice index to take; if None, the middle slice is used.

    فارسی:
        یک اسلایس دوبعدی از حجم ۳بعدی یا ۴بعدی برمی‌گرداند.
        - اگر حجم ۳بعدی باشد: [D,H,W]
        - اگر ۴بعدی باشد: [C,D,H,W] (کانال‌ها در بعد اول)، این تابع روی سه بعد فضایی برش می‌زند و بعد کانال را نادیده می‌گیرد.
        axis: محور فضایی برای برش (۰=عمق D، ۱=ارتفاع H، ۲=عرض W)
        index: شماره‌ی اسلایس؛ اگر None باشد اسلایس وسط انتخاب می‌شود.
    """
    if volume.ndim not in (3, 4):
        raise ValueError(f"[viz3d.take_slice] volume must be 3D or 4D, got shape {volume.shape}")

    # If 4D, drop channel dim by taking the first channel by default
    # اگر ۴بعدی بود، به‌صورت پیش‌فرض کانال صفر را در نظر می‌گیریم (کانال‌ها دادهٔ فضایی نیستند)
    if volume.ndim == 4:
        # keep spatial [D,H,W]
        volume = volume[0]

    if axis not in (0, 1, 2):
        raise ValueError(f"[viz3d.take_slice] axis must be 0,1,2; got {axis}")

    spatial_shape = volume.shape  # (D,H,W)
    if index is None:
        index = spatial_shape[axis] // 2  # middle slice / اسلایس وسط

    # clamp index to valid range / محدودسازی اندیس به بازه‌ی معتبر
    index = int(max(0, min(index, spatial_shape[axis] - 1)))

    # extract slice / استخراج اسلایس
    sl = np.take(volume, indices=index, axis=axis)

    # ensure contiguous float32 for plotting libs / تضمین float32 پیوسته برای نمایش
    return np.ascontiguousarray(sl).astype(np.float32)


# ----------------------------------------------------------------------
# Optional helpers useful for visualization pipelines (not required by demo)
# توابع کمکی اختیاری برای نمایش (در دمو استفاده نمی‌شوند، ولی مفیدند)
# ----------------------------------------------------------------------

def normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    English: Min-max normalize to [0,1], safe if constant array.
    فارسی: نرمال‌سازی مین-مکس به بازه [0,1] با پایداری برای آرایه‌ی ثابت.
    """
    x = x.astype(np.float32, copy=False)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)

def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    English: Map integer labels to simple RGB colors for visualization.
    فارسی: نگاشت لیبل‌های عددی به رنگ‌های ساده RGB برای نمایش.
    Labels: 0=background, 1=red, 2=green, 3=blue (others wrap).
    """
    mask = mask.astype(np.int32, copy=False)
    h, w = mask.shape[-2], mask.shape[-1]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    # simple palette / پالت ساده
    colors = np.array([
        [0.0, 0.0, 0.0],  # bg
        [1.0, 0.0, 0.0],  # class 1 -> red
        [0.0, 1.0, 0.0],  # class 2 -> green
        [0.0, 0.0, 1.0],  # class 3 -> blue
        [1.0, 1.0, 0.0],  # extras (yellow)
        [1.0, 0.0, 1.0],  # magenta
        [0.0, 1.0, 1.0],  # cyan
    ], dtype=np.float32)

    idx = mask % len(colors)
    rgb = colors[idx]
    return rgb  # [H,W,3] in 0..1

def overlay_gray_with_mask(gray: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    English: Create an RGB image by overlaying a colorized mask over a grayscale slice.
    فارسی: ساخت تصویر RGB با قرار دادن ماسک رنگی روی اسلایس خاکستری.
    """
    g = normalize01(gray)
    rgb = np.stack([g, g, g], axis=-1)  # [H,W,3]
    cm = colorize_mask(mask)
    return np.clip((1 - alpha) * rgb + alpha * cm, 0.0, 1.0)

__all__ = ["take_slice", "normalize01", "colorize_mask", "overlay_gray_with_mask"]
