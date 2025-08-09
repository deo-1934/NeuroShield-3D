# English: BraTS-style dataset that yields 3D patches (with safe debug fallback).
# فارسی: دیتاست سبک BraTS که پچ‌های سه‌بعدی می‌دهد (با حالت امنِ دیباگ در نبود داده).

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Sequence, Optional, Dict, Any, List
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.nii_io import load_nii  # English: NIfTI helpers | فارسی: توابع کمکی NIfTI

# ------------------------- Helpers / توابع کمکی -------------------------

def _zscore_inplace(x: np.ndarray) -> np.ndarray:
    """
    English: Z-score normalize ignoring zeros (common for MRI).
    فارسی: نرمال‌سازی Z-score با نادیده گرفتن صفرها (مرسوم در MRI).
    """
    m = x != 0
    if m.sum() > 10:
        mu = float(x[m].mean())
        std = float(x[m].std())
        if std > 1e-8:
            x[m] = (x[m] - mu) / std
    return x

def _ensure_min_size(vol: np.ndarray, target: Tuple[int,int,int]) -> np.ndarray:
    """
    English: Pad (constant=0) a 3D volume so each dim >= target dim.
    فارسی: حجم سه‌بعدی را با صفر پد می‌کند تا هر بُعد حداقل به اندازه‌ی هدف برسد.
    """
    dz = max(target[0] - vol.shape[0], 0)
    dy = max(target[1] - vol.shape[1], 0)
    dx = max(target[2] - vol.shape[2], 0)
    if dz or dy or dx:
        pad = ((dz//2, dz - dz//2), (dy//2, dy - dy//2), (dx//2, dx - dx//2))
        vol = np.pad(vol, pad, mode="constant", constant_values=0)
    return vol

def _random_patch_start(shape: Tuple[int,int,int], patch: Tuple[int,int,int], rng: np.random.Generator) -> Tuple[int,int,int]:
    """
    English: Choose a random top-left-front index so patch fits inside the volume.
    فارسی: نقطه شروع تصادفی انتخاب می‌کند تا پچ داخل حجم جا شود.
    """
    z = 0 if shape[0] <= patch[0] else rng.integers(0, shape[0] - patch[0] + 1)
    y = 0 if shape[1] <= patch[1] else rng.integers(0, shape[1] - patch[1] + 1)
    x = 0 if shape[2] <= patch[2] else rng.integers(0, shape[2] - patch[2] + 1)
    return int(z), int(y), int(x)

def _crop(vol: np.ndarray, start: Tuple[int,int,int], size: Tuple[int,int,int]) -> np.ndarray:
    """English: Crop 3D [Z,Y,X]. | فارسی: برش سه‌بعدی [Z,Y,X]."""
    z, y, x = start
    dz, dy, dx = size
    return vol[z:z+dz, y:y+dy, x:x+dx]

def _make_synthetic_case(patch_size: Tuple[int,int,int], C: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    English: Create a synthetic multi-modal volume and a spherical lesion mask.
    فارسی: ساخت حجم چندکاناله مصنوعی و ماسک کروی برای تست بدون داده واقعی.
    """
    D, H, W = patch_size
    vol = rng.normal(0, 1, size=(C, D, H, W)).astype(np.float32)
    mask = np.zeros((D, H, W), dtype=np.int16)
    # spherical lesion
    cz, cy, cx = D//2, H//2, W//2
    r = min(D, H, W) // 6
    zz, yy, xx = np.ogrid[:D, :H, :W]
    sphere = (zz-cz)**2 + (yy-cy)**2 + (xx-cx)**2 <= r*r
    mask[sphere] = 1  # English: treat as ET for debug | فارسی: به عنوان کلاس ET برای دیباگ
    return vol, mask

# ------------------------- Dataset / دیتاست -------------------------

class BratsPatches(Dataset):
    """
    English:
        Expects per-case folders under `root`, each with files:
        {t1.nii.gz, t1ce.nii.gz, t2.nii.gz, flair.nii.gz, mask.nii.gz}
        Returns: image [C,D,H,W] float32, label [D,H,W] int64.
        If `debug=True` and no data exists, generates synthetic samples.

    فارسی:
        انتظار دارد برای هر کیس یک پوشه زیر `root` باشد با فایل‌های:
        {t1.nii.gz، t1ce.nii.gz، t2.nii.gz، flair.nii.gz، mask.nii.gz}
        خروجی: تصویر [C,D,H,W] از نوع float32 و لیبل [D,H,W] از نوع int64.
        اگر `debug=True` باشد و داده‌ای موجود نباشد، نمونه‌های مصنوعی تولید می‌کند.
    """

    def __init__(
        self,
        root: Path | str,
        patch_size: Tuple[int,int,int] = (128,128,128),
        modalities: Sequence[str] = ("t1", "t1ce", "t2", "flair"),
        transforms: Optional[Any] = None,
        case_glob: str = "*",
        debug: bool = False,
        debug_len: int = 16,
        rng_seed: int = 42,
    ):
        self.root = Path(root)
        self.patch_size = tuple(map(int, patch_size))
        self.modalities = tuple(modalities)
        self.transforms = transforms
        self.debug = bool(debug)
        self.debug_len = int(debug_len)
        self.rng = np.random.default_rng(int(rng_seed))

        self.cases: List[Path] = []
        if self.root.exists():
            self.cases = sorted([p for p in self.root.glob(case_glob) if p.is_dir()])

        # English: if no cases and not debug -> length 0 (trainer should handle)
        # فارسی: اگر داده‌ای نبود و دیباگ خاموش بود -> طول ۰ (لوپ آموزش باید این را مدیریت کند)
        if len(self.cases) == 0 and not self.debug:
            print(f"[BratsPatches] No cases found under: {self.root}")

    def __len__(self) -> int:
        return self.debug_len if (self.debug and len(self.cases) == 0) else len(self.cases)

    # --------------------- Real-case loader / بارگذار کیس واقعی ---------------------

    def _load_case(self, case_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        English: Load modalities + mask from a case directory.
        فارسی: بارگذاری مودالیتی‌ها + ماسک از پوشه‌ی یک کیس.
        """
        vols: List[np.ndarray] = []
        ref_affine = None
        ref_header = None
        for m in self.modalities:
            path = case_dir / f"{m}.nii.gz"
            if not path.exists():
                # Try `.nii`
                path = case_dir / f"{m}.nii"
            vol, affine, header = load_nii(path)
            if ref_affine is None:
                ref_affine, ref_header = affine, header
            vols.append(vol.astype(np.float32, copy=False))
        mask_path = case_dir / "mask.nii.gz"
        if not mask_path.exists():
            mask_path = case_dir / "mask.nii"
        mask, _, _ = load_nii(mask_path)
        mask = mask.astype(np.int16, copy=False)
        # Z-score per modality (ignore zeros)
        vols = [_zscore_inplace(v) for v in vols]
        # Stack to [C,D,H,W]
        img = np.stack(vols, axis=0)
        return img, mask

    # ------------------ Patch sampling / نمونه‌برداری پچ ------------------

    def _sample_patch(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        English: Ensure min size, then randomly crop a patch of `self.patch_size`.
        فارسی: حداقل اندازه را تضمین می‌کند و سپس یک پچ تصادفی با اندازه‌ی تعیین‌شده برش می‌دهد.
        """
        # Ensure minimum size
        img = np.stack([_ensure_min_size(img[c], self.patch_size) for c in range(img.shape[0])], axis=0)
        mask = _ensure_min_size(mask, self.patch_size)
        # Choose start
        start = _random_patch_start(mask.shape, self.patch_size, self.rng)
        z, y, x = start
        dz, dy, dx = self.patch_size
        # Crop
        img_patch = img[:, z:z+dz, y:y+dy, x:x+dx]
        msk_patch = mask[z:z+dz, y:y+dy, x:x+dx]
        return img_patch, msk_patch

    # ------------------ Debug synthetic / حالت دیباگ ------------------

    def _make_debug_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        img, mask = _make_synthetic_case(self.patch_size, C=len(self.modalities), rng=self.rng)
        return img, mask

    # ------------------ __getitem__ ------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if len(self.cases) == 0 and self.debug:
            img, msk = self._make_debug_sample()
        else:
            case_dir = self.cases[idx]
            img, msk = self._load_case(case_dir)
            img, msk = self._sample_patch(img, msk)

        sample: Dict[str, Any] = {"image": img, "mask": msk}
        if self.transforms is not None:
            sample = self.transforms(sample)

        # To torch tensors with correct dtypes
        image_t = torch.from_numpy(sample["image"]).float()          # [C,D,H,W]
        mask_t  = torch.from_numpy(sample["mask"]).long()            # [D,H,W]
        return {"image": image_t, "mask": mask_t}
