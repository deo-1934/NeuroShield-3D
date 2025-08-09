# English: NIfTI I/O utilities — safe helpers to load and save medical volumes.
# فارسی: توابع کمکی برای خواندن/نوشتن فایل‌های NIfTI — پیاده‌سازی امن و خوانا.

from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np

try:
    import nibabel as nib
except Exception as e:
    # English: If nibabel isn't installed, raise a clear error message.
    # فارسی: اگر nibabel نصب نباشد، پیام خطای واضح نمایش بده.
    raise ImportError(
        "nibabel is required for NIfTI I/O. Please install it via 'pip install nibabel'."
    ) from e


PathLike = Union[str, Path]


def is_nifti_file(path: PathLike) -> bool:
    """
    English: Returns True if path ends with .nii or .nii.gz
    فارسی: اگر پسوند فایل .nii یا .nii.gz باشد True برمی‌گرداند.
    """
    p = Path(path)
    return p.suffix == ".nii" or "".join(p.suffixes[-2:]) == ".nii.gz"


def _ensure_parent_dir(path: PathLike) -> None:
    """
    English: Creates parent directory if it doesn't exist.
    فارسی: اگر پوشهٔ والد وجود نداشته باشد، آن را می‌سازد.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def load_nii(path: PathLike, dtype: np.dtype = np.float32) -> Tuple[np.ndarray, np.ndarray, "nib.Nifti1Header"]:
    """
    English:
        Load a NIfTI file and return (array, affine, header).
        - array dtype defaults to float32
        - raises FileNotFoundError with a clear message if path doesn't exist

    فارسی:
        یک فایل NIfTI را می‌خوانَد و (آرایه، ماتریس افاین، هدر) برمی‌گرداند.
        - نوع دادهٔ آرایه پیش‌فرض float32 است
        - در صورت نبود فایل، FileNotFoundError با پیام خوانا می‌دهد.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[nii_io.load_nii] File not found: {p}")
    if not is_nifti_file(p):
        raise ValueError(f"[nii_io.load_nii] Not a NIfTI file (.nii or .nii.gz): {p}")

    img = nib.load(str(p))
    # Note: nibabel.get_fdata(dtype=...) returns float64 if dtype not provided; we enforce dtype.
    data = img.get_fdata().astype(dtype, copy=False)
    # Make sure data is contiguous for downstream libs (PyTorch/NumPy expectations)
    data = np.ascontiguousarray(data)
    affine = img.affine
    header = img.header
    return data, affine, header


def save_nii(
    array: np.ndarray,
    out_path: PathLike,
    affine: Optional[np.ndarray] = None,
    header: Optional["nib.Nifti1Header"] = None,
    dtype: np.dtype = np.float32,
) -> None:
    """
    English:
        Save a numpy array to NIfTI at out_path.
        - If affine/header are None, use identity affine and a default header.
        - Casts array to the provided dtype (default float32).
        - Creates parent directories if needed.

    فارسی:
        یک آرایه نامپای را به فایل NIfTI در out_path ذخیره می‌کند.
        - اگر affine/header داده نشود، از افاین واحد و هدر پیش‌فرض استفاده می‌کند.
        - آرایه را به نوع دادهٔ موردنظر (پیش‌فرض float32) تبدیل می‌کند.
        - پوشه‌های والد در صورت نیاز ساخته می‌شوند.
    """
    _ensure_parent_dir(out_path)
    arr = np.asarray(array).astype(dtype, copy=False)
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    img = nib.Nifti1Image(arr, affine, header=header)
    nib.save(img, str(out_path))


def save_like(
    array: np.ndarray,
    reference_path: PathLike,
    out_path: PathLike,
    dtype: np.dtype = np.float32,
) -> None:
    """
    English:
        Save 'array' using affine/header copied from a reference NIfTI.
        Useful to keep geometry consistent with the input volume.

    فارسی:
        آرایه را با استفاده از affine و header فایل مرجع ذخیره می‌کند.
        برای حفظ سازگاری هندسی با حجم ورودی کاربردی است.
    """
    _, ref_affine, ref_header = load_nii(reference_path)
    save_nii(array, out_path, affine=ref_affine, header=ref_header, dtype=dtype)
