# English: Central configuration for paths, training parameters, and dataset settings.
# فارسی: پیکربندی مرکزی مسیرها، پارامترهای آموزش و تنظیمات دیتاست.

from pathlib import Path
from pydantic import BaseModel

class Paths(BaseModel):
    """
    English: All important directory paths in the project.
    فارسی: مسیرهای مهم دایرکتوری‌های پروژه.
    """
    # Root directory of the project (two levels above this file)
    # مسیر ریشه پروژه (دو سطح بالاتر از این فایل)
    root: Path = Path(__file__).resolve().parents[2]
    
    # Raw NIfTI files (downloaded dataset)
    # فایل‌های خام NIfTI (دیتاست دانلود شده)
    data_raw: Path = root / "data" / "raw"
    
    # Preprocessed volumes (after skull-strip, resample, normalize)
    # داده‌های پیش‌پردازش‌شده (پس از حذف جمجمه، تغییر رزولوشن، نرمال‌سازی)
    data_interim: Path = root / "data" / "interim"
    
    # Patch datasets ready for training
    # پچ‌های آماده برای آموزش
    data_processed: Path = root / "data" / "processed"
    
    # Experiments (checkpoints, logs, configs)
    # نتایج آزمایش‌ها (مدل‌های ذخیره شده، لاگ‌ها، کانفیگ‌ها)
    experiments: Path = root / "experiments"

class TrainCfg(BaseModel):
    """
    English: Default training configuration.
    فارسی: پیکربندی پیش‌فرض آموزش.
    """
    img_size: tuple[int, int, int] = (128, 128, 128)  # 3D patch size / اندازه پچ سه‌بعدی
    in_channels: int = 4   # MRI modalities: T1, T1ce, T2, FLAIR / کانال‌های ورودی MRI
    out_channels: int = 3  # Tumor classes: WT, TC, ET / کلاس‌های خروجی تومور
    batch_size: int = 2    # Batch size / اندازه بچ
    lr: float = 2e-4       # Learning rate / نرخ یادگیری
    epochs: int = 100      # Number of training epochs / تعداد ایپاک‌ها
    amp: bool = True       # Automatic Mixed Precision / استفاده از AMP
    num_workers: int = 4   # DataLoader workers / تعداد پردازشگرهای DataLoader

def get_device() -> str:
    """
    English: Returns 'cuda' if GPU is available, otherwise 'cpu'.
    فارسی: اگر GPU موجود باشد 'cuda' برمی‌گرداند، در غیر این صورت 'cpu'.
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

# Instantiate configurations
# نمونه‌سازی پیکربندی‌ها
PATHS = Paths()
CFG = TrainCfg()
