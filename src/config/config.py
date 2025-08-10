# src/config/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch

# پروژه: src/config/config.py  → دو سطح بالاتر = ریشه پروژه
PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class PATHS:
    root        : Path = PROJECT_ROOT
    data_root   : Path = PROJECT_ROOT / "data"
    data_raw    : Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    experiments : Path = PROJECT_ROOT / "experiments"
    logs        : Path = PROJECT_ROOT / "logs"

@dataclass
class CFG:
    in_channels : int = 4
    out_channels: int = 3  # اگر BraTS کامل می‌آوری و 4 کلاس می‌خواهی → 4
    img_size    : tuple[int,int,int] = (128,128,128)
    batch_size  : int = 1
    num_workers : int = 0  # روی ویندوز کم نگه دار
    epochs      : int = 20
    lr          : float = 2e-4
    amp         : bool = True  # اگر CUDA نیست هم مشکلی ندارد

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
