# -*- coding: utf-8 -*-
"""
Organize BraTS-like dataset into nnUNet-style folders:
- imagesTr  ← all training modalities (flair/t1/t1ce/t2)
- labelsTr  ← all training labels (*_seg.*)
- imagesTs  ← validation/test images (no labels)

فارسی: این اسکریپت تمام فایل‌های .nii / .nii.gz را زیر Task01_BrainTumour
پیدا می‌کند و اگر نامشان شامل `_seg` بود به labelsTr می‌برد؛ در غیر این‌صورت
به imagesTr (برای Training) یا imagesTs (برای Validation/Test) منتقل می‌کند.
"""

from pathlib import Path
import shutil

# مسیر ریشه دیتاست (در صورت نیاز تغییر بده)
ROOT = Path(r"D:\NeuroShield-3D\data\raw\Task01_BrainTumour")

# مسیرهای مقصد
IMAGES_TR = ROOT / "imagesTr"
LABELS_TR = ROOT / "labelsTr"
IMAGES_TS = ROOT / "imagesTs"

# ساخت پوشه‌ها اگر نیستند
for p in (IMAGES_TR, LABELS_TR, IMAGES_TS):
    p.mkdir(parents=True, exist_ok=True)

# الگوهای منبع که معمولاً در BraTS داریم (اگر نباشند مشکلی نیست)
CANDIDATE_SOURCES = [
    ROOT,  # کل درخت را هم می‌گردیم
    ROOT / "BraTS2020_TrainingData",
    ROOT / "BraTS2020_ValidationData",
    ROOT / "train",
    ROOT / "val",
    ROOT / "validation",
    ROOT / "imagesTr",   # اگر قبلاً اشتباهی فایل‌ها اینجا ریخته شده باشند
]

def is_label(name: str) -> bool:
    """*_seg* → label"""
    return "_seg" in name.lower()

def is_nii(path: Path) -> bool:
    n = path.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")

def move_safe(src: Path, dst: Path):
    """انتقال امن: اگر مقصد وجود داشت، از رویش رد می‌شویم."""
    if dst.exists():
        print(f"⚠️  Exists, skip: {dst.name}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"→ Move: {src.relative_to(ROOT)}  ->  {dst.relative_to(ROOT)}")
    shutil.move(str(src), str(dst))

moved_imgs, moved_labs, moved_ts = 0, 0, 0

for base in CANDIDATE_SOURCES:
    if not base.exists():
        continue
    for src in base.rglob("*"):
        if not src.is_file():
            continue
        if not is_nii(src):
            continue

        # اگر همین حالا هم در مقصد درست است، رد شو
        if src.is_relative_to(IMAGES_TR) or src.is_relative_to(LABELS_TR) or src.is_relative_to(IMAGES_TS):
            continue

        name = src.name

        # Validation/Test را تشخیص بده (براساس نام پوشه‌ها)
        in_validation_path = any(part.lower() in {"validation", "val"} for part in src.parts)

        if is_label(name):
            dst = LABELS_TR / name
            move_safe(src, dst)
            moved_labs += 1
        else:
            if in_validation_path:
                dst = IMAGES_TS / name
                move_safe(src, dst)
                moved_ts += 1
            else:
                dst = IMAGES_TR / name
                move_safe(src, dst)
                moved_imgs += 1

print("\n✅ Done.")
print(f"  moved training images : {moved_imgs}")
print(f"  moved training labels : {moved_labs}")
print(f"  moved test/val images : {moved_ts}")
print(f"\nFinal dirs:")
print(f"  imagesTr: {len(list(IMAGES_TR.glob('*.nii*')))} files")
print(f"  labelsTr: {len(list(LABELS_TR.glob('*.nii*')))} files")
print(f"  imagesTs: {len(list(IMAGES_TS.glob('*.nii*')))} files")
