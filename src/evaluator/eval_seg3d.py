# English: Evaluation script for 3D tumor segmentation (Dice per-class & mean).
# فارسی: اسکریپت ارزیابی سگمنتیشن سه‌بعدی (دایس به‌ازای هر کلاس و میانگین).

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import json
import time

import torch
from torch.utils.data import DataLoader
from torch import amp  # new AMP API

from src.config.config import PATHS, CFG, get_device
from src.models.unet3d import UNet3D
from src.data_loader.brats_dataset import BratsPatches
from src.utils.metrics import dice_per_class, dice_mean

# -------------------- Helpers / توابع کمکی --------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _has_cases(root: Path) -> bool:
    return root.exists() and any(d.is_dir() for d in root.iterdir())

def _find_latest_ckpt() -> Optional[Path]:
    """
    English: Find the most recent checkpoint_best.pth (fallback to checkpoint_last.pth).
    فارسی: تازه‌ترین چک‌پوینت 'best' را پیدا می‌کند (در صورت نبود، 'last').
    """
    exp = PATHS.experiments
    if not exp.exists():
        return None
    candidates: List[Path] = []
    for run in exp.glob("*"):
        if not run.is_dir():
            continue
        best = run / "checkpoint_best.pth"
        last = run / "checkpoint_last.pth"
        if best.exists(): candidates.append(best)
        elif last.exists(): candidates.append(last)
    if not candidates:
        return None
    # pick most recent by mtime
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def _make_loader(debug: bool) -> Tuple[Optional[DataLoader], bool]:
    """
    English: Build validation DataLoader; fallback to synthetic if needed.
    فارسی: ساخت لودر اعتبارسنجی؛ در صورت نیاز به حالت مصنوعی سوییچ می‌کند.
    """
    val_root = PATHS.data_processed / "val"
    use_debug = debug or (not _has_cases(val_root))
    if use_debug:
        print("[INFO] Eval debug mode ON (synthetic samples).")
    ds_val = BratsPatches(root=val_root, patch_size=CFG.img_size, transforms=None,
                          debug=use_debug, debug_len=8)
    if len(ds_val) == 0:
        print(f"[ERR] No validation data at {val_root} and debug is OFF.")
        return None, use_debug
    pin = (get_device() == "cuda")
    loader = DataLoader(ds_val, batch_size=max(1, CFG.batch_size//2), shuffle=False,
                        num_workers=CFG.num_workers, pin_memory=pin)
    return loader, use_debug

# -------------------- Evaluation / ارزیابی --------------------

@torch.no_grad()
def evaluate(ckpt_path: Path, debug: bool = False, save_preds: Optional[Path] = None) -> Dict:
    """
    English:
      Loads checkpoint, runs inference on val loader, returns metrics dict.
    فارسی:
      چک‌پوینت را لود کرده، روی دادهٔ اعتبارسنجی پیش‌بینی می‌گیرد و متریک‌ها را برمی‌گرداند.
    """
    device_str = get_device()
    device = torch.device(device_str)
    use_amp = (device_str == "cuda") and CFG.amp

    loader, used_debug = _make_loader(debug)
    if loader is None:
        return {"ok": False, "reason": "no_validation_data"}

    # Model
    model = UNet3D(in_channels=CFG.in_channels, out_channels=CFG.out_channels, base=16).to(device)
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)  # tolerate raw state_dict
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if save_preds is not None:
        save_dir = _ensure_dir(save_preds)
    else:
        save_dir = None

    # Accumulators
    sum_dice_per_class = None
    n_batches = 0

    for i, batch in enumerate(loader):
        imgs = batch["image"].to(device, non_blocking=True)   # [B,C,D,H,W]
        masks = batch["mask"].to(device, non_blocking=True)   # [B,D,H,W]
        with amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)                              # [B,C,D,H,W]

        # metrics
        d_pc = dice_per_class(logits, masks)  # tensor [C-1] (bg ignored)
        sum_dice_per_class = d_pc if sum_dice_per_class is None else (sum_dice_per_class + d_pc)
        n_batches += 1

        # optional save predictions as npy (class argmax)
        if save_dir is not None:
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # [B,D,H,W]
            for b in range(preds.shape[0]):
                out_path = save_dir / f"pred_{i:03d}_{b:02d}.npy"
                preds_b = preds[b]
                import numpy as np
                np.save(out_path, preds_b)

    mean_dice_per_class = (sum_dice_per_class / max(1, n_batches)).cpu().tolist()
    mean_dice = sum(mean_dice_per_class) / max(1, len(mean_dice_per_class))

    result = {
        "ok": True,
        "device": device_str,
        "debug": used_debug,
        "ckpt": str(ckpt_path),
        "dice_per_class_noBG": mean_dice_per_class,  # order: classes 1..C-1
        "dice_mean_noBG": float(mean_dice),
        "batches": n_batches,
    }
    return result

# -------------------- CLI / رابط خط فرمان --------------------

def main(ckpt: Optional[str], debug: bool, save_preds: Optional[str]) -> int:
    if ckpt is None:
        found = _find_latest_ckpt()
        if found is None:
            print("[EXIT] No checkpoint provided and none found in experiments/.")
            return 0
        ckpt_path = found
        print(f"[INFO] Using latest checkpoint: {ckpt_path}")
    else:
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            print(f"[EXIT] Checkpoint not found: {ckpt_path}")
            return 0

    save_dir = Path(save_preds) if save_preds is not None else None
    if save_dir is not None:
        _ensure_dir(save_dir)

    t0 = time.time()
    res = evaluate(ckpt_path, debug=debug, save_preds=save_dir)
    dt = time.time() - t0

    # print & save JSON
    print(json.dumps(res, indent=2, ensure_ascii=False))
    out_dir = PATHS.experiments / "eval_reports"
    _ensure_dir(out_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out_dir / f"eval_{stamp}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Eval time: {dt:.1f}s | report: {out_dir / f'eval_{stamp}.json'}")
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="NeuroShield-3D evaluator")
    ap.add_argument("--ckpt", type=str, default=None, help="مسیر چک‌پوینت (.pth). اگر ندی از آخرین فایل experiments استفاده می‌شود.")
    ap.add_argument("--debug", action="store_true", help="اگر دادهٔ واقعی نیست با دادهٔ مصنوعی ارزیابی کن.")
    ap.add_argument("--save-preds", type=str, default=None, help="اگر بدهی، پیش‌بینی‌ها به صورت NPY اینجا ذخیره می‌شوند.")
    args = ap.parse_args()
    raise SystemExit(main(args.ckpt, args.debug, args.save_preds))
