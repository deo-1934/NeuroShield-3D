# English: Minimal training loop (updated to torch.amp API).
# فارسی: لوپ آموزش مینیمال (به‌روزرسانی به API جدید torch.amp).

from __future__ import annotations
import os, json, time
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch import amp  # ✅ New AMP API / API جدید AMP

from src.config.config import CFG, PATHS, get_device
from src.utils.seed import set_seed
from src.utils.metrics import dice_mean
from src.models.unet3d import UNet3D
from src.losses.dice_focal import dice_ce_loss
from src.data_loader.brats_dataset import BratsPatches

# ------------------------- Helpers / توابع کمکی -------------------------

@dataclass
class TrainState:
    best_dice: float = -1.0
    epoch: int = 0
    global_step: int = 0

def _ensure_dir(p: Path) -> Path:
    """English: mkdir -p | فارسی: ساخت پوشه در صورت نبود."""
    p.mkdir(parents=True, exist_ok=True)
    return p

def _has_cases(root: Path) -> bool:
    """English: check if root has at least one case folder. | فارسی: وجود حداقل یک کیس را چک می‌کند."""
    return root.exists() and any(d.is_dir() for d in root.iterdir())

def make_dataloaders(debug: bool = False) -> Tuple[DataLoader, DataLoader, bool]:
    """
    English: Create train/val loaders; fall back to synthetic if debug or no data.
    فارسی: ساخت لودرهای آموزش/اعتبارسنجی؛ در نبود داده یا حالت دیباگ از نمونه مصنوعی استفاده می‌کند.
    """
    train_root = PATHS.data_processed / "train"
    val_root   = PATHS.data_processed / "val"

    use_debug = debug or (not _has_cases(train_root))
    if use_debug:
        print("[INFO] Debug mode is ON. Using synthetic samples (no real data found).")

    ds_train = BratsPatches(root=train_root, patch_size=CFG.img_size, transforms=None,
                            debug=use_debug, debug_len=32)
    ds_val   = BratsPatches(root=val_root,   patch_size=CFG.img_size, transforms=None,
                            debug=use_debug, debug_len=8)

    if len(ds_train) == 0:
        print(f"[WARN] Dataset is empty at {train_root}. Enable --debug or prepare data.")
        return None, None, use_debug

    pin = (get_device() == "cuda")
    loader_train = DataLoader(ds_train, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=pin)
    loader_val   = DataLoader(ds_val, batch_size=max(1, CFG.batch_size//2), shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=pin)
    return loader_train, loader_val, use_debug

def save_checkpoint(run_dir: Path, model: torch.nn.Module, state: TrainState, tag: str):
    """English: Save model/state. | فارسی: ذخیره مدل/وضعیت."""
    _ensure_dir(run_dir)
    ckpt = {
        "state_dict": model.state_dict(),
        "state": state.__dict__,
        "cfg": CFG.model_dump(),
        "paths": PATHS.model_dump()
    }
    torch.save(ckpt, run_dir / f"checkpoint_{tag}.pth")

# ------------------------- Train/Eval / آموزش و ارزیابی -------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, use_amp: bool) -> Tuple[float, float]:
    """
    English: Train one epoch; returns (avg_loss, avg_dice).
    فارسی: آموزش یک ایپاک؛ میانگین (لاست، دایس) را برمی‌گرداند.
    """
    model.train()
    total_loss, total_dice, n_batches = 0.0, 0.0, 0
    for batch in loader:
        imgs  = batch["image"].to(device, non_blocking=True)  # [B,C,D,H,W]
        masks = batch["mask"].to(device, non_blocking=True)   # [B,D,H,W]

        optimizer.zero_grad(set_to_none=True)
        # ✅ New AMP usage
        with amp.autocast("cuda", enabled=use_amp):            # English: autocast on CUDA
            logits = model(imgs)                               # فارسی: پیش‌بینی مدل
            loss   = dice_ce_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            d = dice_mean(logits, masks).item()

        total_loss += float(loss.detach().cpu())
        total_dice += d
        n_batches  += 1

    return total_loss / max(1, n_batches), total_dice / max(1, n_batches)

@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool) -> float:
    """English: Returns mean Dice on validation. | فارسی: دایس میانگین روی دادهٔ اعتبارسنجی."""
    model.eval()
    total_dice, n_batches = 0.0, 0
    for batch in loader:
        imgs  = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with amp.autocast("cuda", enabled=use_amp):
            logits = model(imgs)
        total_dice += dice_mean(logits, masks).item()
        n_batches  += 1
    return total_dice / max(1, n_batches)

# ------------------------- Main / اصلی -------------------------

def main(epochs: int = None, debug: bool = False, run_name: str | None = None) -> int:
    set_seed(42)
    dev_str = get_device()                   # "cuda" or "cpu"
    device  = torch.device(dev_str)
    use_amp = (dev_str == "cuda") and CFG.amp

    run_ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = run_name or f"run-{run_ts}"
    run_dir = PATHS.experiments / run_name
    _ensure_dir(run_dir)

    # Model/opt/scaler (✅ new GradScaler)
    model = UNet3D(in_channels=CFG.in_channels, out_channels=CFG.out_channels, base=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=1e-4)
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    # Data
    loader_train, loader_val, used_debug = make_dataloaders(debug=debug)
    if loader_train is None:
        print("[EXIT] No data available. Provide processed data or run with --debug.")
        return 0

    # Epochs
    E = int(epochs or (5 if used_debug else CFG.epochs))
    state = TrainState()

    # Save basic run info
    (run_dir / "meta.json").write_text(json.dumps({
        "device": str(device), "debug": used_debug, "epochs": E, "cfg": CFG.model_dump()
    }, indent=2), encoding="utf-8")

    print(f"[START] device={device}, epochs={E}, debug={used_debug}, run_dir={run_dir}, amp={use_amp}")

    for ep in range(1, E + 1):
        state.epoch = ep
        tr_loss, tr_dice = train_one_epoch(model, loader_train, optimizer, scaler, device, use_amp)
        val_dice = evaluate(model, loader_val, device, use_amp)

        msg = f"[E{ep:03d}] train_loss={tr_loss:.4f} train_dice={tr_dice:.4f} val_dice={val_dice:.4f}"
        print(msg)
        with open(run_dir / "log.txt", "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        if val_dice > state.best_dice:
            state.best_dice = val_dice
            save_checkpoint(run_dir, model, state, tag="best")

    save_checkpoint(run_dir, model, state, tag="last")
    print(f"[DONE] Best val Dice = {state.best_dice:.4f}. Checkpoints under: {run_dir}")
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="NeuroShield-3D trainer")
    ap.add_argument("--epochs", type=int, default=None, help="تعداد ایپاک‌ها (اگر ندی از CFG یا دیباگ استفاده میشه)")
    ap.add_argument("--debug", action="store_true", help="استفاده از داده مصنوعی برای تست سریع")
    ap.add_argument("--run-name", type=str, default=None, help="نام فولدر experiments")
    args = ap.parse_args()
    raise SystemExit(main(epochs=args.epochs, debug=args.debug, run_name=args.run_name))
