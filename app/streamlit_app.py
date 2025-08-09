# English: Minimal Streamlit demo for NeuroShield-3D (debug-friendly).
# فارسی: دمو مینیمال استریملیت برای NeuroShield-3D (سازگار با حالت دیباگ).

from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch
from torch import amp  # for mixed precision

# Project imports
from src.config.config import PATHS, CFG, get_device
from src.models.unet3d import UNet3D
from src.data_loader.brats_dataset import BratsPatches
from src.utils.viz3d import take_slice

# --------------------------- Helpers / توابع کمکی ---------------------------

def find_latest_ckpt() -> Optional[Path]:
    """
    English: Find most recent checkpoint_best.pth (fallback to checkpoint_last.pth).
    فارسی: جدیدترین چک‌پوینت best را پیدا می‌کند (در نبود، last).
    """
    exp = PATHS.experiments
    if not exp.exists():
        return None
    cands: List[Path] = []
    for run in exp.glob("*"):
        if not run.is_dir():
            continue
        best = run / "checkpoint_best.pth"
        last = run / "checkpoint_last.pth"
        if best.exists(): cands.append(best)
        elif last.exists(): cands.append(last)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

@st.cache_resource(show_spinner=False)
def load_model() -> tuple[UNet3D, str]:
    """
    English: Build model and load latest checkpoint if present.
    فارسی: مدل را می‌سازد و اگر چک‌پوینتی باشد آن را بارگذاری می‌کند.
    """
    device_str = get_device()
    device = torch.device(device_str)
    model = UNet3D(in_channels=CFG.in_channels, out_channels=CFG.out_channels, base=16).to(device)
    ckpt = find_latest_ckpt()
    if ckpt is not None and ckpt.exists():
        data = torch.load(ckpt, map_location=device)
        state = data.get("state_dict", data)
        try:
            model.load_state_dict(state, strict=True)
            status = f"✅ Loaded checkpoint: {ckpt.name}"
        except Exception as e:
            status = f"⚠️ Could not load checkpoint ({ckpt.name}): {e}"
    else:
        status = "⚠️ No checkpoint found — using random weights."
    model.eval()
    return model, status

def get_debug_sample() -> tuple[np.ndarray, np.ndarray]:
    """
    English: Create a synthetic sample using BratsPatches(debug=True).
    فارسی: تولید نمونهٔ مصنوعی با BratsPatches در حالت دیباگ.
    """
    ds = BratsPatches(root=PATHS.data_processed / "val",
                      patch_size=CFG.img_size, debug=True, debug_len=1)
    batch = ds[0]
    img = batch["image"].numpy()  # [C,D,H,W]
    msk = batch["mask"].numpy()   # [D,H,W]
    return img, msk

def run_inference(model: UNet3D, image: np.ndarray) -> np.ndarray:
    """
    English: Run model inference on one sample image [C,D,H,W] -> preds [D,H,W].
    فارسی: اجرای پیش‌بینی مدل روی یک نمونه [C,D,H,W] و برگرداندن ماسک کلاس‌ها [D,H,W].
    """
    device = next(model.parameters()).device
    x = torch.from_numpy(image[None, ...]).to(device)  # [1,C,D,H,W]
    use_amp = (device.type == "cuda") and CFG.amp
    with torch.no_grad(), amp.autocast("cuda", enabled=use_amp):
        logits = model(x)                               # [1,C,D,H,W]
        preds = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int16)
    return preds

def plot_overlay(slice_img: np.ndarray, slice_mask: np.ndarray | None, title: str):
    """
    English: Show grayscale slice with optional mask overlay.
    فارسی: نمایش اسلایس خاکستری با امکان اورلی ماسک.
    """
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(slice_img, cmap="gray")
    if slice_mask is not None:
        # show mask (non-zero) transparently
        overlay = np.ma.masked_where(slice_mask == 0, slice_mask)
        plt.imshow(overlay, alpha=0.4)
    plt.axis("off")
    plt.title(title)
    st.pyplot(fig, clear_figure=True)

# --------------------------- UI / رابط کاربری ---------------------------

st.set_page_config(page_title="NeuroShield-3D Demo", layout="wide")
st.title("🧠 NeuroShield-3D • MRI Tumor Segmentation (Demo)")

with st.sidebar:
    st.header("Settings / تنظیمات")
    axis_name = st.selectbox("Slice axis / محور اسلایس", ["Z (axial)", "Y (coronal)", "X (sagittal)"], index=0)
    axis = {"Z (axial)":2, "Y (coronal)":1, "X (sagittal)":0}[axis_name]  # our array is [C,D,H,W]
    chan_name = st.selectbox("Modality channel / کانال ورودی", ["T1", "T1ce", "T2", "FLAIR"], index=3)
    chan = ["T1","T1ce","T2","FLAIR"].index(chan_name)

st.markdown("این دمو فعلاً با **نمونهٔ مصنوعی (Debug)** کار می‌کند. به‌زودی ورودی NIfTI/ZIP هم اضافه می‌کنیم.")

# Load model once
model, status = load_model()
st.caption(status)

# Prepare sample (debug)
img, msk = get_debug_sample()               # img [C,D,H,W], msk [D,H,W]
shape = img.shape[1:]                       # (D,H,W)
st.write(f"Patch shape: **{shape}**  |  Channels: **{img.shape[0]}**")

# Slice selector
max_index = shape[axis] - 1
idx = st.slider("Slice index / شماره اسلایس", min_value=0, max_value=int(max_index), value=int(max_index//2), step=1)

# Extract slices
# English: image is [C,D,H,W]; convert axis to (D,H,W) order indexing.
# فارسی: تصویر [C,D,H,W] است؛ اندیس‌گذاری روی (D,H,W).
vol = img[chan]                    # [D,H,W]
sl_img = take_slice(vol, axis=axis, index=idx)
sl_gt  = take_slice(msk, axis=axis, index=idx)

col1, col2, col3 = st.columns(3)

with col1:
    plot_overlay(sl_img, None, f"{chan_name} • Slice {idx}")

# Run inference button
if st.button("Run Inference / اجرای پیش‌بینی", type="primary"):
    preds = run_inference(model, img)      # [D,H,W]
    sl_pr = take_slice(preds, axis=axis, index=idx)
    with col2:
        plot_overlay(sl_img, sl_gt, "Ground Truth / ماسک واقعی (دیباگ)")
    with col3:
        plot_overlay(sl_img, sl_pr, "Prediction / پیش‌بینی مدل")
else:
    with col2:
        plot_overlay(sl_img, sl_gt, "Ground Truth / ماسک واقعی (دیباگ)")
    with col3:
        st.info("برای نمایش پیش‌بینی، دکمهٔ «Run Inference» را بزنید.")

st.caption("Tip: اگر چک‌پوینت در `experiments/` باشد، مدل به‌طور خودکار لود می‌شود.")
