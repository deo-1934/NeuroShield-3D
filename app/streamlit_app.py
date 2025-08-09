# Minimal Streamlit demo for NeuroShield-3D (English-only UI)

from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch
from torch import amp

from src.config.config import PATHS, CFG, get_device
from src.models.unet3d import UNet3D
from src.data_loader.brats_dataset import BratsPatches
from src.utils.viz3d import take_slice

# --------------------------- Helpers ---------------------------

def find_latest_ckpt() -> Optional[Path]:
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
    ds = BratsPatches(root=PATHS.data_processed / "val",
                      patch_size=CFG.img_size, debug=True, debug_len=1)
    batch = ds[0]
    img = batch["image"].numpy()  # [C,D,H,W]
    msk = batch["mask"].numpy()   # [D,H,W]
    return img, msk

def run_inference(model: UNet3D, image: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    x = torch.from_numpy(image[None, ...]).to(device)  # [1,C,D,H,W]
    use_amp = (device.type == "cuda") and CFG.amp
    with torch.no_grad(), amp.autocast("cuda", enabled=use_amp):
        logits = model(x)                               # [1,C,D,H,W]
        preds = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.int16)
    return preds

def plot_overlay(slice_img: np.ndarray, slice_mask: np.ndarray | None, title: str):
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(slice_img, cmap="gray")
    if slice_mask is not None:
        overlay = np.ma.masked_where(slice_mask == 0, slice_mask)
        plt.imshow(overlay, alpha=0.4)
    plt.axis("off")
    plt.title(title)
    st.pyplot(fig, clear_figure=True)

# --------------------------- UI ---------------------------

st.set_page_config(page_title="NeuroShield-3D Demo", layout="wide")
st.title("🧠 NeuroShield-3D • MRI Tumor Segmentation (Demo)")

with st.sidebar:
    st.header("Settings")
    axis_name = st.selectbox("Slice axis", ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"], index=0)
    axis = {"Axial (Z)":2, "Coronal (Y)":1, "Sagittal (X)":0}[axis_name]  # image is [C,D,H,W]
    chan_name = st.selectbox("Modality channel", ["T1", "T1ce", "T2", "FLAIR"], index=3)
    chan = ["T1","T1ce","T2","FLAIR"].index(chan_name)

st.markdown("This demo currently uses a **synthetic sample (debug)**. NIfTI/ZIP upload will be added soon.")

model, status = load_model()
st.caption(status)

img, msk = get_debug_sample()               # img [C,D,H,W], msk [D,H,W]
shape = img.shape[1:]                       # (D,H,W)
st.write(f"Patch shape: **{shape}**  |  Channels: **{img.shape[0]}**")

max_index = shape[axis] - 1
idx = st.slider("Slice index", min_value=0, max_value=int(max_index), value=int(max_index//2), step=1)

vol = img[chan]                    # [D,H,W]
sl_img = take_slice(vol, axis=axis, index=idx)
sl_gt  = take_slice(msk, axis=axis, index=idx)

col1, col2, col3 = st.columns(3)

with col1:
    plot_overlay(sl_img, None, f"{chan_name} • Slice {idx}")

if st.button("Run Inference", type="primary"):
    preds = run_inference(model, img)      # [D,H,W]
    sl_pr = take_slice(preds, axis=axis, index=idx)
    with col2:
        plot_overlay(sl_img, sl_gt, "Ground Truth")
    with col3:
        plot_overlay(sl_img, sl_pr, "Prediction")
else:
    with col2:
        plot_overlay(sl_img, sl_gt, "Ground Truth")
    with col3:
        st.info("Click 'Run Inference' to view the prediction.")
