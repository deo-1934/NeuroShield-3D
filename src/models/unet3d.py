# English: Minimal, production-ready 3D U-Net (correct skip mapping: up2↔x2, up1↔x1).
# فارسی: U-Net سه‌بعدی مینیمال و آماده استفاده (اصلاح اتصال اسکیپ‌ها: up2 با x2، up1 با x1).

from __future__ import annotations
from typing import Literal
import torch
import torch.nn as nn

# ----------------------- Building Blocks / بلوک‌های سازنده -----------------------

def _norm3d(num_features: int, kind: Literal["instance","batch"]="instance") -> nn.Module:
    """English: Choose normalization layer. | فارسی: انتخاب لایه نرمال‌سازی."""
    return nn.InstanceNorm3d(num_features) if kind == "instance" else nn.BatchNorm3d(num_features)

class DoubleConv(nn.Module):
    """English: (Conv3D→Norm→ReLU)×2. | فارسی: دابل‌کانولوشن با نرمال‌سازی و رلو."""
    def __init__(self, in_ch: int, out_ch: int, norm: str = "instance", p_drop: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            _norm3d(out_ch, norm), nn.ReLU(inplace=True),
            nn.Dropout3d(p_drop) if p_drop > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            _norm3d(out_ch, norm), nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.block(x)

class Down(nn.Module):
    """English: MaxPool↓ + DoubleConv. | فارسی: مکس‌پول سپس دابل‌کانولوشن."""
    def __init__(self, in_ch: int, out_ch: int, norm: str = "instance", p_drop: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv(in_ch, out_ch, norm=norm, p_drop=p_drop)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.conv(self.pool(x))

class Up(nn.Module):
    """
    English: UpConv→concat skip→DoubleConv. UpConv outputs `out_ch`.
    فارسی: آپ‌کانولوشن→اتصال اسکیپ→دابل‌کانولوشن. خروجی آپ‌کانولوشن `out_ch` است.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str = "instance", p_drop: float = 0.0):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, norm=norm, p_drop=p_drop)

    @staticmethod
    def _center_crop_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """English: Center-crop x to ref spatial shape. | فارسی: کراپ مرکزی x به شکل ref."""
        _, _, Dx, Hx, Wx = x.shape; _, _, Dr, Hr, Wr = ref.shape
        sx = max((Dx-Dr)//2, 0); sy = max((Hx-Hr)//2, 0); sz = max((Wx-Wr)//2, 0)
        return x[:, :, sx:sx+Dr, sy:sy+Hr, sz:sz+Wr]

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align spatial sizes (handles odd dims): crop the larger one
        if x.shape[2:] != skip.shape[2:]:
            if all(a >= b for a, b in zip(skip.shape[2:], x.shape[2:])):  # skip بزرگ‌تر
                skip = self._center_crop_to(skip, x)
            else:  # x بزرگ‌تر
                x = self._center_crop_to(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))

# ----------------------- UNet3D / مدل اصلی -----------------------

class UNet3D(nn.Module):
    """
    English: 3-level 3D U-Net (↓×2, bottleneck, ↑×2). Good for 128³ patches.
    فارسی: U-Net سه‌سطحی (دو داون، باتلنک، دو آپ). مناسب پچ‌های 128³.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3, base: int = 16,
                 norm: Literal["instance","batch"]="instance", p_drop: float = 0.0):
        super().__init__()
        # Encoder
        self.inc   = DoubleConv(in_channels, base,   norm=norm, p_drop=p_drop)   # x1: base,  D
        self.down1 = Down(base,   base*2, norm=norm, p_drop=p_drop)              # x2: 2b,   D/2
        self.down2 = Down(base*2, base*4, norm=norm, p_drop=p_drop)              # x3: 4b,   D/4
        # Bottleneck
        self.bot   = DoubleConv(base*4, base*8, norm=norm, p_drop=p_drop)        # xb: 8b,   D/4
        # Decoder (correct skip mapping)
        self.up2   = Up(in_ch=base*8, skip_ch=base*2, out_ch=base*4, norm=norm, p_drop=p_drop)  # ↔ x2
        self.up1   = Up(in_ch=base*4, skip_ch=base,   out_ch=base*2, norm=norm, p_drop=p_drop)  # ↔ x1
        self.outc  = nn.Conv3d(base*2, out_channels, kernel_size=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """English: He init for convs; zeros bias. | فارسی: مقداردهی He برای کانولوشن‌ها."""
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)     # D
        x2 = self.down1(x1)  # D/2
        x3 = self.down2(x2)  # D/4
        xb = self.bot(x3)    # D/4
        x  = self.up2(xb, x2)  # → D/2
        x  = self.up1(x,  x1)  # → D
        return self.outc(x)

    def num_parameters(self) -> int:
        """English: Number of trainable params. | فارسی: تعداد پارامترهای قابل‌آموزش."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

__all__ = ["UNet3D"]

# ----------------------- Sanity Test / تست سریع -----------------------
if __name__ == "__main__":
    # English: Run `python -m src.models.unet3d`
    # فارسی: اجرا: `python -m src.models.unet3d`
    model = UNet3D(in_channels=4, out_channels=3, base=8)
    x = torch.randn(1, 4, 64, 64, 64)
    y = model(x)
    print("in:", tuple(x.shape), "out:", tuple(y.shape), "params:", model.num_parameters())
    assert y.shape == (1, 3, 64, 64, 64), "Output shape mismatch!"
    print("✔ Sanity OK")
