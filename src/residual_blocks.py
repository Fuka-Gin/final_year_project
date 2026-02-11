import torch
import torch.nn as nn


# ---------------------------------------------------------
# 1. Squeeze-and-Excitation (Channel Attention)
# ---------------------------------------------------------
class SEBlock(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation)

    Helps model focus on informative feature channels.
    Especially useful for deblurring fine textures.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc(scale)
        return x * scale


# ---------------------------------------------------------
# 2. Enhanced Residual Block
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    Enhanced Residual Block for UcGAN Generator

    Structure:
        Conv → IN → ReLU
        Conv → IN
        Channel Attention (SE)
        Residual Scaling
        Skip Connection

    Improvements over basic block:
    - Channel attention improves feature focus
    - Residual scaling stabilizes GAN training
    - Better texture recovery for deblurring
    """

    def __init__(self, channels, use_attention=True, residual_scale=0.1):
        super().__init__()

        self.residual_scale = residual_scale
        self.use_attention = use_attention

        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)

        if self.use_attention:
            self.attention = SEBlock(channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        # Apply attention if enabled
        if self.use_attention:
            out = self.attention(out)

        # Residual scaling improves GAN stability
        out = residual + self.residual_scale * out

        return out