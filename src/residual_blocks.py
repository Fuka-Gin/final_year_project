import torch
import torch.nn as nn

class SEBlock(nn.Module):
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

class ResidualBlock(nn.Module):
    """
    Residual Block used in UcGAN Generator.

    Structure:
        Conv(3×3, C → C) + InstanceNorm + ReLU
        Conv(3×3, C → C) + InstanceNorm
        Skip Connection

    Notes:
    - InstanceNorm is preferred over BatchNorm for image-to-image translation
    - No dropout is used to preserve fine details
    - Padding preserves spatial resolution
    """

    def __init__(self, channels, use_attention=True, residual_scale=0.1):
        super().__init__()

        self.residual_scale = residual_scale
        self.use_attention  = use_attention

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

        if self.use_attention:
            out = self.attention(out)
        return residual + self.residual_scale * out