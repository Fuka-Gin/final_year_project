import torch
import torch.nn as nn


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

    def __init__(self, channels):
        super().__init__()

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

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        return residual + out