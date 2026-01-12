import torch
import torch.nn as nn
from residual_blocks import ResidualBlock


class Generator(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        num_tasks=1,
        num_residual_blocks=8
    ):
        super().__init__()
        self.num_tasks = num_tasks

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels + num_tasks, 64, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )

        self.down1 = self._down(64, 128)
        self.down2 = self._down(128, 256)

        self.residuals = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residual_blocks)]
        )

        self.up1 = self._up(256, 128)
        self.up2 = self._up(128, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, 7, 1, 3),
            nn.Tanh()
        )

    def _down(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True),
        )

    def _up(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, task_vector):
        b, _, h, w = x.shape
        task_map = task_vector.view(b, self.num_tasks, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([x, task_map], dim=1)

        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.residuals(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.final(x)