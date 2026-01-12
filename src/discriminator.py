import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Unified Conditional PatchGAN Discriminator

    Architecture:
    - Input: (B, C + T, H, W)
        C : image channels
        T : number of tasks (one-hot task vector)
    - PatchGAN discriminator with 70x70 receptive field
    - Fully convolutional (outputs a patch-wise realism map)
    """

    def __init__(self, in_channels=3, num_tasks=4):
        super(Discriminator, self).__init__()

        self.num_tasks = num_tasks

        def disc_block(in_ch, out_ch, stride=2, normalize=True):
            layers = [
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=stride,
                    padding=1,
                    bias=not normalize
                )
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # PatchGAN discriminator layers
        self.model = nn.Sequential(
            # Input: (C + T) x H x W
            *disc_block(in_channels + num_tasks, 64, normalize=False),   # 64 x H/2 x W/2
            *disc_block(64, 128),                                         # 128 x H/4 x W/4
            *disc_block(128, 256),                                        # 256 x H/8 x W/8
            *disc_block(256, 512, stride=1),                              # 512 x H/8 x W/8

            # Output patch map
            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=1
            )
        )

    def forward(self, x, task_vector):
        """
        Args:
            x (Tensor): Input image of shape (B, C, H, W)
            task_vector (Tensor): One-hot task vector of shape (B, T)

        Returns:
            Tensor: PatchGAN output of shape (B, 1, H', W')
        """

        B, _, H, W = x.shape

        # Expand task vector spatially and concatenate
        task_map = task_vector.view(B, self.num_tasks, 1, 1)
        task_map = task_map.expand(-1, -1, H, W)
        x = torch.cat([x, task_map], dim=1)

        return self.model(x)