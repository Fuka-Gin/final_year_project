import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.models.feature_extraction import create_feature_extractor


# ---------------------------------------------------------
# 1. Adversarial Loss
# ---------------------------------------------------------
class GANLoss(nn.Module):
    def __init__(self, gan_mode="hinge"):
        super().__init__()
        self.gan_mode = gan_mode

        if gan_mode == "bce":
            self.loss = nn.BCEWithLogitsLoss()

    def forward(self, prediction, target_is_real):
        if self.gan_mode == "bce":
            target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
            return self.loss(prediction, target)

        elif self.gan_mode == "hinge":
            if target_is_real:
                return torch.mean(F.relu(1.0 - prediction))
            else:
                return torch.mean(F.relu(1.0 + prediction))

        else:
            raise NotImplementedError


# ---------------------------------------------------------
# 2. Reconstruction Loss
# ---------------------------------------------------------
class ReconstructionLoss(nn.Module):
    def __init__(self, mode="l1"):
        super().__init__()
        if mode == "l1":
            self.loss = nn.L1Loss()
        elif mode == "l2":
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Reconstruction loss must be 'l1' or 'l2'")

    def forward(self, pred, target):
        return self.loss(pred, target)


# ---------------------------------------------------------
# 3. Perceptual Loss (VGG19)
# ---------------------------------------------------------
class PerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super().__init__()

        if layers is None:
            layers = ["relu1_2", "relu2_2", "relu3_4", "relu4_4"]

        vgg = vgg19(weights="IMAGENET1K_V1").features
        vgg.eval()

        for p in vgg.parameters():
            p.requires_grad = False

        self.extractor = create_feature_extractor(
            vgg,
            return_nodes={
                "3": "relu1_2",
                "8": "relu2_2",
                "17": "relu3_4",
                "26": "relu4_4",
            },
        )

        self.layers = layers
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        pred_f = self.extractor(pred)
        target_f = self.extractor(target)

        loss = 0.0
        for layer in self.layers:
            loss += self.criterion(pred_f[layer], target_f[layer])

        return loss


# ---------------------------------------------------------
# 4. Total Variation Loss
# ---------------------------------------------------------
class TotalVariationLoss(nn.Module):
    def forward(self, img):
        loss_h = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
        loss_v = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
        return loss_h + loss_v


# ---------------------------------------------------------
# 5. Multi-Scale Loss
# ---------------------------------------------------------
def multi_scale_loss(pred, target):
    loss = 0.0

    # Full scale
    loss += F.l1_loss(pred, target)

    # Half scale
    pred_half = F.interpolate(pred, scale_factor=0.5, mode="bilinear", align_corners=False)
    target_half = F.interpolate(target, scale_factor=0.5, mode="bilinear", align_corners=False)
    loss += F.l1_loss(pred_half, target_half)

    # Quarter scale
    pred_quarter = F.interpolate(pred, scale_factor=0.25, mode="bilinear", align_corners=False)
    target_quarter = F.interpolate(target, scale_factor=0.25, mode="bilinear", align_corners=False)
    loss += F.l1_loss(pred_quarter, target_quarter)

    return loss


# ---------------------------------------------------------
# 6. Edge-Aware Loss (Sobel)
# ---------------------------------------------------------
def edge_loss(pred, target):
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=pred.device)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=pred.device)

    sobel_x = sobel_x.view(1,1,3,3)
    sobel_y = sobel_y.view(1,1,3,3)

    pred_gray = pred.mean(1, keepdim=True)
    target_gray = target.mean(1, keepdim=True)

    pred_edge = F.conv2d(pred_gray, sobel_x, padding=1) + \
                F.conv2d(pred_gray, sobel_y, padding=1)

    target_edge = F.conv2d(target_gray, sobel_x, padding=1) + \
                  F.conv2d(target_gray, sobel_y, padding=1)

    return F.l1_loss(pred_edge, target_edge)


# ---------------------------------------------------------
# 7. Frequency Loss (FFT-based)
# ---------------------------------------------------------
def frequency_loss(pred, target):
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)

    return torch.mean(torch.abs(pred_fft - target_fft))


# ---------------------------------------------------------
# 8. Unified Conditional GAN Loss
# ---------------------------------------------------------
class UcGANLoss(nn.Module):
    def __init__(
        self,
        gan_mode="hinge",
        recon_mode="l1",
        lambda_recon=100.0,
        lambda_perceptual=10.0,
        lambda_tv=0.0,
        lambda_edge=5.0,
        lambda_ms=1.0,
        lambda_freq=0.1
    ):
        super().__init__()

        self.gan_loss = GANLoss(gan_mode)
        self.recon_loss = ReconstructionLoss(recon_mode)
        self.perceptual_loss = PerceptualLoss()
        self.tv_loss = TotalVariationLoss()

        self.lambda_recon = lambda_recon
        self.lambda_percep = lambda_perceptual
        self.lambda_tv = lambda_tv
        self.lambda_edge = lambda_edge
        self.lambda_ms = lambda_ms
        self.lambda_freq = lambda_freq

    # ---------------------------
    # Generator Loss
    # ---------------------------
    def generator_loss(self, pred_fake, fake_img, real_img):

        adv = self.gan_loss(pred_fake, True)
        rec = self.recon_loss(fake_img, real_img)
        perc = self.perceptual_loss(fake_img, real_img)
        tv = self.tv_loss(fake_img)
        edge = edge_loss(fake_img, real_img)
        ms = multi_scale_loss(fake_img, real_img)
        freq = frequency_loss(fake_img, real_img)

        total = (
            adv
            + self.lambda_recon * rec
            + self.lambda_percep * perc
            + self.lambda_tv * tv
            + self.lambda_edge * edge
            + self.lambda_ms * ms
            + self.lambda_freq * freq
        )

        return total, {
            "adv": adv.item(),
            "recon": rec.item(),
            "perceptual": perc.item(),
            "tv": tv.item(),
            "edge": edge.item(),
            "multiscale": ms.item(),
            "frequency": freq.item(),
            "total": total.item(),
        }

    # ---------------------------
    # Discriminator Loss
    # ---------------------------
    def discriminator_loss(self, pred_real, pred_fake):
        real_loss = self.gan_loss(pred_real, True)
        fake_loss = self.gan_loss(pred_fake, False)
        total = real_loss + fake_loss

        return total, {
            "real": real_loss.item(),
            "fake": fake_loss.item(),
            "total": total.item(),
        }