import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.models.feature_extraction import create_feature_extractor


# ---------------------------------------------------------
# 1. Adversarial Loss (GAN Loss)
# ---------------------------------------------------------
class GANLoss(nn.Module):
    """
    Adversarial loss for GAN training.

    Supported modes:
    - 'bce'   : Binary Cross Entropy (Vanilla GAN)
    - 'hinge' : Hinge Loss (default, more stable)
    """

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
            raise NotImplementedError(f"Unsupported GAN mode: {self.gan_mode}")


# ---------------------------------------------------------
# 2. Reconstruction Loss
# ---------------------------------------------------------
class ReconstructionLoss(nn.Module):
    """
    Pixel-wise reconstruction loss.
    - L1 : preferred for sharper images
    - L2 : smoother but may blur
    """

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
    """
    Perceptual loss using pretrained VGG19 feature maps.
    """

    def __init__(self, layers=None):
        super().__init__()

        if layers is None:
            layers = ["relu1_2", "relu2_2", "relu3_4", "relu4_4"]

        vgg = vgg19(weights="IMAGENET1K_V1").features
        vgg.eval()

        for p in vgg.parameters():
            p.requires_grad = False

        self.feature_extractor = create_feature_extractor(
            vgg,
            return_nodes={
                "3": "relu1_2",
                "8": "relu2_2",
                "17": "relu3_4",
                "26": "relu4_4"
            }
        )

        self.layers = layers
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        self.feature_extractor = self.feature_extractor.to(pred.device)
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)

        loss = 0.0
        for layer in self.layers:
            loss += self.criterion(pred_features[layer], target_features[layer])

        return loss


# ---------------------------------------------------------
# 4. Total Variation Loss
# ---------------------------------------------------------
class TotalVariationLoss(nn.Module):
    """
    Encourages spatial smoothness in generated images.
    """

    def forward(self, img):
        loss_h = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
        loss_v = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
        return loss_h + loss_v


# ---------------------------------------------------------
# 5. Unified Conditional GAN (UcGAN) Total Loss
# ---------------------------------------------------------
class UcGANLoss(nn.Module):
    """
    Combined loss for Unified Conditional GAN.

    Generator Objective:
        L_G = L_adv
            + λ_rec * L_rec
            + λ_perc * L_perc
            + λ_tv * L_tv
    """

    def __init__(
        self,
        gan_mode="hinge",
        recon_mode="l1",
        lambda_recon=100.0,
        lambda_perceptual=10.0,
        lambda_tv=0.0
    ):
        super().__init__()

        self.gan_loss = GANLoss(gan_mode)
        self.recon_loss = ReconstructionLoss(recon_mode)
        self.perceptual_loss = PerceptualLoss()
        self.tv_loss = TotalVariationLoss()

        self.lambda_recon = lambda_recon
        self.lambda_percep = lambda_perceptual
        self.lambda_tv = lambda_tv

    # ---------------------------
    # Generator Loss
    # ---------------------------
    def generator_loss(self, pred_fake, fake_img, real_img):
        adv = self.gan_loss(pred_fake, True)
        rec = self.recon_loss(fake_img, real_img)
        perc = self.perceptual_loss(fake_img, real_img)
        tv = self.tv_loss(fake_img)

        total = (
            adv
            + self.lambda_recon * rec
            + self.lambda_percep * perc
            + self.lambda_tv * tv
        )

        return total, {
            "adv": adv.item(),
            "recon": rec.item(),
            "perceptual": perc.item(),
            "tv": tv.item(),
            "total": total.item()
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
            "total": total.item()
        }