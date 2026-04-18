import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import directed_hausdorff


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def denormalize(img):
    """
    Convert image from [-1, 1] to [0, 1]
    """
    return (img + 1.0) / 2.0


# ---------------------------------------------------------
# 1. PSNR
# ---------------------------------------------------------
def compute_psnr(pred, target, max_val=1.0):
    """
    Peak Signal-to-Noise Ratio

    PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(100.0)
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
    return psnr


# ---------------------------------------------------------
# 2. SSIM
# ---------------------------------------------------------
def compute_ssim(pred, target):
    """
    Structural Similarity Index (SSIM)
    Computed per image, averaged over batch
    """
    pred = denormalize(pred).clamp(0, 1)
    target = denormalize(target).clamp(0, 1)

    pred_np = pred.permute(0, 2, 3, 1).cpu().numpy()
    target_np = target.permute(0, 2, 3, 1).cpu().numpy()

    scores = []
    for i in range(pred_np.shape[0]):
        score = ssim(
            pred_np[i],
            target_np[i],
            channel_axis=-1,
            data_range=1.0
        )
        scores.append(score)

    return torch.tensor(scores).mean()

# ---------------------------------------------------------
# 5. Unified Evaluation Wrapper
# ---------------------------------------------------------
def evaluate_metrics(pred, target):
    """
    Returns all metrics in a dictionary
    """
    return {
        "PSNR": compute_psnr(pred, target).item(),
        "SSIM": compute_ssim(pred, target).item()
    }