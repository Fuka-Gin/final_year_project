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
# 3. F-Score (Edge-based)
# ---------------------------------------------------------
def compute_fscore(pred, target, threshold=0.5):
    """
    Edge-based F1 Score (useful for structure preservation)
    """

    pred = denormalize(pred).mean(dim=1)
    target = denormalize(target).mean(dim=1)

    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1


# ---------------------------------------------------------
# 4. Hausdorff Distance
# ---------------------------------------------------------
def compute_hausdorff(pred, target, threshold=0.5):
    """
    Hausdorff Distance between edge maps
    """

    pred = denormalize(pred).mean(dim=1)
    target = denormalize(target).mean(dim=1)

    pred_bin = (pred > threshold).cpu().numpy()
    target_bin = (target > threshold).cpu().numpy()

    distances = []

    for i in range(pred_bin.shape[0]):
        pred_pts = np.argwhere(pred_bin[i])
        target_pts = np.argwhere(target_bin[i])

        if len(pred_pts) == 0 or len(target_pts) == 0:
            distances.append(0.0)
        else:
            d1 = directed_hausdorff(pred_pts, target_pts)[0]
            d2 = directed_hausdorff(target_pts, pred_pts)[0]
            distances.append(max(d1, d2))

    return torch.tensor(distances).mean()


# ---------------------------------------------------------
# 5. Unified Evaluation Wrapper
# ---------------------------------------------------------
def evaluate_metrics(pred, target):
    """
    Returns all metrics in a dictionary
    """
    return {
        "PSNR": compute_psnr(pred, target).item(),
        "SSIM": compute_ssim(pred, target).item(),
        "F-Score": compute_fscore(pred, target).item(),
        "Hausdorff": compute_hausdorff(pred, target).item()
    }