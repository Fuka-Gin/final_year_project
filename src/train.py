import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from generator import Generator
from discriminator import Discriminator
from loss_functions import UcGANLoss
from dataset import UcGANDataset
from metrics import evaluate_metrics

# ---------------------------------------------------------
# Device & Paths
# ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "paired", "deblur")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "results", "samples")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# ---------------------------------------------------------
# Feature Configuration (DO NOT CHANGE LOGIC)
# ---------------------------------------------------------
FEATURE_CONFIG = {
    "deblur": {
        "id": 0,
        "gan_mode": "hinge",
        "recon_mode": "l2",
        "lambda_recon": 70,
        "lambda_perceptual": 10,
        "lambda_tv": 2,
    }
}

FEATURE = "deblur"
TASK_ID = FEATURE_CONFIG[FEATURE]["id"]
NUM_TASKS = len(FEATURE_CONFIG)

# ---------------------------------------------------------
# Task Vector
# ---------------------------------------------------------
def get_task_vector(batch_size, task_id):
    task = torch.zeros(batch_size, NUM_TASKS, device=DEVICE)
    task[:, task_id] = 1.0
    return task

# ---------------------------------------------------------
# Loss Factory
# ---------------------------------------------------------
def create_loss_function():
    cfg = FEATURE_CONFIG[FEATURE]
    return UcGANLoss(
        gan_mode=cfg["gan_mode"],
        recon_mode=cfg["recon_mode"],
        lambda_recon=cfg["lambda_recon"],
        lambda_perceptual=cfg["lambda_perceptual"],
        lambda_tv=cfg["lambda_tv"],
    )

# ---------------------------------------------------------
# Checkpoint Utilities
# ---------------------------------------------------------
def save_checkpoint(epoch, G, D, g_opt, d_opt):
    torch.save(
        {
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "g_opt": g_opt.state_dict(),
            "d_opt": d_opt.state_dict(),
        },
        os.path.join(CHECKPOINT_DIR, "last.pth"),
    )

def load_checkpoint(G, D, g_opt, d_opt):
    path = os.path.join(CHECKPOINT_DIR, "last.pth")
    if not os.path.exists(path):
        return 0

    checkpoint = torch.load(path, map_location=DEVICE)
    G.load_state_dict(checkpoint["G"])
    D.load_state_dict(checkpoint["D"])
    g_opt.load_state_dict(checkpoint["g_opt"])
    d_opt.load_state_dict(checkpoint["d_opt"])

    print(f"🔁 Resuming training from epoch {checkpoint['epoch'] + 1}")
    return checkpoint["epoch"] + 1

# ---------------------------------------------------------
# Training Function
# ---------------------------------------------------------
def train(
    epochs=150,
    batch_size=16,
    lr=2e-4,
):
    # --------------------
    # Dataset
    # --------------------

    train_ds = UcGANDataset(DATA_ROOT, mode="train")
    val_ds = UcGANDataset(DATA_ROOT, mode="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # --------------------
    # Models
    # --------------------
    G = Generator(num_tasks=NUM_TASKS).to(DEVICE)
    D = Discriminator(num_tasks=NUM_TASKS).to(DEVICE)
    print("Generator and Discriminator initialized.")
    # --------------------
    # Loss & Optimizers
    # --------------------
    criterion = create_loss_function()

    g_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # --------------------
    # Resume Training
    # --------------------
    start_epoch = load_checkpoint(G, D, g_opt, d_opt)

    # --------------------
    # Training Loop
    # --------------------
    for epoch in range(start_epoch, epochs):
        print(f"Epoch [{epoch+1}/{epochs}] has started.")
        G.train()
        D.train()

        for real_input, real_target in train_loader:
            real_input = real_input.to(DEVICE)
            real_target = real_target.to(DEVICE)

            task_vector = get_task_vector(real_input.size(0), TASK_ID)

            # ---- Train Discriminator ----
            fake_img = G(real_input, task_vector).detach()

            d_opt.zero_grad()
            pred_real = D(real_target, task_vector)
            pred_fake = D(fake_img, task_vector)
            d_loss, _ = criterion.discriminator_loss(pred_real, pred_fake)
            d_loss.backward()
            d_opt.step()

            # ---- Train Generator ----
            g_opt.zero_grad()
            fake_img = G(real_input, task_vector)
            pred_fake = D(fake_img, task_vector)
            g_loss, _ = criterion.generator_loss(
                pred_fake, fake_img, real_target
            )
            g_loss.backward()
            g_opt.step()

        # --------------------
        # Validation
        # --------------------
        print("Evaluating on validation set...")
        G.eval()
        metrics = {"PSNR": [], "SSIM": [], "F-Score": [], "Hausdorff": []}

        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
                fake = G(inp, get_task_vector(inp.size(0), TASK_ID))
                m = evaluate_metrics(fake, tgt)
                for k in metrics:
                    metrics[k].append(m[k])

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"PSNR: {np.mean(metrics['PSNR']):.2f} | "
            f"SSIM: {np.mean(metrics['SSIM']):.4f}"
        )

        # --------------------
        # Save Sample & Checkpoint
        # --------------------
        save_image(
            torch.cat([inp, fake, tgt], 0),
            os.path.join(SAMPLE_DIR, f"deblur_epoch_{epoch+1}.png"),
            normalize=True,
            value_range=(-1, 1),
        )

        save_checkpoint(epoch, G, D, g_opt, d_opt)

    torch.save(G.state_dict(), os.path.join(CHECKPOINT_DIR, "G_deblur.pth"))
    print("✅ Training completed successfully")

# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Model training has started...")
    train()