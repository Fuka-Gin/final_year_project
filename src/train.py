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

# Device & Paths
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "results", "samples")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)


# Feature Configuration
FEATURE_CONFIG = {
    "deblur": {
        "id": 0,
        "gan_mode": "hinge",
        "recon_mode": "l2",
        "lambda_recon": 70,
        "lambda_perceptual": 10,
        "lambda_tv": 2,
        "lambda_edge": 5,
        "lambda_ms": 1,
        "lambda_freq": 0.1
    },
    "lowlight": {
        "id": 1,
        "gan_mode": "hinge",
        "recon_mode": "l1",
        "lambda_recon": 100,
        "lambda_perceptual": 10,
        "lambda_tv": 2,
        "lambda_edge": 3,
        "lambda_ms": 1,
        "lambda_freq": 0.05
    },
    "artistic": {
        "id": 2,
        "gan_mode": "hinge",
        "recon_mode": "l1",
        "lambda_recon": 10,
        "lambda_perceptual": 40,
        "lambda_tv": 1,
        "lambda_edge": 2,
        "lambda_ms": 1,
        "lambda_freq": 0.1
    }
}

FEATURE = "artistic"
TASK_ID = FEATURE_CONFIG[FEATURE]["id"]
NUM_TASKS = len(FEATURE_CONFIG)

if FEATURE == "artistic":
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "unpaired", "artistic")
else:
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "paired", FEATURE)

# Task Vector
def get_task_vector(batch_size, task_id):
    task = torch.zeros(batch_size, NUM_TASKS, device=DEVICE)
    task[:, task_id] = 1.0
    return task

# Loss Function
def create_loss_function():
    cfg = FEATURE_CONFIG[FEATURE]
    return UcGANLoss(
        gan_mode=cfg["gan_mode"],
        recon_mode=cfg["recon_mode"],
        lambda_recon=cfg["lambda_recon"],
        lambda_perceptual=cfg["lambda_perceptual"],
        lambda_tv=cfg["lambda_tv"],
        lambda_edge=cfg["lambda_edge"],
        lambda_ms=cfg["lambda_ms"],
        lambda_freq=cfg["lambda_freq"],
    )

# Checkpoint Utilities
def save_checkpoint(epoch, G, D, g_opt, d_opt):
    torch.save(
        {
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "g_opt": g_opt.state_dict(),
            "d_opt": d_opt.state_dict(),
        },
        os.path.join(CHECKPOINT_DIR, f"G_{FEATURE}.pth"),
    )

def load_checkpoint(G, D, g_opt, d_opt):

    path = os.path.join(CHECKPOINT_DIR, f"G_{FEATURE}.pth")

    if not os.path.exists(path):
        print("No checkpoint found. Starting training from scratch.")
        return 0

    print(f"Loading checkpoint: {path}")

    checkpoint = torch.load(path, map_location=DEVICE)

    # Case 1: Full checkpoint dictionary
    if isinstance(checkpoint, dict) and "G" in checkpoint:

        G.load_state_dict(checkpoint["G"], strict=False)
        D.load_state_dict(checkpoint["D"], strict=False)

        try:
            g_opt.load_state_dict(checkpoint["g_opt"])
            d_opt.load_state_dict(checkpoint["d_opt"])
        except:
            print("⚠ Optimizer state incompatible. Reinitializing.")

        start_epoch = checkpoint["epoch"] + 1

        print(f"🔁 Resuming training from epoch {start_epoch}")

        return start_epoch

    # Case 2: Generator-only weights
    else:

        print("⚠ Generator-only checkpoint detected.")

        model_dict = G.state_dict()

        for k in model_dict.keys():

            if k in checkpoint:

                if model_dict[k].shape == checkpoint[k].shape:
                    model_dict[k] = checkpoint[k]

                # handle expanded first conv layer
                elif "initial.0.weight" in k:

                    old_w = checkpoint[k]
                    new_w = model_dict[k]

                    new_w[:, :old_w.shape[1], :, :] = old_w

                    torch.nn.init.normal_(
                        new_w[:, old_w.shape[1]:, :, :],
                        mean=0,
                        std=0.02
                    )

                    model_dict[k] = new_w

        G.load_state_dict(model_dict)

        print("✔ Generator weights transferred (expanded tasks)")

        return 0

# Training Function
def train(
    epochs=100,
    batch_size=8,
    lr=1e-4,
):
    # Dataset
    paired = FEATURE != "artistic"

    train_ds = UcGANDataset(DATA_ROOT, mode="train", paired=paired)
    val_ds = UcGANDataset(DATA_ROOT, mode="val", paired=paired)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Models
    G = Generator(num_tasks=NUM_TASKS).to(DEVICE)
    D = Discriminator(num_tasks=NUM_TASKS).to(DEVICE)
    print("Generator and Discriminator initialized.")

    # Loss & Optimizers
    criterion = create_loss_function()

    g_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Resume Training
    start_epoch = load_checkpoint(G, D, g_opt, d_opt)

    # Training Loop
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

        # Validation
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

        # Save Sample & Checkpoint
        save_image(
            torch.cat([inp, fake, tgt], 0),
            os.path.join(SAMPLE_DIR, f"{FEATURE}_epoch_{epoch+1}.png"),
            normalize=True,
            value_range=(-1, 1),
        )

        save_checkpoint(epoch, G, D, g_opt, d_opt)

    save_checkpoint(epoch, G, D, g_opt, d_opt)
    print(f"✅ Training completed for {FEATURE} feature")

# Entry Point
if __name__ == "__main__":
    print("Model training has started...")
    train()