import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from generator import Generator

# ---------------------------------------------------------
# CONFIG (MATCH TRAINING EXACTLY)
# ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASK_ID = 0
NUM_TASKS = 1
IMG_SIZE = 256


# ---------------------------------------------------------
# Inference Function
# ---------------------------------------------------------
def infer(
    image_path,
    checkpoint_dir="checkpoints",
    output_dir="results/inference",
):
    os.makedirs(output_dir, exist_ok=True)

    print("🔄 Processing... Please wait")

    # -------------------------------------------------
    # Resolve Checkpoint Path (IMPORTANT FIX)
    # -------------------------------------------------
    final_ckpt = os.path.join(checkpoint_dir, "G_deblur.pth")
    last_ckpt = os.path.join(checkpoint_dir, "last.pth")

    if os.path.exists(final_ckpt):
        print("✅ Using final trained model: G_deblur.pth")
        state_dict = torch.load(final_ckpt, map_location=DEVICE)
    elif os.path.exists(last_ckpt):
        print("⚠️ Final model not found. Using last checkpoint: last.pth")
        checkpoint = torch.load(last_ckpt, map_location=DEVICE)
        state_dict = checkpoint["G"]
    else:
        raise FileNotFoundError(
            "No trained model found. Please train the model first."
        )

    # -------------------------------------------------
    # Load Generator
    # -------------------------------------------------
    G = Generator(num_tasks=NUM_TASKS).to(DEVICE)
    G.load_state_dict(state_dict)
    G.eval()

    # -------------------------------------------------
    # Transform (MATCH TRAINING)
    # -------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    # -------------------------------------------------
    # Load Image
    # -------------------------------------------------
    assert os.path.exists(image_path), f"Image not found: {image_path}"

    input_pil = Image.open(image_path).convert("RGB")
    input_tensor = transform(input_pil).unsqueeze(0).to(DEVICE)

    # -------------------------------------------------
    # Task Vector
    # -------------------------------------------------
    task_vector = torch.zeros(1, NUM_TASKS, device=DEVICE)
    task_vector[:, TASK_ID] = 1.0

    # -------------------------------------------------
    # Forward Pass
    # -------------------------------------------------
    with torch.no_grad():
        output = G(input_tensor, task_vector)

    # -------------------------------------------------
    # Denormalize [-1,1] → [0,1]
    # -------------------------------------------------
    input_img = (input_tensor.squeeze(0).cpu() + 1) / 2
    output_img = (output.squeeze(0).cpu() + 1) / 2

    input_img = input_img.clamp(0, 1)
    output_img = output_img.clamp(0, 1)

    # -------------------------------------------------
    # Save Results
    # -------------------------------------------------
    input_save = os.path.join(output_dir, "input.png")
    output_save = os.path.join(output_dir, "output.png")
    compare_save = os.path.join(output_dir, "comparison.png")

    save_image(input_img, input_save)
    save_image(output_img, output_save)
    save_image(torch.cat([input_img, output_img], dim=2), compare_save)

    print("✅ Processing complete")
    print(f"✔ Input saved to: {input_save}")
    print(f"✔ Output saved to: {output_save}")
    print(f"✔ Comparison saved to: {compare_save}")


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":

    infer(
        image_path=r"D:\PROJECTS\Final_Year_Project\GAN dataset\blur_dataset_scaled\motion_blurred\0_IPHONE-SE_M.JPG"
    )
