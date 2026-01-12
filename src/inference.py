import os
import torch
from PIL import Image
import torchvision.transforms as transforms

from generator import Generator

# ---------------------------------------------------------
# CONFIG (MATCH TRAINING EXACTLY)
# ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE = "deblur"
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
    # Load Generator
    # -------------------------------------------------
    model_path = os.path.join(checkpoint_dir, "G_deblur.pth")
    assert os.path.exists(model_path), f"Checkpoint not found: {model_path}"

    G = Generator(num_tasks=NUM_TASKS).to(DEVICE)
    G.load_state_dict(torch.load(model_path, map_location=DEVICE))
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
    # Denormalize Output [-1,1] → [0,1]
    # -------------------------------------------------
    output_img = output.squeeze(0).cpu()
    output_img = (output_img + 1) / 2
    output_img = output_img.clamp(0, 1)

    output_pil = transforms.ToPILImage()(output_img)

    # -------------------------------------------------
    # Save Output
    # -------------------------------------------------
    save_path = os.path.join(
        output_dir,
        f"deblur_{os.path.basename(image_path)}"
    )
    output_pil.save(save_path)

    print("✅ Processing complete")
    print(f"✔ Output saved to: {save_path}")

    # -------------------------------------------------
    # DISPLAY IMAGES (SYSTEM IMAGE VIEWER)
    # -------------------------------------------------
    print("🖼️ Displaying input and output images...")

    input_pil.resize((IMG_SIZE, IMG_SIZE)).show(title="Input (Blurred)")
    output_pil.show(title="Output (Deblurred)")


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":

    infer(
        image_path=r"D:\PROJECTS\Final_Year_Project\implementation\data\paired\deblur\input\3_HUAWEI-NOVA-LITE_.jpg"
    )
