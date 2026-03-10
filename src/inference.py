import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from generator import Generator

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES = {
    "1": ("deblur", 0, 1),
    "2": ("lowlight", 1, 2),
    "3": ("artistic", 2, 3)
}

# ---------------------------------------------------------
# Inference Function
# ---------------------------------------------------------
def infer(image_path, feature, task_id, num_tasks):
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    output_dir = os.path.join(PROJECT_ROOT, "results", "inference")

    os.makedirs(output_dir, exist_ok=True)

    print("🔄 Processing... Please wait")

    # -------------------------------------------------
    # Load Model
    # -------------------------------------------------
    model_path = os.path.join(checkpoint_dir, f"G_{feature}.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\nTrain the model first."
        )

    print(f"✅ Using trained model: {model_path}")

    G = Generator(num_tasks=num_tasks).to(DEVICE)
    G.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    G.eval()

    # -------------------------------------------------
    # Transform
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
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    input_pil = Image.open(image_path).convert("RGB")
    input_tensor = transform(input_pil).unsqueeze(0).to(DEVICE)

    # -------------------------------------------------
    # Task Vector
    # -------------------------------------------------
    task_vector = torch.zeros(1, num_tasks, device=DEVICE)
    task_vector[:, task_id] = 1.0

    # -------------------------------------------------
    # Forward Pass
    # -------------------------------------------------
    with torch.no_grad():
        output = G(input_tensor, task_vector)

    # -------------------------------------------------
    # Convert back to image
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
# Feature Selector
# ---------------------------------------------------------
def select_feature():

    print("\nSelect Feature:")
    print("1 → Deblur")
    print("2 → Low-light Enhancement")

    choice = input("Enter choice: ")

    if choice not in FEATURES:
        print("Invalid choice!")
        exit()

    return FEATURES[choice]


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":

    feature, task_id, num_tasks = select_feature()

    image_path = input("\nEnter image path: ")

    infer(image_path, feature, task_id, num_tasks)