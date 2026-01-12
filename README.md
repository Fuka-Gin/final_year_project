# 🖼️ UcGAN – Unified Conditional GAN for Image Restoration

This project implements a **Unified Conditional Generative Adversarial Network (UcGAN)** for **image restoration tasks**, with a primary focus on **Blur → Sharp Image Restoration (Deblurring)**.

The model is designed to be **extensible**, allowing additional image-to-image tasks (e.g., denoising, low-light enhancement) to be integrated using the same architecture by changing datasets and task conditions.

---

## 🚀 Key Features

* ✅ **Unified Conditional GAN (UcGAN) Architecture**
* ✅ **ResNet-based Generator**
* ✅ **PatchGAN Discriminator**
* ✅ **Task-conditioning using one-hot vectors**
* ✅ **Perceptual Loss using pretrained VGG19**
* ✅ **Automatic Train/Validation Split**
* ✅ **Paired Data Augmentation**
* ✅ **Resume Training & Fine-tuning Support**
* ✅ **Single-image Inference Support**

---

## 📌 Implemented Feature

### ✔ Blur → Sharp Image Restoration (Deblurring)

* **Input:** Blurred image
* **Output:** Restored sharp image
* **Training:** Supervised (paired dataset)

This feature is fully implemented, trained, and tested.

---

## 🧠 Model Architecture

### Generator

* ResNet-based encoder–decoder
* Task-conditioning via spatially expanded one-hot vectors
* Instance Normalization for stability
* Tanh output for normalized image generation

### Discriminator

* Conditional PatchGAN (70×70)
* Receives both image and task vector
* Predicts realism at patch level

---

## 📂 Project Structure

```
UcGAN/
├── src/
│   ├── train.py              # Training script (with resume support)
│   ├── inference.py          # Single-image inference
│   ├── generator.py          # Generator network
│   ├── discriminator.py      # Discriminator network
│   ├── residual_blocks.py    # Residual blocks
│   ├── loss_functions.py     # GAN + Reconstruction + Perceptual loss
│   ├── dataset.py            # Dataset loader with augmentation
│   └── metrics.py            # PSNR, SSIM, etc.
│
├── data/                     # (Ignored in GitHub)
│   └── paired/
│       └── deblur/
│           ├── input/
│           └── target/
│
├── checkpoints/              # Saved model weights (ignored)
├── results/
│   └── samples/              # Training & inference outputs
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

### Primary Dataset (Recommended)

* **GoPro Deblurring Dataset**
* Real-world motion blur with paired sharp ground truth

🔗 Dataset link:
[https://seungjunnah.github.io/Datasets/gopro](https://seungjunnah.github.io/Datasets/gopro)

### Expected Dataset Format

```
data/paired/deblur/
├── input/    # Blurred images
└── target/   # Sharp images
```

> ⚠️ Datasets are **not included** in this repository due to size and licensing constraints.

---

## 🔧 Installation

### 1️⃣ Create Environment

```bash
pip install -r requirements.txt
```

### 2️⃣ Verify PyTorch

```bash
python -c "import torch; print(torch.__version__)"
```

---

## 🏋️ Training the Model

### Start Training

```bash
python src/train.py
```

### Resume Training

Training automatically resumes from the last saved checkpoint if available.

---

## 🧪 Inference (Single Image)

```bash
python src/inference.py
```

You can provide **absolute image paths** inside `inference.py`.

---

## 📈 Loss Functions Used

* **Adversarial Loss:** Hinge Loss
* **Reconstruction Loss:** L2 (for deblurring)
* **Perceptual Loss:** VGG19 feature loss
* **Total Variation Loss:** Spatial smoothness

> The VGG19 network is used **only for perceptual loss** and is not trained.

---

## 🧠 Pretrained Models

* ✔ **VGG19 (ImageNet)** used for perceptual loss
* ❌ Generator & Discriminator trained **from scratch**
* ❌ Trained weights are **not uploaded** to GitHub

---

## 📁 GitHub Usage Policy

The following are **excluded** from the repository:

* Datasets
* Model checkpoints (`.pth`)
* Cache files

These are listed in `.gitignore`.

---

## 📌 Future Extensions

The architecture supports additional tasks such as:

* Image Denoising
* Low-light Enhancement
* Super-resolution
* Reflection Removal

Only dataset and loss configuration changes are required.

---

## 🧾 Academic Note

This project follows standard research practices used in IEEE publications:

* No dataset redistribution
* Clear training pipeline
* Reproducible experiments
* Proper use of pretrained feature extractors

---

## 👨‍🎓 Author

**Final Year Project – Unified Conditional GAN for Image Restoration**

---

## 📜 License

This project is intended for **academic and research use only**.