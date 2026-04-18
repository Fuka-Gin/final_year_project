# 🚀 UcGAN Image Enhancement System

## 📌 Overview

UcGAN is a deep learning-based image enhancement system that performs:

* Deblurring
* Low-light enhancement
* Image quality analysis
* Intelligent recommendations

The system uses a **GAN-based architecture** along with **YOLO object detection** and **BLIP captioning** to provide a complete image enhancement pipeline.

---

## 🧠 Key Features

* 🔍 Image Quality Analysis (Brightness, Blur, Contrast, Noise)
* 💡 Smart Recommendations
* 🎯 User-guided enhancement (Deblur / Low-light)
* 📊 Before vs After comparison
* 🧾 Output report generation
* 🖼️ Object detection (YOLOv8)
* 📝 Image captioning (BLIP)
* 🌐 Interactive UI (Gradio)

---

## 🏗️ System Architecture

```text
Input Image
   ↓
Image Analysis
   ↓
Suggestions
   ↓
User selects feature
   ↓
Generator (UcGAN)
   ↓
Enhanced Image
   ↓
Output Analysis
   ↓
Final Report
```

---

## ⚙️ Key Components

### 1. Generator (G)

* Enhances images (Deblur / Low-light)
* Uses CNN-based architecture
* Takes input image + task vector

---

### 2. Discriminator (D)

* Classifies images as real or fake
* Improves generator quality

---

### 3. Loss Functions

* GAN Loss → realism
* L1 Loss → accuracy
* Perceptual Loss → visual quality
* TV Loss → smoothness

---

### 4. Analysis Module

* Calculates brightness, blur, contrast
* Generates recommendations
* Evaluates improvement

---

### 5. YOLO (Object Detection)

* Detects objects in output image

---

### 6. BLIP (Captioning)

* Generates image description

---

## 🛠️ Installation

### Step 1: Clone Project

```bash
git clone <your-repo-url>
cd UcGAN
```

---

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4: Install Tesseract (Optional)

Download from:
https://github.com/tesseract-ocr/tesseract

---

## 📂 Project Structure

```text
UcGAN/
│
├── src/
│   ├── generator.py
│   ├── discriminator.py
│   ├── train.py
│   ├── inference.py
│   ├── analysis.py
│   ├── app.py
│
├── data/
├── checkpoints/
├── results/
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 🔹 Run Training

```bash
python train.py
```

---

### 🔹 Run Inference (CLI)

```bash
python inference.py
```

---

### 🔹 Run Web Application (Recommended)

```bash
python app.py
```

Then open browser:

```text
http://127.0.0.1:7860
```

---

## 🔄 Workflow

1. Upload image
2. Click **Analyze Image**
3. View suggestions
4. Select enhancement feature
5. Click **Enhance Image**
6. View:

   * Enhanced image
   * Improvement report
   * New suggestions

---

## 📊 Evaluation Metrics

* PSNR (Peak Signal-to-Noise Ratio)
* SSIM (Structural Similarity Index)

---

## 💻 Hardware Used

* Intel i3 CPU
* 8GB RAM
* No GPU (local)
* Kaggle GPU used for training

---

## 🚀 Future Improvements

* Add more enhancement tasks
* Automatic feature selection
* Mobile deployment
* Real-time processing

---

## 🏁 Conclusion

UcGAN provides a unified and intelligent framework for image enhancement by combining deep learning with user-guided analysis and evaluation.