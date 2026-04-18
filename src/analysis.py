import os
import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# OPTIONAL IMPORTS
# ---------------------------------------------------------
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except:
    TESSERACT_AVAILABLE = False

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    BLIP_AVAILABLE = True
except:
    BLIP_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO_AVAILABLE = False


DEVICE = "cuda" if (BLIP_AVAILABLE and torch.cuda.is_available()) else "cpu"

_blip_model = None
_blip_processor = None
_yolo_model = None


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
def load_blip():
    global _blip_model, _blip_processor
    if _blip_model is None:
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(DEVICE)
        _blip_model.eval()
    return _blip_processor, _blip_model


def load_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8n.pt")  # fast + lightweight
    return _yolo_model


# ---------------------------------------------------------
# IMAGE ANALYSIS
# ---------------------------------------------------------
def analyze_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray) / 255 * 100

    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur = min(lap / 1500 * 100, 100)

    contrast = min(np.std(gray) / 80 * 100, 100)

    noise = min(
        np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0)) / 30 * 100,
        100
    )

    text = 0
    if TESSERACT_AVAILABLE:
        try:
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            conf = [int(x) for x in data["conf"] if x.isdigit()]
            text = np.mean(conf) if conf else 0
        except:
            text = 0

    overall = (
        0.3 * brightness +
        0.3 * blur +
        0.2 * contrast +
        0.1 * (100 - noise) +
        0.1 * text
    )

    return {
        "brightness": round(brightness, 1),
        "blur": round(blur, 1),
        "contrast": round(contrast, 1),
        "noise": round(noise, 1),
        "text": round(text, 1),
        "overall": round(overall, 1)
    }


# ---------------------------------------------------------
# RECOMMENDATIONS (SIMPLE LANGUAGE)
# ---------------------------------------------------------
def generate_recommendations(m):
    rec = []

    # --- FIXING ISSUES ---
    if m["brightness"] < 40:
        rec.append("Image is too dark. Increase brightness using low-light enhancement.")
    else:
        rec.append("Brightness is good, but can still be slightly improved.")

    if m["blur"] < 40:
        rec.append("Image is blurry. Apply deblurring to improve sharpness.")
    else:
        rec.append("Image is sharp, but fine details can be enhanced further.")

    if m["contrast"] < 40:
        rec.append("Improve contrast for better visibility.")
    else:
        rec.append("Contrast is acceptable, but can be enhanced for better depth.")

    if m["noise"] > 60:
        rec.append("Image has noise. Apply noise reduction.")
    else:
        rec.append("Image is clean, but slight smoothing can improve quality.")

    # --- IMPROVEMENT SUGGESTIONS (NEW) ---
    rec.append("Enhance colors to make the image more vibrant and visually appealing.")
    rec.append("Increase image resolution for sharper and clearer details.")
    rec.append("Apply combined enhancement (deblur + low-light) for best overall result.")

    # Keep only 5 clean suggestions
    return rec[:5]


# ---------------------------------------------------------
# CAPTION
# ---------------------------------------------------------
def generate_caption(path):
    if not BLIP_AVAILABLE:
        return "Caption not available"

    try:
        processor, model = load_blip()
        img = Image.open(path).convert("RGB")

        inputs = processor(img, return_tensors="pt").to(DEVICE)
        out = model.generate(**inputs)

        caption = processor.decode(out[0], skip_special_tokens=True)
        caption = caption.capitalize()

        # clean repetition
        caption = caption.replace("and a", "with a")
        caption = caption.replace("sitting on a table", "placed on a table")

        return caption

    except:
        return "Could not generate caption"


# ---------------------------------------------------------
# YOLO OBJECT DETECTION
# ---------------------------------------------------------
def detect_objects(path):
    if not YOLO_AVAILABLE:
        return []

    try:
        model = load_yolo()
        results = model(path, conf=0.4, verbose=False)

        names = model.names
        detected = []

        for r in results:
            for cls in r.boxes.cls:
                detected.append(names[int(cls)])

        return list(set(detected))[:5]

    except:
        return []


# ---------------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------------
def generate_full_report(input_path, output_path):
    analysis = analyze_image(input_path)
    rec = generate_recommendations(analysis)

    caption = generate_caption(output_path)
    objects = detect_objects(output_path)

    return {
        "analysis": analysis,
        "recommendations": rec,
        "caption": caption,
        "objects": objects
    }


# ---------------------------------------------------------
# UI FORMAT (PROFESSIONAL)
# ---------------------------------------------------------
def progress_bar(value):
    filled = int(value // 10)
    return "█" * filled + "░" * (10 - filled)


def format_report_for_display(r, feature):

    a = r["analysis"]

    lines = []
    lines.append("╔══════════════════════════════════════╗")
    lines.append("║        UcGAN Image Assistant         ║")
    lines.append("╚══════════════════════════════════════╝\n")

    lines.append(f" Enhancement applied : {feature}\n")

    lines.append("── IMAGE QUALITY ─────────────────────")
    lines.append(f" Brightness  {progress_bar(a['brightness'])}  {a['brightness']}%")
    lines.append(f" Sharpness   {progress_bar(a['blur'])}  {a['blur']}%")
    lines.append(f" Contrast    {progress_bar(a['contrast'])}  {a['contrast']}%")
    lines.append(f" Noise       {progress_bar(a['noise'])}  {a['noise']}%")
    lines.append(f" Overall     {progress_bar(a['overall'])}  {a['overall']}%\n")

    lines.append("── SIMPLE SUGGESTIONS ─────────────────")
    for i, s in enumerate(r["recommendations"], 1):
        lines.append(f" {i}. {s}")

    lines.append("\n── OUTPUT IMAGE ───────────────────────")
    lines.append(f" Caption : {r['caption']}")

    if r["objects"]:
        lines.append(f" Objects : {', '.join(r['objects'])}")
    else:
        lines.append(" Objects : Not clearly detected")

    lines.append("\n═══════════════════════════════════════")

    return "\n".join(lines)