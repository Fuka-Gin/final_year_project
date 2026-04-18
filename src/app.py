import os
import sys
import torch
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generator import Generator
from analysis import (
    analyze_image,
    generate_recommendations,
    generate_full_report,
    format_report_for_display,
    progress_bar,
)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE  = 256

PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "..", "checkpoints")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "..", "results", "inference")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_MAP = {
    "Deblur":                ("deblur",   0),
    "Low-light Enhancement": ("lowlight", 1),
}

_model_cache = {}
_temp_input_path = os.path.join(OUTPUT_DIR, "_stage1_input.png")


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------
def load_model(feature: str):
    if feature in _model_cache:
        return _model_cache[feature]

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"G_{feature}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train the {feature} model first."
        )

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = (
        checkpoint["G"]
        if isinstance(checkpoint, dict) and "G" in checkpoint
        else checkpoint
    )
    num_tasks = state_dict["initial.0.weight"].shape[1] - 3
    G = Generator(num_tasks=num_tasks).to(DEVICE)
    G.load_state_dict(state_dict, strict=False)
    G.eval()
    _model_cache[feature] = G
    return G


# ---------------------------------------------------------
# STAGE 1 — Analyse input image
# ---------------------------------------------------------
def stage1_analyse(image: Image.Image):
    """
    Runs when user clicks 'Analyse Image'.
    Returns the pre-enhancement report text only.
    """
    if image is None:
        return "Please upload an image first."

    # Save to disk so analysis.py can read it with OpenCV
    image.save(_temp_input_path)

    try:
        metrics = analyze_image(_temp_input_path)
        recs    = generate_recommendations(metrics)
    except Exception as exc:
        return f"Analysis failed: {exc}"

    a = metrics
    lines = [
        "╔══════════════════════════╗",
        "║      Image Analysis    ║",
        "╚══════════════════════════╝",
        "",
        "── IMAGE QUALITY ──────────────────────",
        f"  Brightness  {progress_bar(a['brightness'])}  {a['brightness']}%",
        f"  Sharpness   {progress_bar(a['blur'])}  {a['blur']}%",
        f"  Contrast    {progress_bar(a['contrast'])}  {a['contrast']}%",
        f"  Noise       {progress_bar(a['noise'])}  {a['noise']}%  (lower = cleaner)",
        f"  Overall     {progress_bar(a['overall'])}  {a['overall']}%",
        "",
        "── WHAT'S WRONG & WHAT TO DO ──────────",
    ]

    for i, rec in enumerate(recs, 1):
        lines.append(f"  {i}. {rec}")

    lines += [
        "",
        "── RECOMMENDED ENHANCEMENT ────────────",
    ]

    # Auto-suggest feature based on dominant issue
    if a["blur"] < 45 and a["brightness"] < 45:
        lines.append("  Both Deblur and Low-light Enhancement are recommended.")
        lines.append("  Start with Low-light Enhancement first, then Deblur.")
    elif a["blur"] < 45:
        lines.append("  → Select  Deblur  below and click 'Run Enhancement'.")
    elif a["brightness"] < 45:
        lines.append("  → Select  Low-light Enhancement  below and click 'Run Enhancement'.")
    else:
        lines.append("  Image quality is reasonable. Either enhancement may help.")

    lines += ["", "═══════════════════════════════════════"]
    return "\n".join(lines)


# ---------------------------------------------------------
# STAGE 2 — Enhance + post-enhancement report
# ---------------------------------------------------------
def stage2_enhance(image: Image.Image, feature_label: str):
    """
    Runs when user clicks 'Run Enhancement'.
    Returns: original_pil, enhanced_pil, post_report_text
    """
    if image is None:
        return None, None, "Please upload an image first."

    feature, task_id = FEATURE_MAP[feature_label]

    try:
        G = load_model(feature)
    except FileNotFoundError as e:
        return None, None, str(e)

    num_tasks = G.num_tasks

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    task_vector  = torch.zeros(1, num_tasks, device=DEVICE)
    task_vector[:, task_id] = 1.0

    with torch.no_grad():
        output = G(input_tensor, task_vector)

    input_img  = (input_tensor.squeeze(0).cpu() + 1) / 2
    output_img = (output.squeeze(0).cpu() + 1) / 2
    input_img  = input_img.clamp(0, 1)
    output_img = output_img.clamp(0, 1)

    to_pil     = transforms.ToPILImage()
    input_pil  = to_pil(input_img)
    output_pil = to_pil(output_img)

    # Save both for analysis
    input_save  = os.path.join(OUTPUT_DIR, f"{feature}_input_temp.png")
    output_save = os.path.join(OUTPUT_DIR, f"{feature}_output_temp.png")
    input_pil.save(input_save)
    output_pil.save(output_save)

    # Analyse both before and after
    try:
        before = analyze_image(input_save)
        after  = analyze_image(output_save)
        recs   = generate_recommendations(after)

        # BLIP caption + YOLO for output only
        try:
            from analysis import generate_caption, detect_objects
            caption = generate_caption(output_save)
            objects = detect_objects(output_save)
        except Exception:
            caption = "Unavailable"
            objects = []

        report_text = _format_post_report(before, after, recs, caption, objects, feature_label)
    except Exception as exc:
        report_text = f"Post-enhancement analysis failed: {exc}"

    return input_pil, output_pil, report_text


def _delta_str(before_val, after_val):
    """Return a coloured delta string like +12.3% or -4.1%."""
    d = after_val - before_val
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}%"


def _format_post_report(before, after, recs, caption, objects, feature_label):
    lines = [
        "╔═══════════════════════════════╗",
        "║     Enhancement Report     ║",
        "╚═══════════════════════════════╝",
        "",
        f"  Enhancement applied : {feature_label}",
        "",
        "── BEFORE vs AFTER ────────────────────",
        f"  {'Metric':<12} {'Before':>8}   {'After':>8}   {'Change':>8}",
        "  " + "─" * 44,
    ]

    metrics_order = [
        ("Brightness",  "brightness"),
        ("Sharpness",   "blur"),
        ("Contrast",    "contrast"),
        ("Noise ↓",     "noise"),
        ("Overall",     "overall"),
    ]

    for label, key in metrics_order:
        b = before[key]
        a = after[key]
        delta = _delta_str(b, a)
        # For noise, improvement = lower, so flip the arrow logic
        if key == "noise":
            arrow = "✓" if a < b else ("✗" if a > b + 2 else "~")
        else:
            arrow = "✓" if a > b else ("✗" if a < b - 2 else "~")
        lines.append(f"  {label:<12} {b:>7.1f}%   {a:>7.1f}%   {delta:>6}  {arrow}")

    lines += [
        "",
        "── HOW MUCH DID IT IMPROVE? ───────────",
    ]

    overall_delta = after["overall"] - before["overall"]
    if overall_delta >= 15:
        lines.append(f"  Significant improvement (+{overall_delta:.1f}%). Enhancement worked well.")
    elif overall_delta >= 5:
        lines.append(f"  Moderate improvement (+{overall_delta:.1f}%). Image is noticeably better.")
    elif overall_delta >= 0:
        lines.append(f"  Slight improvement (+{overall_delta:.1f}%). Subtle but positive change.")
    else:
        lines.append(f"  Minimal change ({overall_delta:.1f}%). Consider trying the other feature.")

    lines += [
        "",
        "── FURTHER IMPROVEMENT SUGGESTIONS ────",
    ]
    for i, rec in enumerate(recs, 1):
        lines.append(f"  {i}. {rec}")

    lines += [
        "",
        "── OUTPUT IMAGE DESCRIPTION ────────────",
        f"  Caption : {caption}",
        f"  Objects : {', '.join(objects) if objects else 'None detected'}",
        "",
        "═══════════════════════════════════════",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------
css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400&display=swap');

:root {
    --bg:      #0a0a0f;
    --surface: #111118;
    --border:  #1e1e2e;
    --accent:  #7c6af7;
    --accent2: #c084fc;
    --text:    #e8e8f0;
    --muted:   #6b6b80;
    --green:   #4ade80;
    --radius:  12px;
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1300px !important;
    margin: 0 auto !important;
    padding: 2rem 1.5rem !important;
}

/* ── Header ─────────────────────────────── */
#header {
    text-align: center;
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
#header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #7c6af7, #c084fc, #f472b6);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 0 0 0.4rem !important;
    letter-spacing: -0.02em;
}
#header p {
    color: var(--muted) !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}

/* ── Step badges ─────────────────────────── */
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c6af7, #c084fc);
    color: white;
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.75rem;
}

/* ── Buttons ─────────────────────────────── */
#analyse-btn {
    background: var(--surface) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 8px !important;
    color: var(--accent) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    padding: 0.65rem 1.5rem !important;
    width: 100%;
    cursor: pointer !important;
    transition: all 0.2s !important;
    letter-spacing: 0.03em;
}
#analyse-btn:hover {
    background: var(--accent) !important;
    color: white !important;
}

#enhance-btn {
    background: linear-gradient(135deg, #7c6af7, #c084fc) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    padding: 0.65rem 1.5rem !important;
    width: 100%;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
#enhance-btn:hover  { opacity: 0.88 !important; transform: translateY(-1px) !important; }
#enhance-btn:active { transform: translateY(0) !important; }

/* ── Report textboxes ────────────────────── */
#pre-report textarea,
#post-report textarea {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.76rem !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--green) !important;
    line-height: 1.6 !important;
}

/* ── Section labels ─────────────────────── */
.section-label {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 0.5rem !important;
}

label span {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--muted) !important;
    font-size: 0.82rem !important;
}

/* ── Divider ─────────────────────────────── */
.stage-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}
"""

# ---------------------------------------------------------
# UI — Two-stage layout
# ---------------------------------------------------------
with gr.Blocks(title="UcGAN — Image Enhancement") as demo:

    gr.HTML("""
    <div id="header">
        <h1>UcGAN Image Enhancement</h1>
        <p>Analyse first · Choose your enhancement · See exactly what improved</p>
    </div>
    """)

    # ════════════════════════════════════════════════════════
    # STAGE 1 ROW
    # ════════════════════════════════════════════════════════
    gr.HTML('<div style="margin-bottom:0.5rem"><span class="step-badge">Step 1 — Analyse</span></div>')

    with gr.Row():
        # Input upload
        with gr.Column(scale=1, min_width=280):
            gr.HTML('<div class="section-label">Upload Image</div>')
            input_image = gr.Image(type="pil", label="", height=280)
            analyse_btn = gr.Button("Analyse Image", elem_id="analyse-btn")

        # Pre-enhancement report
        with gr.Column(scale=2, min_width=420):
            gr.HTML('<div class="section-label">Image Quality Report</div>')
            pre_report = gr.Textbox(
                label="",
                interactive=False,
                lines=20,
                max_lines=25,
                elem_id="pre-report",
                placeholder="Upload an image and click 'Analyse Image' to see the quality report and suggestions.",
            )

    gr.HTML('<hr class="stage-divider">')

    # ════════════════════════════════════════════════════════
    # STAGE 2 ROW
    # ════════════════════════════════════════════════════════
    gr.HTML('<div style="margin-bottom:0.5rem"><span class="step-badge">Step 2 — Enhance</span></div>')

    with gr.Row():
        # Feature selector + enhance button
        with gr.Column(scale=1, min_width=280):
            gr.HTML('<div class="section-label">Select Feature</div>')
            feature_radio = gr.Radio(
                choices=list(FEATURE_MAP.keys()),
                value="Deblur",
                label="",
                interactive=True,
            )
            enhance_btn = gr.Button("Run Enhancement", elem_id="enhance-btn")

            gr.HTML('<div class="section-label" style="margin-top:1.2rem">Original</div>')
            original_out = gr.Image(type="pil", label="", height=220, interactive=False)

            gr.HTML('<div class="section-label" style="margin-top:0.8rem">Enhanced</div>')
            enhanced_out = gr.Image(type="pil", label="", height=220, interactive=False)

        # Post-enhancement report
        with gr.Column(scale=2, min_width=420):
            gr.HTML('<div class="section-label">Enhancement Report</div>')
            post_report = gr.Textbox(
                label="",
                interactive=False,
                lines=38,
                max_lines=45,
                elem_id="post-report",
                placeholder="Select a feature and click 'Run Enhancement' to see how the image improved.",
            )

    # ── Wire up buttons ──────────────────────────────────────────────────────
    analyse_btn.click(
        fn=stage1_analyse,
        inputs=[input_image],
        outputs=[pre_report],
    )

    enhance_btn.click(
        fn=stage2_enhance,
        inputs=[input_image, feature_radio],
        outputs=[original_out, enhanced_out, post_report],
    )

if __name__ == "__main__":
    demo.launch(share=False, css=css)