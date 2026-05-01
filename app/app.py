
import gradio as gr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")
from PIL import Image
import io

# ── Custom loss functions ──────────────────────────────────────
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    dice = 1 - dice_coefficient(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

# ── Load model ─────────────────────────────────────────────────
model = tf.keras.models.load_model(
    "lung_tumor_model.keras",
    custom_objects={
        "dice_coefficient": dice_coefficient,
        "bce_dice_loss": bce_dice_loss
    }
)
print("Model loaded ✓")

# ── Helper: get bounding box ───────────────────────────────────
def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, rmax, cmax

# ── Helper: get location label ─────────────────────────────────
def get_location(bbox, img_size=128):
    if bbox is None:
        return "N/A"
    rmin, cmin, rmax, cmax = bbox
    cx = (cmin + cmax) / 2
    cy = (rmin + rmax) / 2
    v = "Upper" if cy < img_size // 2 else "Lower"
    h = "Left"  if cx < img_size // 2 else "Right"
    return f"{v}-{h} region"

# ── Predict function ───────────────────────────────────────────
def predict(npy_file):
    try:
        volume = np.load(npy_file.name)

        if volume.ndim == 3:
            volume = volume[..., np.newaxis]
        if volume.ndim == 4:
            volume = volume[np.newaxis]
        if volume.max() > 1.0:
            volume = volume / 255.0

        if volume.shape[1:] != (8, 128, 128, 1):
            return None, f"❌ Wrong shape: {volume.shape}. Expected (1,8,128,128,1)"

        pred       = model.predict(volume, verbose=0)
        pred_masks = pred[0, :, :, :, 0]           # (8, 128, 128)
        pred_bin   = (pred_masks > 0.5).astype(np.float32)
        input_vol  = volume[0, :, :, :, 0]         # (8, 128, 128)

        # ── Detection decision ─────────────────────────────────
        total_px      = int(np.sum(pred_bin))
        active_slices = int(np.sum(pred_bin.sum(axis=(1, 2)) > 0))
        max_conf      = float(pred_masks.max())
        detected      = total_px > 50 and max_conf > 0.5

        # Best slice = most nodule pixels
        best_slice = int(np.argmax(pred_bin.sum(axis=(1, 2))))
        bbox       = get_bbox(pred_bin[best_slice])
        location   = get_location(bbox)

        # Risk level
        if total_px > 800:
            risk = "High"
            risk_color = "#ff4444"
        elif total_px > 300:
            risk = "Medium"
            risk_color = "#ffaa00"
        else:
            risk = "Low"
            risk_color = "#44ff44"

        # ── Plot ───────────────────────────────────────────────
        fig, axes = plt.subplots(3, 8, figsize=(22, 9))
        fig.patch.set_facecolor("#0f0f0f")

        status_text = "⚠ TUMOR DETECTED" if detected else "✓ NO TUMOR DETECTED"
        status_color = "#ff4444" if detected else "#44ff88"
        fig.suptitle(status_text, color=status_color, fontsize=16,
                     fontweight="bold", y=1.02)

        for s in range(8):
            is_best = (s == best_slice)
            border_color = "#ff4444" if is_best else "#333333"

            # Row 0 — Input
            axes[0, s].imshow(input_vol[s], cmap="gray")
            axes[0, s].set_title(f"Slice {s}" + (" ★" if is_best else ""),
                                 color="#ffdd00" if is_best else "white",
                                 fontsize=8, fontweight="bold" if is_best else "normal")
            axes[0, s].axis("off")
            for spine in axes[0, s].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)

            # Row 1 — Heatmap
            axes[1, s].imshow(pred_masks[s], cmap="hot", vmin=0, vmax=1)
            axes[1, s].set_title(f"conf={pred_masks[s].max():.2f}",
                                 color="white", fontsize=7)
            axes[1, s].axis("off")

            # Row 2 — Overlay + bounding box on best slice
            axes[2, s].imshow(input_vol[s], cmap="gray")
            axes[2, s].imshow(pred_bin[s], cmap="Reds", alpha=0.5)

            if is_best and bbox is not None:
                rmin, cmin, rmax, cmax = bbox
                rect = mpatches.Rectangle(
                    (cmin, rmin), cmax - cmin, rmax - rmin,
                    linewidth=2, edgecolor="#ff4444",
                    facecolor="none", linestyle="--"
                )
                axes[2, s].add_patch(rect)
                axes[2, s].text(
                    cmin, rmin - 4, "TUMOR",
                    color="#ff4444", fontsize=7, fontweight="bold"
                )

            axes[2, s].set_title(f"{int(pred_bin[s].sum())}px",
                                 color="white", fontsize=7)
            axes[2, s].axis("off")

        for row, label in enumerate(["Input CT", "Confidence", "Segmentation"]):
            axes[row, 0].set_ylabel(label, color="white", fontsize=9)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100,
                    bbox_inches="tight", facecolor="#0f0f0f")
        buf.seek(0)
        result_img = Image.open(buf).copy()
        plt.close()

        # ── Detection report ───────────────────────────────────
        if detected:
            info = (
                f"⚠️  TUMOR DETECTED\n"
                f"{'─'*35}\n"
                f"Location      : {location}\n"
                f"Best slice    : Slice {best_slice} (★ highlighted)\n"
                f"Active slices : {active_slices} / 8\n"
                f"Nodule size   : {total_px} pixels\n"
                f"Confidence    : {max_conf*100:.1f}%\n"
                f"Risk level    : {risk}\n"
                f"{'─'*35}\n"
                f"⚠ This is a research demo only.\n"
                f"Consult a radiologist for diagnosis."
            )
        else:
            info = (
                f"✅  NO TUMOR DETECTED\n"
                f"{'─'*35}\n"
                f"Max confidence : {max_conf*100:.1f}%\n"
                f"Nodule pixels  : {total_px}\n"
                f"{'─'*35}\n"
                f"⚠ This is a research demo only.\n"
                f"Consult a radiologist for diagnosis."
            )

        return result_img, info

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ── Gradio UI ──────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Monochrome(),
               title="Lung Tumor Segmentation") as demo:

    gr.Markdown("""
    # 🫁 Lung Tumor Segmentation — AI Demo
    **Model:** 3D U-Net | **Dataset:** LIDC-IDRI | **Val Dice:** 0.7484  
    Upload a `.npy` CT volume → Model segments and detects tumor region
    """)

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.File(label="Upload .npy CT Volume", file_types=[".npy"])
            btn = gr.Button("🔍 Run Detection", variant="primary")
            gr.Markdown("""
            **How to prepare input:**
```python
            import numpy as np
            sample = X_val[0]  # (8,128,128,1)
            np.save("sample.npy", sample)
```
            Upload `sample.npy` here.
            """)

        with gr.Column(scale=2):
            out_img  = gr.Image(label="Detection Result")
            out_text = gr.Textbox(label="🏥 Detection Report", lines=10)

    btn.click(fn=predict, inputs=inp, outputs=[out_img, out_text])

    gr.Markdown("""
    ---
    > ★ = Best slice (most tumor pixels) | Red box = Tumor location  
    > Row 1: Input CT | Row 2: Confidence heatmap | Row 3: Segmentation overlay  
    > Built by **TanishDevX** | LIDC-IDRI | 3D U-Net ~1.4M params
    """)

demo.launch()
