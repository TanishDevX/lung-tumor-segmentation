
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

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    bce     = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    focal_w = alpha * tf.pow(1.0 - p_t, gamma)
    return tf.reduce_mean(focal_w * bce)

def focal_dice_loss(y_true, y_pred):
    return 0.5 * focal_loss(y_true, y_pred) + 0.5 * (1.0 - dice_coefficient(y_true, y_pred))

# ── Load model ─────────────────────────────────────────────────
model = tf.keras.models.load_model(
    "lung_tumor_model.keras",
    custom_objects={
        "focal_dice_loss":  focal_dice_loss,
        "dice_coefficient": dice_coefficient,
        "focal_loss":       focal_loss
    }
)
print("Model loaded ✓")

# ── Helper: bounding box ───────────────────────────────────────
def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, cmin, rmax, cmax

# ── Helper: location label ─────────────────────────────────────
def get_location(bbox, img_size=128):
    if bbox is None:
        return "N/A"
    rmin, cmin, rmax, cmax = bbox
    cx = (cmin + cmax) / 2
    cy = (rmin + rmax) / 2
    v = "Upper" if cy < img_size // 2 else "Lower"
    h = "Left"  if cx < img_size // 2 else "Right"
    return f"{v}-{h} region"

# ── Predict ────────────────────────────────────────────────────
def predict(npy_file):
    try:
        volume = np.load(npy_file.name)

        if volume.ndim == 3:
            volume = volume[..., np.newaxis]
        if volume.shape == (128, 128, 8, 1):
            volume = np.transpose(volume, (2, 0, 1, 3))
        if volume.max() > 1.0:
            volume = volume / 255.0

        if volume.shape != (8, 128, 128, 1):
            return None, f"❌ Wrong shape: {volume.shape}. Expected (8,128,128,1)"

        inp  = volume[np.newaxis, ...].astype(np.float32)
        pred = model.predict(inp, verbose=0)[0]
        pred_bin = (pred > 0.45).astype(np.float32)

        total_voxels   = int(np.sum(pred_bin))
        max_confidence = float(pred.max())
        active_slices  = int(np.sum([pred_bin[s].sum() > 0 for s in range(8)]))
        detected       = total_voxels > 10 and max_confidence > 0.45

        slice_counts = [int(pred_bin[s].sum()) for s in range(8)]
        best_slice   = int(np.argmax(slice_counts))
        bbox         = get_bbox(pred_bin[best_slice, :, :, 0])
        location     = get_location(bbox)

        if total_voxels > 800:
            risk, risk_color = "High",   "#ff4444"
        elif total_voxels > 300:
            risk, risk_color = "Medium", "#ffaa00"
        else:
            risk, risk_color = "Low",    "#44ff44"

        fig, axes = plt.subplots(3, 8, figsize=(22, 9))
        fig.patch.set_facecolor("#0f0f0f")

        status_text  = "⚠ TUMOUR DETECTED"  if detected else "✓ NO TUMOUR DETECTED"
        status_color = "#ff4444"             if detected else "#44ff88"
        fig.suptitle(status_text, color=status_color,
                     fontsize=16, fontweight="bold", y=1.02)

        for s in range(8):
            is_best    = (s == best_slice)
            ct_slice   = volume[s, :, :, 0]
            pred_slice = pred[s, :, :, 0]
            bin_slice  = pred_bin[s, :, :, 0]
            border_col = "#ff4444" if is_best else "#333333"

            axes[0, s].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
            axes[0, s].set_title(
                f"Slice {s}" + (" ★" if is_best else ""),
                color="#ffdd00" if is_best else "white",
                fontsize=8, fontweight="bold" if is_best else "normal"
            )
            axes[0, s].axis("off")
            axes[0, s].set_facecolor("#0f0f0f")
            for spine in axes[0, s].spines.values():
                spine.set_edgecolor(border_col)
                spine.set_linewidth(2 if is_best else 0.5)

            axes[1, s].imshow(pred_slice, cmap="hot", vmin=0, vmax=1)
            axes[1, s].set_title(f"conf={pred_slice.max():.2f}", color="white", fontsize=7)
            axes[1, s].axis("off")
            axes[1, s].set_facecolor("#0f0f0f")

            axes[2, s].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
            if bin_slice.max() > 0:
                overlay = np.zeros((*bin_slice.shape, 4))
                overlay[bin_slice == 1] = [1, 0.2, 0.2, 0.55]
                axes[2, s].imshow(overlay)
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

            axes[2, s].set_title(f"{int(bin_slice.sum())}px", color="white", fontsize=7)
            axes[2, s].axis("off")
            axes[2, s].set_facecolor("#0f0f0f")

        for row, label in enumerate(["CT Scan", "Heatmap", "Overlay"]):
            axes[row, 0].set_ylabel(label, color="white", fontsize=9, labelpad=6)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120,
                    bbox_inches="tight", facecolor="#0f0f0f")
        plt.close(fig)
        buf.seek(0)
        result_img = Image.open(buf).copy()

        if detected:
            report = (
                f"⚠️  TUMOUR DETECTED\n"
                f"{'─'*35}\n"
                f"Location      : {location}\n"
                f"Best slice    : Slice {best_slice} (★ highlighted)\n"
                f"Active slices : {active_slices} / 8\n"
                f"Nodule size   : {total_voxels} voxels\n"
                f"Confidence    : {max_confidence*100:.1f}%\n"
                f"Risk level    : {risk}\n"
                f"{'─'*35}\n"
                f"⚠ Research demo only.\n"
                f"Consult a radiologist for diagnosis."
            )
        else:
            report = (
                f"✅  NO TUMOUR DETECTED\n"
                f"{'─'*35}\n"
                f"Max confidence : {max_confidence*100:.1f}%\n"
                f"Nodule voxels  : {total_voxels}\n"
                f"Active slices  : {active_slices} / 8\n"
                f"{'─'*35}\n"
                f"⚠ Research demo only.\n"
                f"Consult a radiologist for diagnosis."
            )

        return result_img, report

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ── Gradio UI ──────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Monochrome(),
    title="Lung Tumour Segmentation",
    css="""
    body, .gradio-container, .main, footer {
        background-color: #0f0f0f !important;
        color: white !important;
    }
    html {
        background-color: #0f0f0f !important;
    }
    .block, .form, .box, .panel {
        background-color: #1a1a1a !important;
        border-color: #333333 !important;
    }
    textarea, input, .input-text {
        background-color: #1a1a1a !important;
        color: white !important;
    }
    button.primary {
        background-color: #cc0000 !important;
    }
    """
) as demo:

    gr.Markdown("""
    # 🫁 Lung Tumour Segmentation — AI Demo
    **Model:** 3D Attention U-Net | **Dataset:** LIDC-IDRI | **Test Dice:** 0.7842 | **Sensitivity:** 0.8136
    Upload a `.npy` CT volume of shape `(8, 128, 128, 1)` → Model segments and detects tumour region.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.File(label="Upload .npy CT Volume", file_types=[".npy"])
            btn = gr.Button("🔍 Run Detection", variant="primary")
            gr.Markdown("""
            **How to prepare input:**
```python
import numpy as np
sample = X_val[0]        # shape (8,128,128,1)
np.save("sample.npy", sample)
```
            Then upload `sample.npy` here.
            """)

        with gr.Column(scale=2):
            out_img = gr.Image(label="Detection Result", type="pil")
            out_txt = gr.Textbox(label="🏥 Detection Report", lines=12)

    btn.click(fn=predict, inputs=inp, outputs=[out_img, out_txt])

    gr.Markdown("""
    ---
    > ★ = Best slice (most tumour voxels) | Red dashed box = Tumour location
    > Row 1: Input CT | Row 2: Confidence heatmap | Row 3: Segmentation overlay
    > **Built by TanishDevX** | LIDC-IDRI | 3D Attention U-Net | Focal+Dice Loss | Threshold: 0.45
    """)

demo.launch()
