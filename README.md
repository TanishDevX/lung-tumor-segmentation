# 🫁 Lung Tumor Segmentation using 3D Attention U-Net

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Dataset](https://img.shields.io/badge/Dataset-LIDC--IDRI-green)
![Demo](https://img.shields.io/badge/Demo-HuggingFace-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A deep learning pipeline for automated lung tumour segmentation from 3D CT scans, built using a 3D Attention U-Net trained progressively across 3 stages on the LIDC-IDRI dataset. Final model achieves **Test Dice 0.7842** with only 1.4M parameters on a free T4 GPU.

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 🚀 Live Demo | [Hugging Face Spaces](https://huggingface.co/spaces/TanishDevX/lung-tumor-segmentation) |
| 📂 GitHub Repo | [TanishDevX/lung-tumor-segmentation](https://github.com/TanishDevX/lung-tumor-segmentation) |
| 🗂️ Dataset | [LIDC-IDRI on TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/) |

> 💡 **Quick Test:** Download any `.npy` file from [`samples/`](./samples) and upload it to the demo — no setup needed!

---

## 📌 Problem Statement

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection of pulmonary nodules in CT scans is critical for improving survival rates. Manual annotation by radiologists is time-consuming, subjective, and prone to inter-observer variability.

This project addresses **automated lung tumour segmentation** from 3D CT volumes — identifying the exact location and boundary of nodules across multiple CT slices simultaneously.

**Key challenges:**
- Small nodule size relative to full CT volume — severe class imbalance
- High variability in nodule shape, size, and location across patients
- Boundary nodules attached to pleural walls are hard to distinguish
- Limited annotated training data (~2,629 samples after preprocessing)

---

## 💡 Solution

A **3D Attention U-Net** was chosen for its ability to capture spatial context across multiple CT slices simultaneously using 3D convolutions, while attention gates focus the model on tumour-relevant regions and suppress irrelevant background activations. The model was trained progressively across 3 stages, each building on the previous one's saved weights.

**Key design decisions:**
- 3D convolutions over 2D — captures inter-slice spatial relationships
- Attention gates at every skip connection — focuses learning on tumour regions
- Focal + Dice combined loss — handles severe class imbalance better than Dice alone
- Progressive fine-tuning — avoids overfitting on a small dataset
- SpatialDropout3D + BatchNormalization — regularisation at every conv block

---

## 🗂️ Dataset

**LIDC-IDRI** (Lung Image Database Consortium and Image Database Resource Initiative)

| Property | Details |
|----------|---------|
| Source | The Cancer Imaging Archive (TCIA) |
| Format | NumPy arrays (X_train, y_train, X_val, y_val, X_test, y_test) |
| Train / Val / Test | ~1,840 / ~394 / 395 samples |
| Input shape | (N, 8, 128, 128, 1) |
| Masks | 4 annotator masks merged via max projection |

**Preprocessing pipeline:**
- Resize all slices to 128×128
- Normalize pixel values to [0, 1]
- Pad or crop volumes to fixed depth of 8 slices
- Binarize masks (threshold > 0)
- Transpose to (N, 8, 128, 128, 1) for 3D convolution compatibility

---

## 🏗️ Model Architecture

**3D Attention U-Net — 1,478,868 parameters (~1.4M)**

| Component | Details |
|-----------|--------|
| Input | (N, 8, 128, 128, 1) |
| Encoder Block 1 | Conv3D × 2, 16 filters, BN + ReLU + SpatialDropout3D → MaxPool3D (1,2,2) |
| Encoder Block 2 | Conv3D × 2, 32 filters, BN + ReLU + SpatialDropout3D → MaxPool3D (1,2,2) |
| Encoder Block 3 | Conv3D × 2, 64 filters, BN + ReLU + SpatialDropout3D → MaxPool3D (1,2,2) |
| Encoder Block 4 | Conv3D × 2, 128 filters, BN + ReLU + SpatialDropout3D → MaxPool3D (1,2,2) |
| Decoder Blocks | UpSampling3D + Attention Gate + Skip connection + Conv3D × 2 |
| Output | Conv3D 1 filter, Sigmoid activation |

Attention gates at every skip connection learn to suppress irrelevant activations and focus on tumour regions before features are merged in the decoder. Pooling uses (1, 2, 2) — spatial downsampling only, depth preserved. Sigmoid output produces a per-voxel probability map thresholded at 0.5 for binary segmentation.

---

## 📓 Notebook Walkthrough

### [`EDA.ipynb`](./notebooks/EDA.ipynb) — Exploratory Data Analysis
- Explored LIDC-IDRI structure: patients, nodules, slices, annotator masks
- Visualised raw CT slices alongside ground truth masks
- Analysed nodule size distribution and slice depth variability
- Identified class imbalance — informed loss function choice
- Decided 128×128 resolution and 8-slice depth from data distribution

### [`Attention_M1.ipynb`](./notebooks/Attention_M1.ipynb) — Baseline Attention U-Net
- Designed the 3D Attention U-Net architecture from scratch
- Loss: BCE + Dice (0.5 each) | Optimizer: Adam | Epochs: 30 | Batch size: 4
- Model was still improving at epoch 30 — not yet converged
- Saved weights: `attention_unet_best.keras`
- **Result: Val Dice 0.7429 | Test Dice 0.7234**

### [`Attention_M2.ipynb`](./notebooks/Attention_M2.ipynb) — Fine-tuning Attempt
- Loaded NB1 weights; unfroze all layers except BatchNorm
- Cosine LR schedule: 1e-4 → 1e-6 | Batch size: 2 | Early stopped at epoch 11/21
- Cosine LR decayed too fast (~920 steps/epoch) — premature convergence
- Saved weights: `M4_finetuned_best.keras`
- **Result: Val Dice 0.7471 | Test Dice 0.7380** — marginal improvement, limited by LR schedule

### [`Attention_M3.ipynb`](./notebooks/Attention_M3.ipynb) — Final Best Model
- Loaded NB2 weights; switched loss to **Focal + Dice** (0.5 each)
- Warmup LR: 2e-5 → 8e-5 over first epochs, then ReduceLROnPlateau
- Unfroze BatchNorm at epoch 5 — allowed running stats to adapt to new batch size
- Trained for 50 epochs, patience 15 | Batch size: 2
- Saved weights: `NB3_best.keras` ← **final best model**
- **Result: Val Dice 0.7700 | Test Dice 0.7842** — best across all stages

### [`Deployment.ipynb`](./notebooks/Deployment.ipynb) — Evaluation & Deployment
- Full metric evaluation on held-out test set: Dice, IoU, Sensitivity, Precision, Specificity
- Threshold sensitivity analysis — optimal at 0.50
- Failure case analysis: sub-centimeter nodules and boundary nodules identified
- Built and deployed Gradio app on Hugging Face Spaces
- Saved demo samples for public testing

---

## 📊 Results

### Model Progression
| Stage | Val Dice | Test Dice | Loss | Key Change |
|-------|----------|-----------|------|------------|
| NB1 Baseline | 0.7429 | 0.7234 | BCE + Dice | Base attention architecture |
| NB2 Fine-tune | 0.7471 | 0.7380 | BCE + Dice | Cosine LR, unfroze encoder |
| NB3 Final | **0.7700** | **0.7842** | Focal + Dice | Warmup LR + BatchNorm unfreeze |

### Why NB3 Over NB2?
NB2 used a cosine LR schedule that decayed too aggressively (~920 steps/epoch), causing premature convergence at epoch 11. NB3 fixed this with a warmup schedule and introduced Focal + Dice loss:
```python
loss = 0.5 * focal_loss(gamma=2.0, alpha=0.25) + 0.5 * (1 - dice_coefficient)
```
Focal loss down-weights easy background voxels and focuses learning on hard tumour positives — critical for the severe class imbalance in this dataset. Dice handles region-level overlap quality.

### Final Test Set Metrics — NB3 vs NB1
| Metric | NB1 Baseline | NB2 | NB3 Final | Δ vs NB1 |
|--------|-------------|-----|-----------|----------|
| Dice | 0.7234 | 0.7380 | **0.7842** | +0.0608 ▲ |
| IoU | 0.6031 | 0.6172 | **0.6638** | +0.0607 ▲ |
| Sensitivity | 0.7168 | 0.7957 | **0.8136** | +0.0968 ▲ |
| Precision | 0.8229 | 0.7528 | 0.7999 | -0.0230 ▼ |
| Specificity | 0.9994 | 0.9990 | **0.9993** | +0.0003 ▲ |
| Dice Std | 0.2219 | 0.2051 | **0.1465** | -0.0754 ▼ (better) |

> Dice improved +6 points from baseline to final. Sensitivity rose +9.7 points — model detects significantly more real tumours. Dice Std dropped from 0.22 → 0.15, meaning predictions became more consistent across patients.

### Performance vs Published Research
| Metric | Our Model (NB3) | Typical LIDC-IDRI Range |
|--------|----------------|-------------------------|
| Test Dice | **0.7842** | 0.70 – 0.85 |
| IoU | 0.6638 | 0.60 – 0.75 |
| Sensitivity | 0.8136 | 0.75 – 0.85 |
| Parameters | 1.47M | 5M – 50M+ typical |

> Achieving 0.78 Dice with 1.4M parameters on a free T4 GPU — most published results with similar scores use far larger models and multi-GPU setups.

---

## 🔄 Training Pipeline

```
LIDC-IDRI Dataset
       ↓
Preprocessing (resize → normalize → pad → binarize)
       ↓
NB1 — Train 3D Attention U-Net from scratch (BCE+Dice, 30 epochs)    → Test Dice 0.7234
       ↓
NB2 — Fine-tune with cosine LR, unfreeze encoder                      → Test Dice 0.7380
       ↓
NB3 — Focal+Dice loss, warmup LR, BatchNorm unfreeze at epoch 5       → Test Dice 0.7842 ✓ Best
       ↓
Evaluation on held-out test set + Gradio Deployment
```

---

## 🚀 Demo & Usage

### Option 1 — Live Demo (No Setup Required)
1. Go to [Hugging Face Space](https://huggingface.co/spaces/TanishDevX/lung-tumor-segmentation)
2. Download any `.npy` file from [`samples/`](./samples)
3. Upload and click **Run Detection**
4. View segmentation across all 8 slices — heatmap, overlay, and detection report

### Option 2 — Prepare Your Own Sample
```python
import numpy as np

X_val = np.load('X_val.npy')  # shape: (N, 8, 128, 128, 1)
sample = X_val[0]             # shape: (8, 128, 128, 1)
np.save('my_sample.npy', sample)
# Upload my_sample.npy to the demo
```

### What the App Shows
- 3-row visualisation grid per slice: CT scan | raw heatmap | overlay with tumour mask
- Bounding box on the slice with the highest tumour voxel count
- Location label: e.g. *Upper-Right region*, *Lower-Left region*
- Summary panel: detection status, voxel count, max confidence, best slice index

### Sample Files Included
| Files | Count | Description |
|-------|-------|-------------|
| `demo_sample_0` to `demo_sample_9` | 10 | Best predictions — Dice > 0.85 |
| `random_sample_0` to `random_sample_4` | 5 | Random samples — mixed difficulty |

---

## 📁 Repository Structure

```
lung-tumor-segmentation/
│
├── notebooks/
│   ├── EDA.ipynb                       # Dataset exploration & analysis
│   ├── Attention_M1.ipynb              # Baseline 3D Attention U-Net training
│   ├── Attention_M2.ipynb              # Fine-tuning attempt (cosine LR)
│   ├── Attention_M3.ipynb              # Final model — Focal+Dice, warmup LR
│   └── Deployment.ipynb                # Metrics, evaluation & Gradio app
│
├── app/
│   ├── app.py                          # Gradio web application
│   └── requirements.txt               # Python dependencies
│
├── results/
│   ├── M2_training_log.csv            # NB1 training history
│   ├── M2_phase2_log.csv              # NB2 training history
│   ├── M2_phase2_retry_log.csv        # NB2 retry history
│   ├── M3_training_log.csv            # NB3 training history
│   └── *.png                          # Dice score training plots
│
├── samples/
│   ├── demo_sample_0.npy .. 9.npy     # Best predictions (Dice > 0.85)
│   └── random_sample_0.npy .. 4.npy   # Random validation samples
│
└── README.md
```

---

## ⚙️ Environment

| Tool | Details |
|------|---------|
| Platform | Google Colab (T4 GPU) |
| Python | 3.10 |
| TensorFlow | 2.x |
| Batch size | 4 (NB1) → 2 (NB2 onwards) |
| Optimizer | Adam + ReduceLROnPlateau + EarlyStopping |
| Regularisation | L2 (1e-4) + SpatialDropout3D + BatchNormalization |

---

## ⚠️ Known Limitations

| Limitation | Details |
|------------|---------|
| Mild overfitting | Train Dice ~0.82 vs Val Dice ~0.77 after epoch 20 (gap ~0.05) |
| Sub-centimeter nodules | Nodules < 50 voxels frequently missed |
| Architecture ceiling | Filters 16→32→64→128 maxed out for T4 memory |
| Input format | App requires exact `.npy` format — no DICOM or NIfTI support yet |
| Clinical use | Not validated for clinical use — **research only** |

---

## 🔮 Future Improvements

- **Wider architecture** — increase filters to 32→64→128→256 (requires more GPU memory)
- **Data augmentation** — random flips, rotations, elastic deformations during training
- **Test-Time Augmentation (TTA)** — average predictions over augmented inputs (+0.01–0.02 Dice expected)
- **Ensemble NB1 + NB3** — different biases, complementary errors
- **DICOM / NIfTI input** — real-world usability in the Gradio app

---

## 👤 Author

**TanishDevX**  
[Hugging Face](https://huggingface.co/TanishDevX) • [GitHub](https://github.com/TanishDevX)

---

> ⚠️ This project is for educational and research purposes only. Not intended for clinical diagnosis or medical use.
