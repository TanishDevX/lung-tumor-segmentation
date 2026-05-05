# 🫁 Lung Tumor Segmentation using 3D Attention U-Net

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Dataset](https://img.shields.io/badge/Dataset-LIDC--IDRI-green)
![Demo](https://img.shields.io/badge/Demo-HuggingFace-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A deep learning pipeline for automated lung tumour segmentation from 3D CT scans, built using a 3D Attention U-Net trained progressively across 4 stages on the LIDC-IDRI dataset. Best model achieves **Test Dice 0.7842** with only 1.4M parameters on a free T4 GPU.

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 🚀 Live Demo | [Hugging Face Spaces](https://huggingface.co/spaces/TanishDevX/lung-tumor-segmentation) |
| 📂 GitHub Repo | [TanishDevX/lung-tumor-segmentation](https://github.com/TanishDevX/lung-tumor-segmentation) |
| 🗂️ Dataset | [LIDC-IDRI on Kaggle](https://www.kaggle.com/datasets/zhangweiled/lidcidri/data) |

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

A **3D Attention U-Net** was chosen for its ability to capture spatial context across multiple CT slices simultaneously, while attention gates focus the model on tumour-relevant regions and suppress irrelevant background activations. The model was trained progressively across 4 stages, each building on the previous one's saved weights.

**Key design decisions:**
- 3D convolutions over 2D — captures inter-slice spatial relationships
- Attention gates at every skip connection — focuses learning on tumour regions
- Focal + Dice combined loss — handles severe class imbalance better than BCE alone
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

Attention gates at every skip connection suppress irrelevant activations and focus on tumour regions. Pooling uses (1, 2, 2) — spatial downsampling only, depth preserved. Sigmoid output produces a per-voxel probability map thresholded at 0.5.

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
- **Note:** augmentation was written using Python `if` inside `tf.data.map()` — a silent bug that caused conditions to be evaluated once at graph build time, not per sample. Fixed in NB4 using `tf.cond`.
- Saved weights: `attention_unet_best.keras`
- **Result: Val Dice 0.7429 | Test Dice 0.7234**

### [`Attention_M2.ipynb`](./notebooks/Attention_M2.ipynb) — Fine-tuning Attempt
- Loaded NB1 weights; unfroze all layers except BatchNorm
- Cosine LR schedule: 1e-4 → 1e-6 | Batch size: 2 | Early stopped at epoch 11/21
- Cosine LR decayed too fast (~920 steps/epoch) — premature convergence
- Saved weights: `M4_finetuned_best.keras`
- **Result: Val Dice 0.7471 | Test Dice 0.7380**

### [`Attention_M3.ipynb`](./notebooks/Attention_M3.ipynb) — Best Model ✓
- Loaded NB2 weights; switched loss to **Focal + Dice** (0.5 each)
- Warmup LR: 2e-5 → 8e-5, then ReduceLROnPlateau | BatchNorm unfrozen at epoch 5
- Trained 50 epochs, patience 15 | Batch size: 2
- Saved weights: `NB3_best.keras` ← **best model, used in deployment**
- **Result: Val Dice 0.7700 | Test Dice 0.7842**

### [`Attention_M4.ipynb`](./notebooks/Attention_M4.ipynb) — Augmentation Experiment
- Identified and fixed the `tf.cond` augmentation bug from NB1
- Loaded NB3 weights; fine-tuned with H-flip, V-flip, depth-flip, Gaussian noise, brightness jitter
- All augmentations use `tf.cond` — evaluated per sample inside the tf graph
- LR warmup: 5e-6 → 2e-5 | 30 epochs, patience 10 | Batch size: 2
- **Result: Val Dice 0.7631 | Test Dice 0.7672 — did not beat NB3**
- *Analysis: NB3 had already converged to a strong optimum. Augmentation-based perturbation disrupted well-tuned weights without enough epochs to recover. Dice Std increased (0.1465 → 0.1870) — less consistent predictions. NB3 remains best.*

### [`Deployment.ipynb`](./notebooks/Deployment.ipynb) — Evaluation & Deployment
- Full metric evaluation on held-out test set: Dice, IoU, Sensitivity, Precision, Specificity
- Threshold sensitivity analysis — optimal at 0.50
- Failure case analysis: sub-centimeter nodules and boundary nodules identified
- Built and deployed Gradio app on Hugging Face Spaces using NB3_best.keras

---

## 📊 Results

### Model Progression
| Stage | Val Dice | Test Dice | Loss | Key Change |
|-------|----------|-----------|------|------------|
| NB1 Baseline | 0.7429 | 0.7234 | BCE + Dice | Base attention architecture |
| NB2 Fine-tune | 0.7471 | 0.7380 | BCE + Dice | Cosine LR, unfroze encoder |
| NB3 Final | **0.7700** | **0.7842** | Focal + Dice | Warmup LR + BatchNorm unfreeze |
| NB4 Augmentation | 0.7631 | 0.7672 | Focal + Dice | Fixed tf.cond augmentation |

### Final Metrics — NB3 (Best) vs All Stages
| Metric | NB1 | NB2 | NB3 ✓ | NB4 |
|--------|-----|-----|--------|-----|
| Dice | 0.7234 | 0.7380 | **0.7842** | 0.7672 |
| IoU | 0.6031 | 0.6172 | **0.6638** | 0.6501 |
| Sensitivity | 0.7168 | 0.7957 | **0.8136** | 0.7895 |
| Precision | 0.8229 | 0.7528 | 0.7999 | **0.8041** |
| Specificity | 0.9994 | 0.9990 | **0.9993** | 0.9992 |
| Dice Std | 0.2219 | 0.2051 | **0.1465** | 0.1870 |

### Why NB4 Didn't Beat NB3
NB3 had already converged to a strong optimum over 50 epochs. Fine-tuning from this point with augmentation effectively perturbed well-calibrated weights — the model needed more epochs to adapt to the augmented distribution than the 30 allowed. The higher Dice Std in NB4 (0.187 vs 0.147) confirms the model became less consistent. This is a known risk of augmentation fine-tuning from a converged model; the correct approach would be to train NB1 from scratch with augmentation enabled from epoch 1.

### Performance vs Published Research
| Metric | Our Best (NB3) | Typical LIDC-IDRI Range |
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
NB1 — Train 3D Attention U-Net from scratch (BCE+Dice, 30 epochs)     → Test Dice 0.7234
       ↓
NB2 — Fine-tune with cosine LR, unfreeze encoder                       → Test Dice 0.7380
       ↓
NB3 — Focal+Dice loss, warmup LR, BatchNorm unfreeze at epoch 5        → Test Dice 0.7842 ✓ Best
       ↓
NB4 — Fixed tf.cond augmentation fine-tune from NB3                    → Test Dice 0.7672 (experiment)
       ↓
Evaluation on held-out test set + Gradio Deployment (NB3 model)
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

### Sample Files Included
| Files | Count | Description |
|-------|-------|-------------|
| `demo_sample_XXX` to `demo_sample_XXX` | 10 | Best predictions — Dice > 0.85 |
---

## 📁 Repository Structure

```
lung-tumor-segmentation/
│
├── notebooks/
│   ├── EDA.ipynb                       # Dataset exploration & analysis
│   ├── Attention_M1.ipynb              # Baseline 3D Attention U-Net
│   ├── Attention_M2.ipynb              # Fine-tuning attempt (cosine LR)
│   ├── Attention_M3.ipynb              # Best model — Focal+Dice, warmup LR
│   ├── Attention_M4.ipynb              # Augmentation experiment (tf.cond fix)
│   └── Deployment.ipynb                # Metrics, evaluation & Gradio app
│
├── app/
│   ├── app.py                          # Gradio web application
│   └── requirements.txt               # Python dependencies
│
├── results/
│   ├── M2_training_log.csv            # NB1 training history
│   ├── M2_phase2_log.csv              # NB2 training history
│   ├── M3_training_log.csv            # NB3 training history
│   ├── M4_training_log.csv            # NB4 training history
│   └── *.png                          # Dice & loss plots for all stages
│
├── samples/
│   ├── demo_sample_X.npy .. XX.npy
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
| Mild overfitting | Train Dice ~0.82 vs Val Dice ~0.77 in NB3 (gap ~0.05) |
| Sub-centimeter nodules | Nodules < 50 voxels frequently missed |
| Architecture ceiling | Filters 16→32→64→128 maxed out for T4 memory |
| Augmentation timing | NB4 showed augmentation fine-tune from converged model is ineffective — needs to be applied from NB1 |
| Input format | App requires exact `.npy` format — no DICOM or NIfTI support yet |
| Clinical use | Not validated for clinical use — **research only** |

---

## 🔮 Future Improvements

- **Retrain NB1 with augmentation from scratch** — NB4 showed that augmentation fine-tuning from a converged model is ineffective; applying it from epoch 1 is the correct approach and would likely yield +0.01–0.03 Dice
- **Wider architecture** — increase filters to 32→64→128→256 (requires more GPU memory)
- **Test-Time Augmentation (TTA)** — average predictions over augmented inputs
- **Ensemble NB1 + NB3** — different biases, complementary errors
- **DICOM / NIfTI input** — real-world usability in the Gradio app

---

## 👤 Author

**TanishDevX**  
[Hugging Face](https://huggingface.co/TanishDevX) • [GitHub](https://github.com/TanishDevX)

---

> ⚠️ This project is for educational and research purposes only. Not intended for clinical diagnosis or medical use.
