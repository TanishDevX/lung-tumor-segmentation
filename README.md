# 🫁 Lung Tumor Segmentation using 3D U-Net

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Dataset](https://img.shields.io/badge/Dataset-LIDC--IDRI-green)
![Demo](https://img.shields.io/badge/Demo-HuggingFace-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A deep learning pipeline for automated lung nodule segmentation from 3D CT scans, built using a custom 3D U-Net architecture trained progressively across 4 stages on the LIDC-IDRI dataset.

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

This project addresses **automated lung nodule segmentation** from 3D CT volumes — identifying the exact location and boundary of nodules across multiple CT slices simultaneously.

**Key challenges:**
- Small nodule size relative to full CT volume — severe class imbalance
- High variability in nodule shape, size, and location across patients
- Boundary nodules attached to pleural walls are hard to distinguish
- Limited annotated training data (~2630 samples after preprocessing)

---

## 💡 Solution

A **3D U-Net** was chosen for its ability to capture spatial context across multiple CT slices simultaneously using 3D convolutions. The model was trained progressively across 4 stages, each addressing a specific limitation of the previous one.

**Key design decisions:**
- 3D convolutions over 2D — captures inter-slice spatial relationships
- BCE + Dice combined loss — handles class imbalance better than Dice alone
- Progressive fine-tuning — avoids overfitting on small dataset
- Skip connections — preserves spatial detail in decoder path

---

## 🗂️ Dataset

**LIDC-IDRI** (Lung Image Database Consortium and Image Database Resource Initiative)

| Property | Details |
|----------|---------|
| Source | The Cancer Imaging Archive (TCIA) |
| Format | PNG slices per nodule |
| Structure | patient → nodule → images + mask-0..3 |
| Total samples | 2630 nodule volumes |
| Train / Val | 2104 / 526 (80/20 split) |
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

**3D U-Net — ~1.4M parameters**

| Component | Details |
|-----------|--------|
| Input | (N, 8, 128, 128, 1) |
| Encoder Block 1 | Conv3D × 2, 32 filters, BN + ReLU → MaxPool3D (1,2,2) |
| Encoder Block 2 | Conv3D × 2, 64 filters, BN + ReLU → MaxPool3D (1,2,2) |
| Bottleneck | Conv3D × 2, 128 filters, BN + ReLU |
| Decoder Block 1 | UpSampling3D + Skip connection + Conv3D × 2, 64 filters |
| Decoder Block 2 | UpSampling3D + Skip connection + Conv3D × 2, 32 filters |
| Output | Conv3D 1 filter, Sigmoid activation |

Skip connections from encoder to decoder preserve fine-grained spatial detail. Sigmoid output produces a per-voxel probability map thresholded at 0.5 for binary segmentation.

---

## 📓 Notebook Walkthrough

### [`M1_EDA.ipynb`](./notebooks/M1_EDA.ipynb) — Exploratory Data Analysis
- Explored LIDC-IDRI structure: patients, nodules, slices, annotator masks
- Visualised raw CT slices alongside ground truth masks
- Analysed nodule size distribution and slice depth variability
- Identified class imbalance — informed loss function choice in M4
- Decided 128×128 resolution and 8-slice depth from data distribution

### [`M2_BaseModel.ipynb`](./notebooks/M2_BaseModel.ipynb) — Baseline 3D U-Net
- Built 3D U-Net from scratch with encoder-bottleneck-decoder structure
- Loss: Dice only | Optimizer: Adam (lr=1e-4) | Epochs: 20
- Saved preprocessed arrays to Drive for reuse across all notebooks
- **Result: Val Dice 0.71**

### [`M3_Augmented.ipynb`](./notebooks/M3_Augmented.ipynb) — Data Augmentation
- Loaded M2 baseline and continued training with on-the-fly augmentation
- Augmentations: random horizontal flip, vertical flip, Gaussian noise
- Custom infinite data generator for memory-efficient augmented batches
- **Result: Val Dice 0.7580** — note: Dice-only loss is optimistic on imbalanced data

### [`M4_FineTuning.ipynb`](./notebooks/M4_FineTuning.ipynb) — Fine-tuning
- Replaced Dice-only loss with **BCE + Dice combined loss** (0.5 weighted)
- Phase 1: Froze encoder (layers 0–27), trained decoder → Val Dice **0.7441**
- Phase 2: Unfroze layers 28–37, LR=2e-5 → Val Dice **0.7484** (best)
- Used ReduceLROnPlateau + EarlyStopping to control overfitting

### [`M5_Evaluation_Deployment.ipynb`](./notebooks/M5_Evaluation_Deployment.ipynb) — Evaluation & Deployment
- Full metric evaluation: Dice, IoU, Sensitivity, Precision, Specificity
- Failure case analysis: boundary nodules and sub-centimeter nodules identified
- Threshold sensitivity analysis (0.25–0.55) — optimal at 0.50
- Built and deployed Gradio app on Hugging Face Spaces
- Saved 15 demo samples for public testing

---

## 📊 Results

### Model Progression
| Stage | Val Dice | Loss | Key Change |
|-------|----------|------|------------|
| M2 Baseline | 0.7100 | Dice only | Base architecture |
| M3 Augmented | 0.7580 | Dice only | On-the-fly augmentation |
| M4 Phase 1 | 0.7441 | BCE + Dice | Frozen encoder, better loss |
| M4 Phase 2 | **0.7484** | BCE + Dice | Partial decoder unfreeze |

### Why M4 Over M3 Despite Lower Dice?
M3 used Dice loss alone — optimistic on heavily imbalanced data. M4 introduced BCE + Dice:
```python
loss = 0.5 * BCE + 0.5 * Dice
```
BCE enforces pixel-level correctness; Dice handles class imbalance. The lower M4 Dice reflects a more calibrated model — confirmed by measuring sensitivity and specificity, which were not tracked in M3.

### Final Validation Metrics (M4 Phase 2)
| Metric | Score | Interpretation |
|--------|-------|----------------|
| Dice | 0.7484 | Good overlap overall |
| IoU | 0.5932 | Strict pixel-level overlap |
| Sensitivity | 0.7350 | 73.5% of nodules detected |
| Precision | 0.7951 | 79.5% of predictions correct |
| Specificity | 0.9992 | Near-zero false positive rate |

### Prediction Quality Distribution
| Category | Count | % |
|----------|-------|---|
| Perfect (Dice > 0.85) | 162 | 30.8% |
| Good (0.70 – 0.85) | 227 | 43.2% |
| Mediocre (0.40 – 0.70) | 74 | 14.1% |
| Failed (0.01 – 0.40) | 24 | 4.6% |
| Complete Miss (≤ 0.01) | 39 | 7.4% |

> 74% of validation samples predicted Good to Perfect.

---

## 🔄 Training Pipeline

```
LIDC-IDRI Dataset
       ↓
Preprocessing (resize → normalize → pad → binarize)
       ↓
M2 — Train 3D U-Net from scratch (Dice loss)        → 0.71
       ↓
M3 — Continue with on-the-fly augmentation           → 0.7580
       ↓
M4 Phase 1 — Freeze encoder, BCE+Dice loss           → 0.7441
       ↓
M4 Phase 2 — Unfreeze layers 28-37, LR=2e-5         → 0.7484 ✓ Best
       ↓
Evaluation + Gradio Deployment
```

---

## 🚀 Demo & Usage

### Option 1 — Live Demo (No Setup Required)
1. Go to [Hugging Face Space](https://huggingface.co/spaces/TanishDevX/lung-tumor-segmentation)
2. Download any `.npy` file from [`samples/`](./samples)
3. Upload and click **Run Detection**
4. View segmentation across all 8 slices with detection report

### Option 2 — Prepare Your Own Sample
```python
import numpy as np

X_val = np.load('X_val.npy')
X_val = np.transpose(X_val, (0, 3, 1, 2, 4))  # → (N, 8, 128, 128, 1)

sample = X_val[0]  # shape: (8, 128, 128, 1)
np.save('my_sample.npy', sample)
# Upload my_sample.npy to the demo
```

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
│   ├── M1_EDA.ipynb                    # Dataset exploration & analysis
│   ├── M2_BaseModel.ipynb              # 3D U-Net baseline training
│   ├── M3_Augmented.ipynb              # Augmentation strategy
│   ├── M4_FineTuning.ipynb             # Fine-tuning Phase 1 & 2
│   └── M5_Evaluation_Deployment.ipynb  # Metrics, evaluation & Gradio app
│
├── app/
│   ├── app.py                          # Gradio web application
│   └── requirements.txt               # Python dependencies
│
├── results/
│   ├── M4_training_log.csv            # Phase 1 training history
│   ├── M4_phase2_log.csv              # Phase 2 training history
│   └── M4_phase2_retry_log.csv        # Phase 2 retry history
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
| Batch size | 2 |
| Optimizer | Adam + ReduceLROnPlateau + EarlyStopping |

---

## ⚠️ Known Limitations

| Limitation | Details |
|------------|---------|
| Boundary nodules | Pleural/chest wall nodules missed — max confidence ~0.32 |
| Sub-centimeter nodules | Nodules < 50 voxels frequently missed |
| Dataset size | ~2630 samples limits generalisation |
| Clinical use | Not validated for clinical use — **research only** |

---

## 🔮 Future Improvements

- **Test Time Augmentation (TTA)** — average predictions over augmented inputs (+0.01–0.02 Dice)
- **Attention U-Net** — attention gates to focus on nodule regions
- **Phase 3 fine-tuning** — unfreeze deeper encoder layers with very low LR
- **Focal + Dice loss** — penalise hard negatives more aggressively
- **Ensemble M3 + M4** — combine model predictions for marginal gain

---

## 👤 Author

**TanishDevX**  
[Hugging Face](https://huggingface.co/TanishDevX) • [GitHub](https://github.com/TanishDevX)

---

> ⚠️ This project is for educational and research purposes only. Not intended for clinical diagnosis or medical use.

---

## 🤝 Contributors

| Name | GitHub | Role |
|------|--------|------|
| TanishDevX | [@TanishDevX](https://github.com/TanishDevX) | Project Lead |
| | | |
| | | |

