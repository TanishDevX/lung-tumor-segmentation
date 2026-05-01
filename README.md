# 🫁 Lung Tumor Segmentation

A 3D U-Net based lung tumor segmentation system trained on the LIDC-IDRI dataset, with progressive model improvement across 4 stages.

## 🔗 Live Demo
**[Try it on Hugging Face Spaces](https://huggingface.co/spaces/TanishDevX/lung-tumor-segmentation)**

> 💡 **Quick Test:** Download any `.npy` file from the [`samples/`](./samples) folder and upload it directly to the demo — no setup needed!

---

## 🧪 Try It Yourself

1. Go to the [`samples/`](./samples) folder in this repo
2. Download any `.npy` file
3. Open the [Live Demo](https://huggingface.co/spaces/TanishDevX/lung-tumor-segmentation)
4. Upload the `.npy` file and click **Run Detection**
5. View segmentation across all 8 CT slices with tumor location report

### Sample Files Included
| Folder | Count | Description |
|--------|-------|-------------|
| `demo_sample_0` to `demo_sample_9` | 10 files | Best predictions — Dice > 0.85, clear tumor detection |
| `random_sample_0` to `random_sample_4` | 5 files | Random validation samples — mixed difficulty |

Each file contains 8 consecutive CT slices in shape `(8, 128, 128, 1)`.

---

## 📊 Results

| Model | Val Dice | Loss Function | Notes |
|-------|----------|---------------|-------|
| M2 — Baseline 3D U-Net | 0.7100 | Dice only | Base architecture |
| M3 — Augmented | 0.7580 | Dice only | Flips + Gaussian noise |
| M4 Phase 1 — Fine-tuned | 0.7441 | BCE + Dice | Frozen encoder, new loss |
| M4 Phase 2 — Fine-tuned | **0.7484** | BCE + Dice | Partial decoder unfreeze |

### Why M4 Over M3 Despite Lower Dice?
M3 achieved a higher raw Dice score (0.7580) but used Dice loss alone, which
can be misleading on imbalanced medical datasets where background pixels
heavily outnumber nodule pixels.

M4 introduced a combined **BCE + Dice loss**:
```python
loss = 0.5 * BCE + 0.5 * Dice
```
- **BCE** enforces pixel-level accuracy and penalises false positives/negatives directly
- **Dice** handles class imbalance between nodule and background regions
- Together they produce more reliable, calibrated predictions

The slight Dice drop in M4 reflects the model being more honest — less
overconfident on easy background regions — while delivering measurably
better sensitivity and specificity.

### Validation Metrics (Best Model — M4 Phase 2)
| Metric | Score | Interpretation |
|--------|-------|----------------|
| Dice Coefficient | 0.7484 | Good overlap overall |
| IoU | 0.5932 | Strict overlap measure |
| Sensitivity | 0.7350 | Detects 73.5% of nodules |
| Precision | 0.7951 | 79.5% of predictions correct |
| Specificity | 0.9992 | Near-zero false positives |

### Prediction Quality Distribution
| Category | Count | % |
|----------|-------|---|
| Perfect (Dice > 0.85) | 162 | 30.8% |
| Good (0.70 – 0.85) | 227 | 43.2% |
| Mediocre (0.40 – 0.70) | 74 | 14.1% |
| Failed (0.01 – 0.40) | 24 | 4.6% |
| Complete Miss (≤ 0.01) | 39 | 7.4% |

---

## 🏗️ Architecture

- **Model:** 3D U-Net (~1.4M parameters)
- **Input:** (N, 8, 128, 128, 1) — 8 consecutive CT slices per volume
- **Loss:** BCE + Dice (0.5 weighted combination)
- **Encoder:** 2 blocks (32, 64 filters) + bottleneck (128 filters)
- **Decoder:** 2 blocks with skip connections + sigmoid output
- **Fine-tuning:** Encoder frozen, decoder layers 28–37 progressively unfrozen

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
├── samples/                           # Ready-to-use demo .npy files
│   ├── demo_sample_0.npy  ..  9.npy   # Best predictions (Dice > 0.85)
│   └── random_sample_0.npy .. 4.npy   # Random validation samples
│
└── README.md
```

---

## 🗂️ Dataset

- **Name:** LIDC-IDRI (Lung Image Database Consortium)
- **Format:** PNG slices organized by patient → nodule → images/masks
- **Samples:** 2630 total (2104 train / 526 val)
- **Preprocessing:** Resize to 128×128, normalize to [0,1], pad/crop to 8 slices
- **Masks:** 4 annotator masks merged using max projection

---

## 🔄 Training Pipeline

```
M2 Baseline → M3 Augmented → M4 Phase 1 (freeze encoder)
                                       ↓
                              M4 Phase 2 (unfreeze layers 28–37)
                                       ↓
                              Best Val Dice: 0.7484
```

---

## ⚙️ Environment

- Google Colab (T4 GPU)
- TensorFlow 2.x
- Batch size: 2
- Optimizer: Adam with ReduceLROnPlateau + EarlyStopping

---

## ⚠️ Known Limitations

- **Boundary nodules** — pleural/chest wall nodules are hard to detect (max confidence ~0.32)
- **Sub-centimeter nodules** — nodules under 50 voxels may be missed
- **For research and educational purposes only — not for clinical use**

---

## 👤 Author

**TanishDevX**  
[Hugging Face](https://huggingface.co/TanishDevX) • [GitHub](https://github.com/TanishDevX)
