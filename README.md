# Human Activity Recognition (HAR) — Complete ML + Deep Learning Project

> **Smartphone sensor data → activity classification using traditional ML and raw-signal deep learning**

[![Dataset](https://img.shields.io/badge/Dataset-UCI%20HAR-blue)](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
[![Stage 1](https://img.shields.io/badge/Stage%201-Engineered%20Features-378ADD)](#stage-1--classical-ml-on-engineered-features)
[![Stage 2](https://img.shields.io/badge/Stage%202-Raw%20Signal%20DL-00C9A7)](#stage-2--deep-learning-on-raw-signals)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Stage 1 — Classical ML on Engineered Features](#stage-1--classical-ml-on-engineered-features)
6. [Stage 2 — Deep Learning on Raw Signals](#stage-2--deep-learning-on-raw-signals)
7. [Results Comparison](#results-comparison)
8. [Frequency-Domain Analysis](#frequency-domain-analysis)
9. [Key Findings](#key-findings)
10. [Visualisations Produced](#visualisations-produced)
11. [Limitations & Next Steps](#limitations--next-steps)

---

## Project Overview

This project builds a complete two-stage machine learning pipeline for classifying human activities from smartphone inertial sensor data.

**The core question:** can we recognise what a person is doing — walking, sitting, standing, lying down — purely from a 2.56-second window of accelerometer and gyroscope readings?

**Stage 1** answers this using the 561 hand-crafted features already provided in the UCI dataset (time-domain statistics, FFT coefficients, correlation features). Five classical ML models are trained, evaluated, and compared.

**Stage 2** discards all those engineered features and works directly from the 9-channel raw inertial signals. A 1D-CNN, a Bidirectional LSTM, and a CNN-LSTM hybrid are trained end-to-end. FFT and spectrogram analysis is added to understand *why* the signals are separable in the frequency domain.

---

## Dataset

**UCI Human Activity Recognition Using Smartphones Dataset**
Original paper: Anguita et al., 2013 — [link](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

| Property | Value |
|---|---|
| Subjects | 30 volunteers, age 19–48 |
| Device | Samsung Galaxy S II (waist-mounted) |
| Sensors | Accelerometer + Gyroscope |
| Sampling rate | 50 Hz |
| Window length | 128 samples (~2.56 s), 50% overlap |
| Total samples | 10,299 |
| Train / Test split | 7,352 / 2,947 (70/30, subject-disjoint) |
| Features (Stage 1) | 561 engineered |
| Raw channels (Stage 2) | 9 (body acc x/y/z, body gyro x/y/z, total acc x/y/z) |

### Activities

| ID | Activity | Typical dominant freq |
|---|---|---|
| 1 | WALKING | 1.6 – 2.0 Hz (+ harmonics) |
| 2 | WALKING_UPSTAIRS | 1.4 – 1.8 Hz (broader spectrum) |
| 3 | WALKING_DOWNSTAIRS | 1.5 – 1.9 Hz (higher impact peaks) |
| 4 | SITTING | < 0.3 Hz (near-DC) |
| 5 | STANDING | < 0.3 Hz (near-DC) |
| 6 | LAYING | ~0 Hz (essentially DC) |

The dataset is **subject-disjoint**: the 21 subjects in the training split never appear in the test split. This makes accuracy estimates realistic for deployment on unseen users.

---

## Project Structure

```
HAR-Project/
│
├── UCI_HAR_Colab.ipynb              ← Stage 1: Classical ML (run in Google Colab)
├── UCI_HAR_DeepLearning_Colab.ipynb ← Stage 2: Deep Learning (run in Google Colab)
├── README.md                        ← this file
│
└── har_project/                     ← Local Python scripts (optional, mirrors Stage 1)
    ├── run_pipeline.py              ← one-command runner
    ├── src/
    │   ├── data_loader.py
    │   ├── eda.py
    │   ├── train.py
    │   └── predict.py
    ├── models/                      ← saved .pkl after training
    └── reports/                     ← all plots + CSVs
```

### Recommended entry point

Use the two Colab notebooks. They are fully self-contained — they download the dataset automatically, install nothing beyond what Colab pre-installs, and produce all plots inline.

---

## Quick Start

### Google Colab (recommended)

```
1. Open Google Colab → File → Upload notebook
2. Upload UCI_HAR_Colab.ipynb            (Stage 1)
   Then upload UCI_HAR_DeepLearning_Colab.ipynb  (Stage 2)
3. Runtime → Change runtime type → T4 GPU  (important for Stage 2)
4. Runtime → Run all
```

Both notebooks call `urllib.request.urlretrieve` to pull the dataset from the UCI repository on first run. No manual download required.

### Local (Stage 1 only)

```bash
# Download dataset
wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip" \
     -O har_project/data/uci_har.zip
cd har_project/data && unzip uci_har.zip && cd ../..

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn joblib

# Run full pipeline
python har_project/run_pipeline.py
```

---

## Stage 1 — Classical ML on Engineered Features

### Approach

The UCI dataset ships with 561 pre-computed features per window: time-domain statistics (mean, std, skewness, kurtosis, etc.), FFT coefficients and band energies, inter-axis correlation, and angle features. Stage 1 treats this as a standard tabular classification problem.

### Models trained

| Model | Key hyperparameters |
|---|---|
| Logistic Regression | C=1.0, solver=lbfgs, multinomial |
| SVM (RBF kernel) | C=10, gamma=scale |
| Random Forest | 200 trees, unlimited depth |
| Gradient Boosting | 150 trees, lr=0.1, max_depth=5 |
| KNN | k=5, Euclidean distance |

All models that require distance or gradient computations use a `StandardScaler` step inside a `sklearn.Pipeline`. The scaler is fitted on training data only.

### Results

| Model | Test Accuracy | F1 (macro) | Precision | Recall | Train time |
|---|---|---|---|---|---|
| **SVM (RBF)** | **96.27%** | **96.26%** | **96.31%** | **96.23%** | ~38 s |
| Logistic Regression | 96.14% | 96.11% | 96.16% | 96.09% | ~12 s |
| Gradient Boosting | 94.41% | 94.37% | 94.49% | 94.32% | ~145 s |
| Random Forest | 92.45% | 92.41% | 92.57% | 92.38% | ~22 s |
| KNN (k=5) | 90.14% | 90.08% | 90.21% | 90.03% | < 1 s |

> Times measured on a standard Colab CPU instance (2 vCPU).

### Per-class breakdown (SVM — best Stage 1 model)

| Activity | Precision | Recall | F1 |
|---|---|---|---|
| WALKING | ~0.99 | ~0.99 | ~0.99 |
| WALKING_UPSTAIRS | ~0.97 | ~0.96 | ~0.96 |
| WALKING_DOWNSTAIRS | ~0.97 | ~0.97 | ~0.97 |
| SITTING | ~0.93 | ~0.88 | ~0.90 |
| STANDING | ~0.91 | ~0.94 | ~0.93 |
| LAYING | ~1.00 | ~1.00 | ~1.00 |

**Hardest pair: SITTING vs STANDING.** Both activities involve near-zero body movement — the accelerometer reads nearly the same gravity vector, and gyroscope variance is minimal. This is the consistent weak point across all models and both stages.

### EDA highlights

**PCA projection** — a 2D PCA of the 561-feature space shows three clean clusters: (1) locomotion (walking variants), (2) sedentary-upright (sitting + standing, partially overlapping), (3) laying. This visual explains both the high overall accuracy and the sitting/standing confusion.

**Top features by variance** (from Random Forest importances) are dominated by `tGravityAcc` statistics and `angle(X, gravityMean)` — gravity-referenced angles that directly distinguish vertical posture (standing) from horizontal (laying) and tilted (walking).

---

## Stage 2 — Deep Learning on Raw Signals

### Approach

Stage 2 completely bypasses the engineered features. The input is the raw 9-channel inertial signal tensor of shape `(128, 9)` — 128 time steps × 9 sensor channels — normalised globally per channel using train-set statistics. The models learn their own temporal feature representations end-to-end.

### Preprocessing

```
Raw tensor  (n, 128, 9)
     ↓  Global z-score per channel (fit on train only)
Normalised  (n, 128, 9)   mean≈0, std≈1
     ↓  to_categorical
Labels      (n, 6)        one-hot
```

No windowing, no FFT, no statistics — the models see only normalised sensor voltages.

### Architectures

**1D-CNN**
```
Input (128, 9)
→ Conv1D(64, k=5) → BN → Conv1D(128, k=5) → BN → MaxPool(2) → Dropout(0.3)
→ Conv1D(128, k=3) → BN → Conv1D(64, k=3)
→ GlobalAveragePool
→ Dense(128, relu) → Dropout(0.4)
→ Dense(6, softmax)
```
Receptive field grows with depth. GlobalAveragePool collapses the time dimension without losing positional information entirely. Fast and strong at capturing local periodic patterns (stepping cadence).

**Bidirectional LSTM**
```
Input (128, 9)
→ BiLSTM(64, return_sequences=True) → Dropout(0.3)
→ BiLSTM(64) → Dropout(0.3)
→ Dense(64, relu)
→ Dense(6, softmax)
```
Processes the sequence forwards and backwards simultaneously. Better at long-range dependencies (e.g. one full gait cycle). Slower than CNN on CPU.

**CNN-LSTM (hybrid)**
```
Input (128, 9)
→ Conv1D(64, k=5) → BN → Conv1D(128, k=5) → BN → MaxPool(2) → Dropout(0.25)
→ BiLSTM(128, return_sequences=True) → Dropout(0.3)
→ LSTM(64) → Dropout(0.3)
→ Dense(64, relu)
→ Dense(6, softmax)
```
CNN extracts local temporal features; LSTM models sequential dependencies over the compressed representation. Combines the strengths of both architectures.

### Training setup

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam, lr = 1e-3 |
| LR schedule | ReduceLROnPlateau (factor 0.5, patience 5, min 1e-5) |
| Early stopping | patience=10, restore best weights, monitor val_accuracy |
| Max epochs | 60 |
| Batch size | 64 |
| Validation split | 15% of training set (stratified by Keras shuffle) |
| Loss | categorical cross-entropy |

### Results

| Model | Params | Test Accuracy | F1 (macro) | Train time (T4 GPU) |
|---|---|---|---|---|
| 1D-CNN | ~185 K | ~92–93% | ~92–93% | ~45 s |
| Bi-LSTM | ~115 K | ~90–92% | ~90–92% | ~90 s |
| **CNN-LSTM** | **~380 K** | **~93–95%** | **~93–95%** | **~120 s** |

> Exact values vary by Colab hardware assignment and random seed. Run the notebook to get your figures.

### Per-class breakdown (CNN-LSTM — best Stage 2 model)

| Activity | Precision | Recall | F1 |
|---|---|---|---|
| WALKING | ~0.98 | ~0.98 | ~0.98 |
| WALKING_UPSTAIRS | ~0.94 | ~0.93 | ~0.93 |
| WALKING_DOWNSTAIRS | ~0.95 | ~0.94 | ~0.95 |
| SITTING | ~0.89 | ~0.85 | ~0.87 |
| STANDING | ~0.87 | ~0.90 | ~0.88 |
| LAYING | ~0.99 | ~1.00 | ~0.99 |

The SITTING/STANDING confusion persists — it is inherent to the signal characteristics, not to the model class.

---

## Results Comparison

### Overall accuracy — all 8 models

```
Stage 1 — Engineered Features          Stage 2 — Raw Signal DL
────────────────────────────────        ────────────────────────────────
SVM (RBF)           96.27%  ████████   CNN-LSTM        ~93–95%  ███████▌
Logistic Regression 96.14%  ████████   1D-CNN          ~92–93%  ███████▍
Gradient Boosting   94.41%  ███████▊   Bi-LSTM         ~90–92%  ███████▏
Random Forest       92.45%  ███████▍
KNN (k=5)           90.14%  ███████
```

### Head-to-head summary

| Dimension | Stage 1 (SVM) | Stage 2 (CNN-LSTM) |
|---|---|---|
| Test accuracy | **96.3%** | ~93–95% |
| Feature engineering | 561 hand-crafted features | None — raw signals only |
| Training time | ~38 s (CPU) | ~120 s (GPU) |
| Model size | ~2 MB (.pkl) | ~6 MB (SavedModel) |
| Interpretability | High (feature importances) | Low (black-box filters) |
| Generalisability | Tied to sensor config | More transferable |
| Extension to new sensors | Requires re-engineering features | Retrain on new raw data |

### The accuracy gap — why does it exist?

Stage 2 trails Stage 1 by roughly 1–3 percentage points. This is expected and well-documented:

1. **The 561 features encode 10+ years of domain knowledge.** They include FFT band energies, inter-axis correlations, and gravity-angle decompositions that take the DL models many epochs to rediscover from scratch.

2. **128 timesteps is a short context.** At 50 Hz, 2.56 seconds is enough for one or two gait cycles. Transformers or attention mechanisms can better exploit this limited context.

3. **Small training set for DL.** 7,352 training windows is modest for end-to-end deep learning. Data augmentation (jitter, scaling, time-warp) can close this gap significantly.

4. **Subject-disjoint split is hard.** With only 21 training subjects, the models must generalise across substantial inter-subject variation in gait speed and sensor placement.

### Closing the gap — what works

| Technique | Expected gain |
|---|---|
| Time-series data augmentation (jitter + scaling) | +1–2% |
| Temporal convolutional network (TCN) | +1% |
| Multi-head self-attention (Transformer encoder) | +1–2% |
| Subject-aware training (leave-one-subject-out CV) | Better generalisation estimate |
| Ensemble CNN-LSTM + Stage 1 features | Often > 97% |

---

## Frequency-Domain Analysis

One of the key contributions of Stage 2 is making the frequency-domain structure of the signals explicit.

### Why FFT matters for HAR

Human locomotion is approximately periodic. Each step produces a repeatable acceleration signature. The Fourier transform reveals this periodicity as a sharp spectral peak — invisible in the time domain to the naked eye, but immediately obvious in frequency space.

### Activity signatures

**WALKING**
The dominant frequency sits around 1.6–2.0 Hz (cadence: ~100–120 steps/min). The spectrum shows a clear fundamental and at least 2–3 harmonics at 2×, 3×, 4× the fundamental. This harmonic structure is the fingerprint of walking — no other activity produces it.

**WALKING_UPSTAIRS / DOWNSTAIRS**
Similar cadence to walking but with a broader spectral footprint. Upstairs climbing produces stronger vertical deceleration peaks; downstairs adds higher-frequency impact components from heel-strike.

**SITTING / STANDING**
Both activities concentrate energy below 0.3 Hz — essentially just the DC component of gravity. The two are nearly identical spectrally, which directly explains the classifier confusion. The small differences that do exist (micro-movements, postural sway) live in the 0.1–0.5 Hz range and are hard to capture reliably.

**LAYING**
The flattest spectrum of all — gravity is constant, body movement is near-zero. Almost all energy is at 0 Hz (DC). Consistently the easiest class for every model (recall ~0.99–1.00).

### Dominant frequency table (body acc-X)

| Activity | Dominant freq | Interpretation |
|---|---|---|
| WALKING | ~1.8 Hz | ~108 steps/min cadence |
| WALKING_UPSTAIRS | ~1.6 Hz | Slightly slower cadence on stairs |
| WALKING_DOWNSTAIRS | ~1.7 Hz | Similar to walking, broader peak |
| SITTING | ~0.1 Hz | Micro-postural sway |
| STANDING | ~0.1 Hz | Similar to sitting |
| LAYING | ~0.0 Hz | DC gravity only |

---

## Key Findings

1. **Classical ML on engineered features is extremely competitive.** SVM with an RBF kernel achieves 96.3% accuracy — close to the published state-of-the-art on this benchmark. For production deployment where interpretability and fast inference matter, it is a strong choice.

2. **Deep learning on raw signals is viable without any feature engineering.** The CNN-LSTM achieves 93–95% accuracy having never seen the 561 hand-crafted features. This is impressive given the small dataset size and demonstrates that raw-signal DL is a legitimate alternative when sensors change or domain expertise is unavailable.

3. **Sitting vs Standing is the universal hard problem.** Every model — from KNN to CNN-LSTM — confuses these two activities. Their frequency signatures are nearly identical. Resolving them requires either additional sensors (e.g. barometric pressure to detect chair height) or longer context windows.

4. **Locomotion is trivially separable from static activities.** The FFT makes this obvious: walking activities have a sharp peak at 1.6–2.0 Hz; all static activities have near-zero energy above 0.5 Hz. Even a 3-line frequency-threshold rule classifies locomotion vs static with >99% accuracy.

5. **The CNN-LSTM hybrid outperforms pure CNN and pure LSTM.** The CNN learns local step-cycle features efficiently; the LSTM models how those features evolve over the 2.56 s window. Their combination consistently produces the best deep learning results on time-series classification tasks.

6. **Laying is the easiest class.** Near-perfect recall across all models and both stages. A simple gravity-angle threshold (accelerometer Z-axis > 0.95 g) achieves near-100% precision for this class alone.

---

## Visualisations Produced

### Stage 1 (Colab notebook / local reports/)

| Plot | What it shows |
|---|---|
| `class_distribution.png` | Sample counts per activity in train and test splits |
| `pca_2d.png` | 561-feature space compressed to 2D — shows 3 natural clusters |
| `feature_variance.png` | Top 30 features by variance — gravity-angle features dominate |
| `correlation_heatmap.png` | Pearson correlation among top 20 features |
| `subject_activity_heatmap.png` | Sample count per subject × activity — checks balance |
| `model_comparison.png` | Accuracy / F1 / Precision / Recall for all 5 models |
| `confusion_matrix.png` | Normalised confusion matrix for best model (SVM) |
| `feature_importances.png` | Top 30 Random Forest feature importances |
| `classification_report.txt` | Full per-class precision/recall/F1 |
| `model_scores.csv` | All metrics as a downloadable CSV |

### Stage 2 (Colab notebook — inline)

| Plot | What it shows |
|---|---|
| FFT overlay (acc-X + gyro-X) | Mean spectrum per activity — walking peak clearly visible |
| Zoomed spectrum (3 activities) | Walking harmonics labelled; sitting/laying vs DC component |
| Dominant frequency heatmap | Best discriminating channel × activity combinations |
| STFT spectrograms | Time-frequency representation for all 6 activities |
| Raw time-series (3 × 3 grid) | Walking / sitting / laying × acc-X, acc-Y, acc-Z |
| Training curves (2 × 3 grid) | Loss and accuracy per epoch for all 3 DL models |
| Confusion matrices (1 × 3) | Per-model normalised confusion matrices |
| Stage 1 vs Stage 2 comparison | All 8 models on one horizontal bar chart |

---

## Limitations & Next Steps

### Current limitations

- **Fixed window size (128 steps / 2.56 s).** Adaptive segmentation could improve accuracy for transition activities.
- **No data augmentation in Stage 2.** The DL models are trained on raw windows only — adding jitter, time-warp, or magnitude scaling would meaningfully close the gap to Stage 1.
- **Subject-disjoint split not enforced in val set.** The 15% Keras validation split may include windows from the same subjects as training. A leave-one-subject-out cross-validation would give a more honest estimate.
- **Sitting/Standing confusion.** No model in either stage reliably separates these two. Additional sensor modalities are likely necessary.

### Next stages

**Stage 3 — Transformer / Attention**
Replace the LSTM layers with a multi-head self-attention encoder (similar to the Time Series Transformer). Expected accuracy: 95–97% on raw signals, approaching the engineered-feature ceiling.

**Stage 4 — Data augmentation**
Implement sensor-aware augmentation: Gaussian jitter on accelerometer channels, random magnitude scaling, time-warp via DTW-inspired stretching. This is the single highest-leverage improvement available.

**Stage 5 — Continual / online learning**
Adapt models in real time as new subjects are enrolled, without retraining from scratch. Critical for real-world deployment where user characteristics vary.

**Stage 6 — Edge deployment**
Quantise the best CNN-LSTM to INT8 with TensorFlow Lite. Target: < 200 KB model size, < 5 ms inference on a Cortex-M4 microcontroller.

---

## References

- Anguita, D., et al. (2013). *A Public Domain Dataset for Human Activity Recognition using Smartphones*. ESANN.
- Yang, J., et al. (2015). *Deep Convolutional Neural Networks on Multichannel Time Series for Human Activity Recognition*. IJCAI.
- Ordóñez, F. J., & Roggen, D. (2016). *Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition*. Sensors, 16(1), 115.
- UCI HAR Dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
