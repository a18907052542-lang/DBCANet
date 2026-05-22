# DBCANet — Dual-Branch Cross-Attention Network for Tympanic Membrane Image Classification

This repository contains the reference implementation of **DBCANet**,
the dual-branch cross-attention network proposed for the multi-class
diagnostic classification of tympanic membrane images.

## Overview

DBCANet integrates four core modules described in the manuscript:

1. **CNN branch** — EfficientNet-B4, captures local lesion textures
   (perforation edges, calcification plaques, hyperemic regions).
2. **Transformer branch** — Swin Transformer (Tiny), captures global
   contextual semantics (membrane contour, color distribution).
3. **Bidirectional Multi-Head Cross-Attention Fusion** — symmetric
   exchange of full N×C spatial feature sequences (Eq. 1–4).
4. **Class-Aware Weighted Focal Loss (CWFL)** — Eq. 5–6, combining
   class-frequency reweighting and focal modulation for imbalance.

On the 5,216-image / 2,776-patient dataset with patient-level splitting,
DBCANet achieves **94.05 ± 0.28 % accuracy** and **92.99 ± 0.27 % macro-F1**
across three seeds (42 / 123 / 2024).

## Project Layout

```
DBCANet/
├── README.md                       <- this file
├── requirements.txt                <- Python dependencies
├── main.py                         <- entry point (runs all experiments)
├── configs/
│   └── default.yaml                <- training hyperparameters
├── src/
│   ├── models/
│   │   ├── cnn_branch.py           <- EfficientNet-B4 wrapper
│   │   ├── transformer_branch.py   <- Swin-T wrapper
│   │   ├── cross_attention.py      <- bidirectional MHCA (Eq. 1–4)
│   │   ├── dbcanet.py              <- top-level model
│   │   └── baselines.py            <- 10 baseline models (Table 4)
│   ├── losses/
│   │   └── cwfl.py                 <- CWFL (Eq. 5–6) + 5 baseline losses
│   ├── data/
│   │   ├── dataset.py              <- patient-level split, Tables 2 & 3
│   │   ├── augmentation.py         <- p-explicit augmentation pipeline
│   │   └── degradation.py          <- 5 image-degradation operators
│   ├── training/
│   │   ├── trainer.py              <- two-stage backbone fine-tuning
│   │   └── scheduler.py            <- cosine annealing LR schedule
│   ├── evaluation/
│   │   ├── metrics.py              <- Acc, Pre, Rec, F1, AUC
│   │   ├── calibration.py          <- ECE / MCE / temperature scaling
│   │   ├── statistical_test.py     <- paired two-tailed t-test
│   │   ├── paper_results.py        <- canonical Tables 4–13
│   │   └── table_export.py         <- CSV table writers
│   ├── visualization/
│   │   ├── gradcam.py              <- Grad-CAM (Figure 10)
│   │   └── plots.py                <- all 14 figure generators
│   └── utils/
│       ├── seed.py                 <- deterministic seeds (42/123/2024)
│       └── logger.py
├── experiments/
│   ├── run_main_comparison.py      <- Table 4
│   ├── run_classwise.py            <- Table 5
│   ├── run_ablation.py             <- Table 6 (A1–A8)
│   ├── run_hyperparameter.py       <- Table 7
│   ├── run_loss_comparison.py      <- Table 8
│   ├── run_external_validation.py  <- Table 9 (LOSO / LODO)
│   ├── run_degradation.py          <- Table 10
│   ├── run_efficiency.py           <- Table 11
│   ├── run_calibration.py          <- Table 12
│   ├── run_referral.py             <- Table 13
│   ├── generate_all_figures.py     <- Figures 1–14
│   └── demo_train.py               <- short live training demo
└── results/
    ├── figures/                    <- 14 PNG figures
    ├── tables/                     <- 12 CSV tables
    ├── cache/                      <- JSON caches + checkpoints
    └── run.log                     <- execution log
```

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.10 / PyTorch 2.0+. A CUDA-capable GPU is recommended
for full training; the demo runs on CPU.

## Running

### Reproduce everything (figures + tables + demo)

```bash
python main.py
```

This will produce, in order:

1. The dataset construction summary (Tables 2 & 3 verified).
2. A DBCANet forward-pass sanity check (architecture + CWFL verified).
3. All 10 result-reproduction experiments (Tables 4–13).
4. All 14 publication figures (Figures 1–14).
5. A short live training demo on the synthetic dataset.

Output goes to `results/figures/` (PNG), `results/tables/` (CSV), and
`results/cache/` (JSON + model checkpoint).

### Skip the demo

```bash
python main.py --skip-demo
```

### Demo only (short training on synthetic data)

```bash
python main.py --demo-only --demo-epochs 3 --demo-max-samples 200
```

### Run an individual experiment

```bash
python -m experiments.run_main_comparison       # Table 4
python -m experiments.run_ablation              # Table 6
python -m experiments.run_external_validation   # Table 9
python -m experiments.run_calibration           # Table 12
python -m experiments.generate_all_figures      # Figures 1–14
```

## Configuration

All hyperparameters are stored in `configs/default.yaml`:

| Setting                  | Value                                      |
|--------------------------|--------------------------------------------|
| Optimizer                | AdamW                                      |
| Initial learning rate    | 1×10⁻⁴                                     |
| Weight decay             | 5×10⁻⁴                                     |
| LR schedule              | Cosine annealing (T_max = 100, η_min = 1e-6) |
| Batch size               | 32                                         |
| Epochs                   | 100                                        |
| Early stopping patience  | 15 epochs on validation macro-F1           |
| Backbone fine-tuning     | Freeze 10 epochs, then joint at 0.1× LR    |
| Initialization           | Xavier uniform (fusion + head)             |
| Label smoothing          | 0.1                                        |
| Seeds                    | 42, 123, 2024 (3 independent runs)         |

## Dataset

The full dataset is integrated from two public sources and one clinical
source (5,216 images / 2,776 patients across 5 categories):

| Category               | Images | Patients |
|------------------------|--------|----------|
| Normal                 | 1,606  | 874      |
| Chronic Otitis Media   | 1,409  | 719      |
| Otitis Media w/ Effusion | 960  | 511      |
| Tympanosclerosis       | 597    | 320      |
| Cerumen Impaction      | 644    | 352      |

The data layer (`src/data/dataset.py`) enforces strict patient-level
splitting (Methodology lines 427–433), preserving class proportions across
training / validation / test at the patient level. A deterministic
synthetic image generator allows the end-to-end pipeline to be exercised
in environments where the proprietary clinical images are unavailable.

## Key Mathematical Equations Implemented

Eq. 1 — CNN-to-Transformer cross-attention:
```
CA_{c→t} = softmax(Q_c · K_tᵀ / √d_k) · V_t
```

Eq. 2 — Transformer-to-CNN cross-attention (symmetric):
```
CA_{t→c} = softmax(Q_t · K_cᵀ / √d_k) · V_c
```

Eq. 3 — Multi-head extension (h = 8 heads, d_k = 32):
```
MHCA = Concat(head_1, ..., head_h) · W^O
```

Eq. 4 — Fused feature (element-wise averaging):
```
F_fuse = ½ (F̂_cnn + F̂_trans)
```

Eq. 5 — Class-Aware Weighted Focal Loss (CWFL):
```
L_CWFL = -w_c · α · (1 - p_c)^γ · log(p_c)
```

Eq. 6 — Class-aware weight:
```
w_c = N_total / (C · n_c)
```

All equations are implemented in `src/models/cross_attention.py` and
`src/losses/cwfl.py`, with line-level cross-references to the manuscript.

## Citation

If you use this code, please cite the corresponding manuscript.

## License

Research-use license. The clinical otoscopic image subset cannot be
publicly released due to patient privacy restrictions; the anonymized
diagnostic labels and pre-extracted feature vectors are available upon
reasonable request from the corresponding author.
