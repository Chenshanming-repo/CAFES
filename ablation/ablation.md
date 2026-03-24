# CAFES Ablation Study

## Overview

This ablation study systematically evaluates the contribution of three key architectural components in CAFES:

1. **Feature Channels** — the multi-channel input representation
2. **SEBlock** — Squeeze-and-Excitation channel attention
3. **Dropout** — regularisation within residual blocks

All experiments share the same training protocol (Adam, lr=1e-3, early stopping with patience=10, 4× LR decay) and differ only in the ablated component. The **baseline** is the full CAFES model (6 channels, SEBlock enabled, dropout=0.2).

---

## 1  Feature Channel Ablation

CAFES constructs a 6-channel input tensor from the raw nanopore current signal:

| Channel | Name | Description |
|---------|------|-------------|
| 0 | `raw` | Raw electrical current |
| 1 | `diff` | First-order difference |
| 2 | `w_mean` | Sliding-window mean (w=3) |
| 3 | `w_std` | Sliding-window std (w=3) |
| 4 | `t_stat` | Two-sample t-statistic for change-point detection |
| 5 | `z_score` | Z-score normalised current |

### Experiments

| ID | Name | Channels kept | # Ch | Hypothesis |
|----|------|---------------|------|------------|
| E0 | `baseline_full` | raw, diff, w_mean, w_std, t_stat, z_score | 6 | Full model (reference) |
| E1 | `feat_no_zscore` | raw, diff, w_mean, w_std, t_stat | 5 | Z-score normalisation may be redundant given raw + statistical features |
| E2 | `feat_no_tstat` | raw, diff, w_mean, w_std, z_score | 5 | T-statistic captures change points; removing it tests whether the CNN learns equivalent patterns |
| E3 | `feat_no_tstat_zscore` | raw, diff, w_mean, w_std | 4 | Minimal statistical augmentation without domain-specific channels |
| E4 | `feat_raw_diff` | raw, diff | 2 | Tests whether temporal derivative alone is sufficient |
| E5 | `feat_raw_only` | raw | 1 | Lower bound — raw signal with no engineered features |

### What to look for

- **Marginal contribution**: compare each 5-channel variant against the 6-channel baseline to isolate the value of `t_stat` vs `z_score`.
- **Diminishing returns**: the gap between 4ch → 6ch vs 1ch → 4ch reveals whether the first few features matter most.
- **Inference cost**: fewer channels reduce FLOPs in the first convolution layer; check `avg_infer_time`.

---

## 2  SEBlock Ablation

The Squeeze-and-Excitation block recalibrates channel responses via global average pooling → two FC layers → sigmoid gating:

```
AdaptiveAvgPool1d(1) → Conv1d(C → C/16) → ReLU → Conv1d(C/16 → C) → Sigmoid → scale
```

It is applied after the second BatchNorm in every `BasicConvResBlock`, before the residual addition.

### Experiments

| ID | Name | SEBlock | Hypothesis |
|----|------|---------|------------|
| E0 | `baseline_full` | Enabled | Full model (reference) |
| E6 | `se_disabled` | Disabled | Tests whether learned channel attention improves classification beyond what residual connections already provide |

### What to look for

- **Accuracy delta**: a significant drop when SEBlock is removed indicates the model benefits from adaptive channel weighting.
- **Parameter overhead**: SEBlock adds `2 × (C × C/16)` parameters per residual block. Compare total parameter counts to judge the efficiency trade-off.
- **Convergence speed**: SEBlock may help the model converge faster by focusing on informative channels early in training.

---

## 3  Dropout Ablation

Dropout is applied after the first ReLU inside each `BasicConvResBlock`:

```
Conv1d → BN → ReLU → Dropout(p) → Conv1d → BN → SEBlock → (+residual) → ReLU
```

### Experiments

| ID | Name | Dropout rate | Hypothesis |
|----|------|-------------|------------|
| E7 | `dropout_0.0` | 0.0 | No regularisation — tests whether the model overfits |
| E8 | `dropout_0.1` | 0.1 | Light regularisation |
| E0 | `baseline_full` | 0.2 | Default (reference) |
| E9 | `dropout_0.3` | 0.3 | Moderate regularisation |
| E10 | `dropout_0.5` | 0.5 | Heavy regularisation — may under-fit |

### What to look for

- **Overfitting signal**: compare train accuracy vs test accuracy. If `dropout_0.0` has high train acc but low test acc, dropout is essential.
- **Sweet spot**: the rate that maximises test F1 without sacrificing recall.
- **Interaction with early stopping**: higher dropout may delay convergence, causing early stopping to trigger prematurely.

---

## Experiment Summary Table

| ID | Experiment | Group | Channels | SEBlock | Dropout |
|----|-----------|-------|----------|---------|---------|
| E0 | `baseline_full` | baseline | 6 (all) | Yes | 0.2 |
| E1 | `feat_no_zscore` | feature_channels | 5 (w/o z_score) | Yes | 0.2 |
| E2 | `feat_no_tstat` | feature_channels | 5 (w/o t_stat) | Yes | 0.2 |
| E3 | `feat_no_tstat_zscore` | feature_channels | 4 (raw+diff+stats) | Yes | 0.2 |
| E4 | `feat_raw_diff` | feature_channels | 2 (raw+diff) | Yes | 0.2 |
| E5 | `feat_raw_only` | feature_channels | 1 (raw) | Yes | 0.2 |
| E6 | `se_disabled` | SEBlock | 6 (all) | No | 0.2 |
| E7 | `dropout_0.0` | dropout | 6 (all) | Yes | 0.0 |
| E8 | `dropout_0.1` | dropout | 6 (all) | Yes | 0.1 |
| E9 | `dropout_0.3` | dropout | 6 (all) | Yes | 0.3 |
| E10 | `dropout_0.5` | dropout | 6 (all) | Yes | 0.5 |

> Total: **11 experiments** (baseline is shared across all three groups).

---

## How to Run

```bash
# Run all 11 experiments
python ablation_study.py \
    -p <pos_data_folder> \
    -n <neg_data_folder> \
    -o ablation/results \
    -g 0

# Run a subset (e.g. feature-channel group only)
python ablation_study.py \
    -p <pos_data_folder> \
    -n <neg_data_folder> \
    -o ablation/results \
    -g 0 \
    -exp baseline_full feat_no_zscore feat_no_tstat feat_no_tstat_zscore feat_raw_diff feat_raw_only
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `-p` | required | Positive dataset folder (contains train/valid/test .npy) |
| `-n` | required | Negative dataset folder |
| `-o` | required | Output root directory |
| `-g` | None | GPU device IDs (e.g. `0` or `0,1`) |
| `-b` | 1024 | Batch size |
| `-e` | 300 | Max epochs |
| `-lr` | 1e-3 | Initial learning rate |
| `-t` | 10 | Early stopping patience |
| `-exp` | all | Space-separated list of experiment names to run |

---

## Output Structure

```
ablation/results/
├── ablation_results.csv          # Metrics for all completed experiments
├── ablation_summary.txt          # Formatted comparison table
├── ablation_feature_channels.pdf # Bar chart — feature channel group
├── ablation_SEBlock.pdf          # Bar chart — SEBlock group
├── ablation_dropout.pdf          # Bar chart — dropout group
├── baseline_full/
│   └── model.pth
├── feat_no_zscore/
│   └── model.pth
├── ...
```

---

## Evaluation Metrics

Each experiment reports:

| Metric | Definition |
|--------|-----------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) × 100 |
| Precision | TP / (TP + FP) × 100 |
| Recall | TP / (TP + FN) × 100 |
| F1 Score | 2 × Precision × Recall / (Precision + Recall) |
| Avg Inference Time | Mean batch inference time (seconds) |
| Training Time | Total wall-clock training time (seconds) |
