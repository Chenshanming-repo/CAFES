# Feature Window Size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a user-configurable feature embedding window size for training/testing and a window-size sweep in ablation studies.

**Architecture:** Reuse `preprocessor.add_features(..., w_len=...)` as the single feature embedding implementation. Add validation and length-preserving rolling-window padding there, then pass the selected value through dataset, training, testing, and ablation call paths. Ablation experiments carry `feature_window_size` as experiment metadata and output it to logs, summaries, and CSV.

**Tech Stack:** Python, NumPy, PyTorch, argparse, pytest.

---

## File Structure

- Create `tests/test_feature_window_size.py`: fast unit tests for feature shape, window-size validation, dataset pass-through, and ablation experiment generation.
- Modify `preprocessor.py`: add window-size validation helpers, preserve rolling feature length for arbitrary windows, and validate `add_features`.
- Modify `dataset.py`: add `feature_window_size` to `LazyTrainDataset` and pass it to `add_features`.
- Modify `trainer.py`: add `--feature_window_size/-fws` and pass it to validation features, lazy train datasets, and test calls.
- Modify `tester.py`: add `--feature_window_size/-fws`; thread the value through `test()` into `add_features`.
- Modify `ablation_study.py`: add `--window_sizes`; generate `feature_window_size` experiments; pass each experiment's window size through train, validation, and test feature generation; include the field in summaries and CSV.
- Modify `README.md`: document the new CLI parameters in train, test, and ablation usage.

---

### Task 1: Add Failing Feature-Window Tests

**Files:**
- Create: `tests/test_feature_window_size.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_feature_window_size.py` with this content:

```python
import argparse

import numpy as np
import pytest
import torch

import dataset as dataset_module
from ablation_study import get_experiments
from dataset import LazyTrainDataset
from preprocessor import add_features, feature_window_size_type


def test_add_features_preserves_length_with_non_default_window():
    signals = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)

    features = add_features(signals, norm=False, s_len=6, w_len=5)

    assert features.shape == (1, 6, 6)
    assert torch.allclose(
        features[0, 2],
        torch.tensor([0.0, 0.0, 3.0, 4.0, 0.0, 0.0]),
    )


def test_add_features_changes_rolling_channels_when_window_changes():
    signals = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)

    features_3 = add_features(signals, norm=False, s_len=6, w_len=3)
    features_5 = add_features(signals, norm=False, s_len=6, w_len=5)

    assert not torch.allclose(features_3[:, 2:4, :], features_5[:, 2:4, :])


def test_feature_window_size_type_rejects_values_below_two():
    with pytest.raises(argparse.ArgumentTypeError, match="at least 2"):
        feature_window_size_type("1")


def test_lazy_train_dataset_passes_feature_window_size(tmp_path, monkeypatch):
    npy_path = tmp_path / "train.npy"
    np.save(npy_path, np.array([np.arange(8, dtype=np.float32)], dtype=object))
    captured = {}

    def fake_add_features(signals, norm=True, s_len=3000, w_len=3):
        captured["signals"] = signals
        captured["norm"] = norm
        captured["s_len"] = s_len
        captured["w_len"] = w_len
        return torch.zeros((1, 6, len(signals[0])), dtype=torch.float32)

    monkeypatch.setattr(dataset_module, "add_features", fake_add_features)

    ds = LazyTrainDataset(
        str(npy_path),
        data_type="pos",
        norm=True,
        cut=0,
        length=6,
        tile=1,
        feature_window_size=5,
    )

    x, y = ds[0]

    assert x.shape == (6, 6)
    assert y == 1
    assert captured["norm"] is True
    assert captured["s_len"] == 6
    assert captured["w_len"] == 5


def test_get_experiments_adds_window_size_sweep_without_duplicating_baseline():
    experiments = get_experiments(window_sizes=[3, 5, 7])
    by_name = {exp["name"]: exp for exp in experiments}

    assert by_name["baseline_full"]["feature_window_size"] == 3
    assert "window_size_3" not in by_name
    assert by_name["window_size_5"]["group"] == "feature_window_size"
    assert by_name["window_size_5"]["feature_window_size"] == 5
    assert by_name["window_size_5"]["channels"] == [0, 1, 2, 3, 4, 5]
    assert by_name["window_size_7"]["feature_window_size"] == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_feature_window_size.py -q
```

Expected: FAIL before implementation. The first expected failure is an import error for `feature_window_size_type` or a signature error for `get_experiments(window_sizes=...)`.

---

### Task 2: Implement Feature Window Validation And Rolling Padding

**Files:**
- Modify: `preprocessor.py`
- Test: `tests/test_feature_window_size.py`

- [ ] **Step 1: Add validation helpers near the imports in `preprocessor.py`**

Add this code after the imports:

```python
def validate_feature_window_size(w_len):
    if int(w_len) != w_len:
        raise ValueError("feature window size must be an integer")
    w_len = int(w_len)
    if w_len < 2:
        raise ValueError("feature window size must be at least 2")
    return w_len


def feature_window_size_type(value):
    try:
        w_len = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "feature window size must be an integer"
        ) from exc
    if w_len < 2:
        raise argparse.ArgumentTypeError(
            "feature window size must be at least 2"
        )
    return w_len
```

- [ ] **Step 2: Update `window_mean_std` to preserve the original signal length**

Replace the existing `window_mean_std` body with:

```python
def window_mean_std(sig, w_len):
    w_len = validate_feature_window_size(w_len)
    sig = sliding_window_view(sig, window_shape=w_len, axis=1)
    w_means = np.mean(sig, axis=2)
    w_stds = np.std(sig, axis=2)

    pad_total = w_len - 1
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    w_means = np.pad(w_means, ((0, 0), (pad_left, pad_right)))
    w_stds = np.pad(w_stds, ((0, 0), (pad_left, pad_right)))

    return torch.tensor(w_means), torch.tensor(w_stds)
```

- [ ] **Step 3: Validate `w_len` at the start of `add_features`**

At the start of `add_features`, before `signal_arr = ...`, add:

```python
    w_len = validate_feature_window_size(w_len)
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_feature_window_size.py::test_add_features_preserves_length_with_non_default_window tests/test_feature_window_size.py::test_add_features_changes_rolling_channels_when_window_changes tests/test_feature_window_size.py::test_feature_window_size_type_rejects_values_below_two -q
```

Expected: PASS for these three tests. Dataset and ablation tests still fail until later tasks.

---

### Task 3: Thread Feature Window Size Through Dataset, Trainer, And Tester

**Files:**
- Modify: `dataset.py`
- Modify: `trainer.py`
- Modify: `tester.py`
- Test: `tests/test_feature_window_size.py`

- [ ] **Step 1: Update `LazyTrainDataset` constructor and feature call**

In `dataset.py`, change the constructor signature to:

```python
    def __init__(self, npy_path, data_type='pos', norm=False,
                 cut=1500, length=3000, tile=3, feature_window_size=3):
```

Store the parameter after `self.param_tile = tile`:

```python
        self.feature_window_size = feature_window_size
```

Change the feature call in `__getitem__` to:

```python
        X = add_features(
            np.array([segment]),
            norm=self.norm,
            s_len=self.param_length,
            w_len=self.feature_window_size,
        )[0].float()
```

- [ ] **Step 2: Add trainer CLI parameter**

In `trainer.py`, import `feature_window_size_type`:

```python
from preprocessor import train_normalization, valid_normalization, add_features, train_non_normalization, \
    valid_non_normalization, feature_window_size_type
```

Add the parser argument after `--length`:

```python
    parser.add_argument("--feature_window_size", "-fws", type=feature_window_size_type, default=3,
                        help="Window size for feature embedding rolling mean/std and t-stat, default 3")
```

- [ ] **Step 3: Pass trainer window size to validation and lazy training**

Change positive validation feature generation to:

```python
    pos_valid_data = add_features(
        valid_process_method(pos_valid_data, args.cut, args.length,
                             args.patches, args.seq_length, args.stride, args.patch_size),
        norm=args.norm,
        s_len=args.length,
        w_len=args.feature_window_size,
    )
```

Change negative validation feature generation to:

```python
    neg_valid_data = add_features(
        valid_process_method(neg_valid_data, args.cut, args.length,
                             args.patches, args.seq_length, args.stride, args.patch_size),
        norm=args.norm,
        s_len=args.length,
        w_len=args.feature_window_size,
    )
```

Change training dataset creation to:

```python
    pos_train_set = LazyTrainDataset(
        os.path.join(args.pos_data_folder, 'train.npy'),
        data_type='pos',
        norm=args.norm,
        cut=args.cut,
        length=args.length,
        tile=args.tiling_fold,
        feature_window_size=args.feature_window_size,
    )
    neg_train_set = LazyTrainDataset(
        os.path.join(args.neg_data_folder, 'train.npy'),
        data_type='neg',
        norm=args.norm,
        cut=args.cut,
        length=args.length,
        tile=args.tiling_fold,
        feature_window_size=args.feature_window_size,
    )
```

Change both test calls to pass the window size:

```python
    tp, fn, pos_infer_time = test_method(model, pos_test_data, 1, args.batch_size, args.cut, args.length,
              args.patches, args.seq_length, args.stride, args.patch_size, log, device, args.norm,
              args.feature_window_size)
    tn, fp, neg_infer_time = test_method(model, neg_test_data, 0, args.batch_size, args.cut, args.length,
              args.patches, args.seq_length, args.stride, args.patch_size, log, device, args.norm,
              args.feature_window_size)
```

- [ ] **Step 4: Update tester CLI and `test()` signature**

In `tester.py`, import `feature_window_size_type`:

```python
from preprocessor import add_features, modified_zscore, feature_window_size_type
```

Change the `test` signature to:

```python
def test(model, reads, label, batch_size, cut, length,
              patches, seq_length, stride, patch_size, log, device, norm,
              feature_window_size=3):
```

Change both `add_features` calls inside `test()` to:

```python
                inputs = add_features(
                    inputs,
                    norm=norm,
                    s_len=length,
                    w_len=feature_window_size,
                )
```

Add the parser argument after `--length`:

```python
    parser.add_argument("--feature_window_size", "-fws", type=feature_window_size_type, default=3,
                        help="Window size for feature embedding rolling mean/std and t-stat, default 3")
```

Change both CLI test calls to append `args.feature_window_size` after `args.norm`.

- [ ] **Step 5: Run dataset and help checks**

Run:

```bash
pytest tests/test_feature_window_size.py::test_lazy_train_dataset_passes_feature_window_size -q
```

Expected: PASS.

Run:

```bash
python trainer.py --help
```

Expected: output contains `--feature_window_size`.

Run:

```bash
python tester.py --help
```

Expected: output contains `--feature_window_size`.

---

### Task 4: Add Ablation Window-Size Sweep

**Files:**
- Modify: `ablation_study.py`
- Test: `tests/test_feature_window_size.py`

- [ ] **Step 1: Import CLI validation**

Change the preprocessor import to:

```python
from preprocessor import add_features, valid_non_normalization, feature_window_size_type
```

- [ ] **Step 2: Add helpers after `CHANNEL_NAMES`**

Add:

```python
DEFAULT_FEATURE_WINDOW_SIZE = 3


def unique_window_sizes(window_sizes):
    if window_sizes is None:
        return [DEFAULT_FEATURE_WINDOW_SIZE]
    unique = []
    for w_len in window_sizes:
        if w_len not in unique:
            unique.append(w_len)
    return unique
```

- [ ] **Step 3: Change experiment generation**

Change `def get_experiments():` to:

```python
def get_experiments(window_sizes=None):
```

After building the existing experiment list and before `return exps`, add:

```python
    for exp in exps:
        exp.setdefault("feature_window_size", DEFAULT_FEATURE_WINDOW_SIZE)

    for w_len in unique_window_sizes(window_sizes):
        if w_len == DEFAULT_FEATURE_WINDOW_SIZE:
            continue
        exps.append(dict(
            name=f"window_size_{w_len}", group="feature_window_size",
            channels=[0, 1, 2, 3, 4, 5],
            channel_desc=f"All 6ch, window={w_len}",
            use_SEBlock=True, dropout=0.2, norm=True,
            use_first_bn=True,
            feature_window_size=w_len,
        ))
```

- [ ] **Step 4: Pass each experiment window size through train/validation/test**

In `run_experiment`, set:

```python
    feature_window_size = exp.get("feature_window_size", DEFAULT_FEATURE_WINDOW_SIZE)
```

Pass it to both lazy train datasets:

```python
        LazyTrainDataset(os.path.join(args.train_pos_data_folder, "train.npy"),
                         data_type="pos", norm=exp["norm"],
                         cut=args.cut, length=args.length,
                         feature_window_size=feature_window_size), ch_idx)
```

```python
        LazyTrainDataset(os.path.join(args.train_neg_data_folder, "train.npy"),
                         data_type="neg", norm=exp["norm"],
                         cut=args.cut, length=args.length,
                         feature_window_size=feature_window_size), ch_idx)
```

Pass it to validation `add_features` calls:

```python
        norm=exp["norm"],
        s_len=args.length,
        w_len=feature_window_size), ch_idx)
```

Pass it to `test_model`:

```python
    metrics = test_model(model, pos_test, neg_test, args.batch_size,
                         args.cut, args.length, ch_idx, device, exp["norm"],
                         feature_window_size=feature_window_size,
                         log_fh=exp_log)
```

Change `test_model` signature to:

```python
def test_model(model, pos_reads, neg_reads, batch_size,
               cut, length, channel_indices, device, norm,
               feature_window_size=DEFAULT_FEATURE_WINDOW_SIZE, log_fh=None):
```

Change both `add_features(buf, norm=norm)` calls inside `test_model` to:

```python
                        add_features(buf, norm=norm, s_len=length,
                                     w_len=feature_window_size), channel_indices)
```

- [ ] **Step 5: Add CLI argument and metadata output**

Add parser argument after `--length`:

```python
    ap.add_argument("--window_sizes", "-ws", type=feature_window_size_type,
                    nargs="+", default=[DEFAULT_FEATURE_WINDOW_SIZE],
                    help="Feature embedding window sizes to evaluate, default 3")
```

Change:

```python
    experiments = get_experiments()
```

to:

```python
    experiments = get_experiments(args.window_sizes)
```

In the experiment logging block, add:

```python
        print(f"  FeatWin  : {exp['feature_window_size']}")
```

In `print_summary`, add a `FWin` column after `Ch`:

```python
    hdr = (f"{'Experiment':<25} {'Ch':<5} {'FWin':<6} {'SE':<6} {'1stBN':<6} {'Drop':<6} "
```

and add the value to each line:

```python
                f"{r.get('feature_window_size', DEFAULT_FEATURE_WINDOW_SIZE):<6} "
```

Add `"feature_window_size"` to the CSV fields after `"channels"`.

- [ ] **Step 6: Run ablation tests and help check**

Run:

```bash
pytest tests/test_feature_window_size.py::test_get_experiments_adds_window_size_sweep_without_duplicating_baseline -q
```

Expected: PASS.

Run:

```bash
python ablation_study.py --help
```

Expected: output contains `--window_sizes`.

---

### Task 5: Documentation And Full Verification

**Files:**
- Modify: `README.md`
- Test: all focused tests

- [ ] **Step 1: Update README train/test usage blocks**

Add `--feature_window_size FEATURE_WINDOW_SIZE` to trainer and tester usage text. Add this bullet under both updated defaults sections:

```markdown
- `--feature_window_size`: 3, controls rolling mean/std and t-stat feature embedding window size
```

- [ ] **Step 2: Add ablation parameter note**

Add this short section under `tester.py`:

```markdown
### ablation_study.py

Run ablation experiments, including optional feature embedding window-size sweeps.

```shell
python ablation_study.py -trp example/zymo/ -trn example/human/ -tep example/zymo/ -ten example/human/ -o example/result/ablation --window_sizes 3 5 7
```

- `--window_sizes`: one or more feature embedding window sizes for the ablation sweep, default `3`
```

- [ ] **Step 3: Run all tests**

Run:

```bash
pytest tests/test_feature_window_size.py -q
```

Expected: PASS.

- [ ] **Step 4: Run syntax checks**

Run:

```bash
python -m py_compile preprocessor.py dataset.py trainer.py tester.py ablation_study.py
```

Expected: exits with status 0.

- [ ] **Step 5: Review changed files**

Run:

```bash
git diff -- preprocessor.py dataset.py trainer.py tester.py ablation_study.py README.md tests/test_feature_window_size.py
```

Expected: only feature-window parameter, ablation sweep, docs, and tests changed.
