# Feature Embedding Window Size Design

## Goal

CAFES currently computes feature embeddings with a hard-coded default window size of 3 in `add_features(..., w_len=3)`. This affects rolling-window mean, rolling-window standard deviation, and t-statistic channels. Users need to override this value for normal training/testing, and the ablation study needs to evaluate the influence of different feature window sizes.

## Scope

- Add a user-facing feature embedding window-size parameter while preserving the default value of 3.
- Thread the parameter through all feature-generation paths used by training, testing, lazy training datasets, and ablation.
- Add an ablation sweep over window sizes using full baseline features.
- Record the selected window size in logs and CSV/summary outputs.
- Do not change model architecture, channel ordering, normalization behavior, patch extraction behavior, or existing defaults.

## CLI Design

Normal training and testing use a single integer:

- `trainer.py`: `--feature_window_size`, short alias `-fws`, default `3`
- `tester.py`: `--feature_window_size`, short alias `-fws`, default `3`

Ablation uses a list:

- `ablation_study.py`: `--window_sizes`, default `[3]`

When ablation receives multiple values, it creates additional experiments named `window_size_<value>` in group `feature_window_size`. Each window-size experiment uses the baseline full-feature model settings:

- channels: `[0, 1, 2, 3, 4, 5]`
- SEBlock: enabled
- dropout: `0.2`
- norm: enabled
- first batch norm: enabled

The existing baseline experiment remains the shared reference. If `--window_sizes` is only `3`, the ablation run keeps current behavior and does not duplicate the baseline with another `window_size_3` experiment.

## Data Flow

`preprocessor.add_features` remains the single embedding implementation. Its existing `w_len` parameter is reused.

The selected window size is passed to:

- validation feature generation in `trainer.py`
- lazy train feature generation via `LazyTrainDataset`
- test-time feature generation in `tester.py`
- train, validation, and test feature generation in `ablation_study.py`

`LazyTrainDataset` gains a `feature_window_size` constructor parameter with default `3` and passes it to `add_features`.

## Validation And Errors

The feature window size must be an integer greater than or equal to 2. This matches the t-statistic implementation, which returns zeros for windows smaller than 2, and avoids silent invalid configurations.

The existing behavior for sequences shorter than the configured signal length remains unchanged.

## Outputs

Training and testing logs already print all parsed CLI arguments, so `feature_window_size` will appear automatically.

Ablation outputs add a `feature_window_size` field to:

- per-experiment dictionaries
- `ablation_results.csv`
- formatted summary table

Plots continue grouping by ablation group. A new `ablation_feature_window_size.pdf` is generated when multiple window sizes are evaluated.

## Tests

Add focused tests for:

- `add_features` accepts non-default `w_len` and keeps output shape `(N, 6, L)`.
- rolling mean/std channels differ when `w_len` changes.
- `LazyTrainDataset` passes `feature_window_size` through to `add_features`.
- ablation experiment generation adds `feature_window_size` experiments when multiple window sizes are configured and keeps the default run non-duplicated.

These tests avoid full model training and should run quickly.
