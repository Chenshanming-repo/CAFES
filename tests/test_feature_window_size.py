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
