"""Tests for windowing, no-look-ahead normalization, and the two-view pipeline."""

import numpy as np
import torch

from chronojepa.data import (
    StandardScaler,
    TwoViewAugmentation,
    WindowDataset,
    build_dataloaders,
    sliding_windows,
    time_split,
)


def _series(num_steps: int = 1000, channels: int = 3) -> np.ndarray:
    # A clear linear trend makes train statistics differ from full-series statistics,
    # so a scaler that peeks at val/test would be caught.
    t = np.linspace(0.0, 1.0, num_steps)[:, None]
    trend = 10.0 * t
    seasonal = np.sin(2.0 * np.pi * 5.0 * t)
    noise = np.random.default_rng(0).standard_normal((num_steps, channels))
    return (trend + seasonal + noise).astype(np.float32)


def test_splits_are_time_ordered_and_disjoint() -> None:
    train, val, test = time_split(1000, train_frac=0.7, val_frac=0.1)
    assert train == (0, 700) and val == (700, 800) and test == (800, 1000)
    assert train[1] <= val[0] and val[1] <= test[0]


def test_windows_stay_within_their_split() -> None:
    windows, starts = sliding_windows(_series(1000), 700, 800, window=24, stride=12)
    assert windows.shape[1:] == (3, 24)  # (num, channels, time)
    assert starts.min() >= 700
    assert starts.max() + 24 <= 800


def test_scaler_uses_train_statistics_only() -> None:
    series = _series(1000)
    train, _, _ = time_split(1000)
    scaler = StandardScaler().fit(series[train[0] : train[1]])
    assert np.allclose(scaler.mean_, series[:700].mean(axis=0))
    assert not np.allclose(scaler.mean_, series.mean(axis=0))


def test_two_views_have_matching_shapes() -> None:
    series = _series(200)
    scaler = StandardScaler().fit(series[:140])
    windows, _ = sliding_windows(scaler.transform(series), 0, 140, window=48, stride=24)
    augment = TwoViewAugmentation(
        jitter_sigma=0.1, scaling_sigma=0.1, mask_ratio=0.2, crop_ratio=0.8
    )
    view1, view2, clean = WindowDataset(windows, augment=augment)[0]
    assert view1.shape == view2.shape
    assert view1.shape[0] == series.shape[1]  # channels preserved
    assert clean.shape == (series.shape[1], 48)


def test_build_dataloaders_smoke_prints_shapes() -> None:
    series = _series(1000)
    loaders, scaler, splits = build_dataloaders(
        series,
        window=24,
        stride=12,
        train_frac=0.7,
        val_frac=0.1,
        batch_size=8,
        augment=TwoViewAugmentation(),
        seed=0,
    )
    view1, view2, clean = next(iter(loaders["train"]))
    shapes = (tuple(view1.shape), tuple(view2.shape), tuple(clean.shape))
    print(f"train batch: view1={shapes[0]} view2={shapes[1]} clean={shapes[2]}")
    assert view1.shape == view2.shape
    assert view1.shape[0] <= 8 and view1.shape[1] == 3
    assert torch.isfinite(view1).all()
    assert np.allclose(scaler.mean_, series[:700].mean(axis=0))
    assert set(loaders) == {"train", "val", "test"}
