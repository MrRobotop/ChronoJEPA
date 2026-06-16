"""Tests for token-feature extraction, trajectory forecasting, and multi-seed runs."""

import numpy as np
import torch

from chronojepa.eval import (
    extract_features,
    format_multiseed_table,
    run_multiseed_comparison,
    run_placement_comparison,
)
from chronojepa.models import PatchTSTEncoder


def _series(num_steps: int = 600, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, num_steps)[:, None]
    return (np.sin(2.0 * np.pi * 4.0 * t) + rng.standard_normal((num_steps, channels))).astype(
        np.float32
    )


def test_extract_features_pool_false_returns_flattened_tokens() -> None:
    encoder = PatchTSTEncoder(
        num_channels=3, patch_len=16, stride=8, d_model=32, depth=1, n_heads=4
    )
    windows = torch.randn(8, 3, 64)
    pooled = extract_features(encoder, windows, torch.device("cpu"), pool=True)
    tokens = extract_features(encoder, windows, torch.device("cpu"), pool=False)
    num_patches = (64 - 16) // 8 + 1
    assert pooled.shape == (8, 32)
    assert tokens.shape == (8, num_patches * 32)


def test_trajectory_forecast_runs_and_is_finite() -> None:
    results = run_placement_comparison(
        _series(),
        placements=("pooled", "dual"),
        steps=8,
        window=32,
        horizon=8,
        stride=8,
        batch_size=16,
        d_model=32,
        num_slices=16,
        device=torch.device("cpu"),
        forecast_mode="trajectory",
    )
    for metric in results.values():
        assert all(np.isfinite(v) for v in metric.values())


def test_multiseed_comparison_aggregates_mean_and_std() -> None:
    agg = run_multiseed_comparison(
        _series(),
        seeds=(0, 1),
        placements=("pooled", "dual"),
        steps=5,
        window=32,
        horizon=8,
        stride=8,
        batch_size=16,
        d_model=32,
        num_slices=16,
        device=torch.device("cpu"),
        forecast_mode="trajectory",
    )
    for placement in ("pooled", "dual"):
        for metric in ("across_time_variance", "effective_rank", "forecast_mae", "forecast_mse"):
            entry = agg[placement][metric]
            assert set(entry) == {"mean", "std", "values"}
            assert len(entry["values"]) == 2
            assert np.isfinite(entry["mean"]) and np.isfinite(entry["std"])
    assert "pooled" in format_multiseed_table(agg)
