"""Tests for anomaly injection and the anomaly-detection placement comparison."""

import numpy as np
import torch

from chronojepa.eval import (
    format_anomaly_table,
    inject_anomaly,
    run_anomaly_comparison,
)


def _series(num_steps: int = 600, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, num_steps)[:, None]
    return (np.sin(2.0 * np.pi * 4.0 * t) + rng.standard_normal((num_steps, channels))).astype(
        np.float32
    )


def test_inject_spike_changes_values_same_shape() -> None:
    windows = np.random.default_rng(0).standard_normal((5, 3, 40)).astype(np.float32)
    out = inject_anomaly(windows, "spike", np.random.default_rng(1), strength=5.0)
    assert out.shape == windows.shape
    assert not np.array_equal(out, windows)


def test_inject_shuffle_is_a_time_permutation() -> None:
    windows = np.random.default_rng(0).standard_normal((5, 3, 40)).astype(np.float32)
    out = inject_anomaly(windows, "shuffle", np.random.default_rng(1))
    assert out.shape == windows.shape
    # a permutation along time preserves each window's sorted values per channel
    assert np.allclose(np.sort(out, axis=2), np.sort(windows, axis=2))
    assert not np.array_equal(out, windows)


def test_inject_block_shuffle_is_a_time_permutation() -> None:
    windows = np.random.default_rng(0).standard_normal((5, 3, 48)).astype(np.float32)
    out = inject_anomaly(windows, "block_shuffle", np.random.default_rng(1), block=8)
    assert out.shape == windows.shape
    assert np.allclose(np.sort(out, axis=2), np.sort(windows, axis=2))


def test_run_anomaly_comparison_structure() -> None:
    agg = run_anomaly_comparison(
        _series(),
        seeds=(0,),
        placements=("pooled", "dual"),
        kinds=("spike", "shuffle"),
        steps=5,
        window=32,
        stride=8,
        batch_size=16,
        d_model=16,
        num_slices=16,
        device=torch.device("cpu"),
    )
    for placement in ("pooled", "dual"):
        for kind in ("spike", "shuffle"):
            entry = agg[placement][kind]
            assert set(entry) == {"mean", "std", "values"}
            assert 0.0 <= entry["mean"] <= 1.0
            assert np.isfinite(entry["std"])
    assert "placement" in format_anomaly_table(agg)
