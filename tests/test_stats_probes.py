"""Tests for the nonlinear MLP probe and the paired-difference statistics."""

import numpy as np

from chronojepa.eval import mlp_probe, paired_difference


def test_mlp_probe_separates_blobs() -> None:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(-2.0, 1.0, (100, 8)), rng.normal(2.0, 1.0, (100, 8))]).astype(
        np.float32
    )
    y = np.array([0] * 100 + [1] * 100)
    accuracy = mlp_probe(x[::2], y[::2], x[1::2], y[1::2])
    assert accuracy > 0.9


def test_paired_difference_detects_real_gap() -> None:
    baseline = [1.00, 1.02, 0.98, 1.01, 0.99]
    treatment = [0.90, 0.92, 0.89, 0.91, 0.90]  # consistently lower
    result = paired_difference(baseline, treatment)
    assert result["mean_diff"] < 0
    assert result["ci95_high"] < 0  # CI excludes zero
    assert result["p_value"] < 0.05
    assert result["n"] == 5


def test_paired_difference_no_gap_is_not_significant() -> None:
    # Differences straddle zero symmetrically, so the mean difference is zero.
    baseline = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    treatment = [1.1, 0.9, 1.1, 0.9, 1.1, 0.9]
    result = paired_difference(baseline, treatment)
    assert result["p_value"] > 0.05
    assert result["ci95_low"] < 0 < result["ci95_high"]  # CI contains zero
