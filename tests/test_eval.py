"""Tests for collapse diagnostics, probes, forecasting, and anomaly scoring."""

import numpy as np
import torch

from chronojepa.eval import (
    MahalanobisScorer,
    across_time_variance,
    collapse_report,
    effective_rank,
    extract_features,
    forecast_linear_probe,
    format_comparison_table,
    knn_probe,
    linear_probe,
    run_placement_comparison,
)
from chronojepa.models import PatchTSTEncoder


def test_across_time_variance_detects_collapse() -> None:
    collapsed = torch.randn(8, 1, 16).expand(8, 20, 16)  # constant along time
    varied = torch.randn(8, 20, 16)
    assert across_time_variance(collapsed) < 1e-6
    assert across_time_variance(varied) > 1e-2


def test_effective_rank_low_for_rank_one() -> None:
    rank_one = torch.randn(200, 1) @ torch.randn(1, 8)
    full = torch.randn(200, 8)
    assert effective_rank(rank_one) < 1.5
    assert effective_rank(full) > 4.0


def test_collapse_report_has_both_metrics() -> None:
    report = collapse_report(torch.randn(8, 20, 16))
    assert {"across_time_variance", "effective_rank"} <= set(report)
    assert all(np.isfinite(v) for v in report.values())


def test_extract_features_is_frozen_and_shaped() -> None:
    encoder = PatchTSTEncoder(
        num_channels=3, patch_len=16, stride=8, d_model=32, depth=1, n_heads=4
    )
    features = extract_features(encoder, torch.randn(10, 3, 64), device=torch.device("cpu"))
    assert isinstance(features, np.ndarray)
    assert features.shape == (10, 32)


def _two_blobs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(-2.0, 1.0, (100, 8)), rng.normal(2.0, 1.0, (100, 8))]).astype(
        np.float32
    )
    y = np.array([0] * 100 + [1] * 100)
    return x[::2], y[::2], x[1::2], y[1::2]


def test_linear_probe_separates_blobs() -> None:
    x_train, y_train, x_test, y_test = _two_blobs()
    assert linear_probe(x_train, y_train, x_test, y_test) > 0.9


def test_knn_probe_separates_blobs() -> None:
    x_train, y_train, x_test, y_test = _two_blobs()
    assert knn_probe(x_train, y_train, x_test, y_test) > 0.9


def test_forecast_linear_probe_low_error_on_linear_signal() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 8)).astype(np.float32)
    weight = rng.standard_normal((8, 3)).astype(np.float32)
    y = (x @ weight + 0.01 * rng.standard_normal((200, 3))).astype(np.float32)
    metrics = forecast_linear_probe(x[:150], y[:150], x[150:], y[150:])
    assert metrics["mae"] >= 0 and np.isfinite(metrics["mae"])
    assert metrics["mse"] < 0.1


def test_mahalanobis_scores_outliers_higher() -> None:
    rng = np.random.default_rng(0)
    train = rng.standard_normal((500, 8)).astype(np.float32)
    scorer = MahalanobisScorer().fit(train)
    inlier = scorer.score(rng.standard_normal((20, 8)).astype(np.float32))
    outlier = scorer.score((rng.standard_normal((20, 8)) + 8.0).astype(np.float32))
    assert outlier.mean() > inlier.mean()


def test_placement_comparison_structure_and_file(tmp_path) -> None:
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 600)[:, None]
    series = (np.sin(2.0 * np.pi * 4.0 * t) + rng.standard_normal((600, 3))).astype(np.float32)
    out = tmp_path / "results.json"

    results = run_placement_comparison(
        series,
        placements=("pooled", "dual"),
        steps=10,
        window=32,
        horizon=8,
        stride=8,
        batch_size=16,
        d_model=32,
        num_slices=16,
        device=torch.device("cpu"),
        results_path=out,
    )

    assert set(results) == {"pooled", "dual"}
    for metric in results.values():
        keys = {"across_time_variance", "effective_rank", "forecast_mae", "forecast_mse"}
        assert keys <= set(metric)
        assert all(np.isfinite(v) for v in metric.values())
    assert out.exists()
    assert "placement" in format_comparison_table(results)
