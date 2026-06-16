"""Run the placement comparison: train each placement, then diagnose and probe it."""

import json
from pathlib import Path

import numpy as np
import torch

from chronojepa.data import TwoViewAugmentation, build_dataloaders
from chronojepa.models import PatchTSTEncoder
from chronojepa.sigreg import make_sigreg
from chronojepa.train import train
from chronojepa.utils.devices import get_device
from chronojepa.utils.seed import set_seed

from .collapse import collapse_report
from .probes import extract_features, forecast_linear_probe


def _forecast_windows(
    series: np.ndarray,
    start: int,
    end: int,
    window: int,
    horizon: int,
    stride: int,
    mode: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Build (input window, target) forecasting pairs within a split.

    With ``mode="mean"`` the target is the per-channel mean over the horizon ``(n, C)``. With
    ``mode="trajectory"`` it is the full next-horizon trajectory flattened ``(n, horizon * C)``,
    which depends on temporal structure and is a fairer test of preventing collapse.
    """
    starts = np.arange(start, end - window - horizon + 1, stride)
    if starts.size == 0:
        raise ValueError(
            f"split [{start}, {end}) is too small for window {window} plus horizon {horizon}"
        )
    inputs = np.stack([series[s : s + window].T for s in starts]).astype(np.float32)
    if mode == "mean":
        targets = np.stack([series[s + window : s + window + horizon].mean(axis=0) for s in starts])
    elif mode == "trajectory":
        targets = np.stack([series[s + window : s + window + horizon].reshape(-1) for s in starts])
    else:
        raise ValueError(f"unknown forecast mode {mode!r}")
    return inputs, targets.astype(np.float32)


def run_placement_comparison(
    series: np.ndarray,
    *,
    placements: tuple[str, ...] = ("pooled", "dual"),
    steps: int = 100,
    window: int = 64,
    horizon: int = 12,
    stride: int = 8,
    batch_size: int = 32,
    d_model: int = 64,
    num_slices: int = 32,
    lam: float = 0.5,
    seed: int = 0,
    device: torch.device | None = None,
    forecast_mode: str = "mean",
    results_path: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Train each placement and report collapse and forecasting metrics on the val split.

    ``forecast_mode="mean"`` probes the pooled feature against the next-horizon mean;
    ``forecast_mode="trajectory"`` probes the flattened token sequence against the full
    horizon, a fairer test of whether preventing collapse helps downstream. Returns
    ``{placement: {across_time_variance, effective_rank, forecast_mae, forecast_mse}}`` and
    optionally writes it as JSON. Splits are time-ordered and the scaler is train-only, so
    there is no look-ahead.
    """
    device = device or get_device()
    channels = series.shape[1]
    pool_features = forecast_mode == "mean"

    loaders, scaler, splits = build_dataloaders(
        series,
        window=window,
        stride=stride,
        batch_size=batch_size,
        augment=TwoViewAugmentation(jitter_sigma=0.1, scaling_sigma=0.1, mask_ratio=0.1),
        seed=seed,
    )
    normalized = scaler.transform(series)
    x_train, y_train = _forecast_windows(
        normalized, *splits["train"], window, horizon, stride, mode=forecast_mode
    )
    x_val, y_val = _forecast_windows(
        normalized, *splits["val"], window, horizon, stride, mode=forecast_mode
    )

    results: dict[str, dict[str, float]] = {}
    for name in placements:
        set_seed(seed)
        encoder = PatchTSTEncoder(
            num_channels=channels, patch_len=16, stride=8, d_model=d_model, depth=2, n_heads=4
        )
        train(
            encoder,
            make_sigreg(name, num_slices=num_slices),
            loaders["train"],
            steps=steps,
            lam=lam,
            device=device,
            seed=seed,
        )

        encoder = encoder.to(device).eval()
        with torch.no_grad():
            tokens, _ = encoder(torch.from_numpy(x_val).to(device))
        report = collapse_report(tokens.cpu())

        features_train = extract_features(
            encoder, torch.from_numpy(x_train), device, pool=pool_features
        )
        features_val = extract_features(
            encoder, torch.from_numpy(x_val), device, pool=pool_features
        )
        forecast = forecast_linear_probe(features_train, y_train, features_val, y_val)

        results[name] = {
            "across_time_variance": report["across_time_variance"],
            "effective_rank": report["effective_rank"],
            "forecast_mae": forecast["mae"],
            "forecast_mse": forecast["mse"],
        }

    if results_path is not None:
        path = Path(results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
    return results


def format_comparison_table(results: dict[str, dict[str, float]]) -> str:
    """Render the comparison as a fixed-width table: placement vs metrics."""
    header = (
        f"{'placement':<12} {'across_time_var':>16} {'eff_rank':>10} "
        f"{'fcast_mae':>10} {'fcast_mse':>10}"
    )
    lines = [header, "-" * len(header)]
    for name, metric in results.items():
        lines.append(
            f"{name:<12} {metric['across_time_variance']:>16.6f} "
            f"{metric['effective_rank']:>10.3f} {metric['forecast_mae']:>10.4f} "
            f"{metric['forecast_mse']:>10.4f}"
        )
    return "\n".join(lines)


_METRICS = ("across_time_variance", "effective_rank", "forecast_mae", "forecast_mse")


def run_multiseed_comparison(
    series: np.ndarray,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    placements: tuple[str, ...] = ("pooled", "dual"),
    results_path: str | Path | None = None,
    **kwargs,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run the comparison once per seed and aggregate each metric to mean, std, and values.

    Returns ``{placement: {metric: {mean, std, values}}}`` so a one-percent single-seed gap
    can be read against its spread across seeds. Extra keyword arguments are forwarded to
    ``run_placement_comparison``.
    """
    per_seed = [
        run_placement_comparison(series, placements=placements, seed=seed, **kwargs)
        for seed in seeds
    ]

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for placement in placements:
        aggregate[placement] = {}
        for metric in _METRICS:
            values = [run[placement][metric] for run in per_seed]
            aggregate[placement][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(v) for v in values],
            }

    if results_path is not None:
        path = Path(results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(aggregate, indent=2))
    return aggregate


def format_multiseed_table(aggregate: dict[str, dict[str, dict[str, float]]]) -> str:
    """Render the multi-seed comparison as placement vs metric, each as mean plus or minus std."""
    header = (
        f"{'placement':<12} {'across_time_var':>20} {'eff_rank':>16} "
        f"{'fcast_mae':>16} {'fcast_mse':>16}"
    )
    lines = [header, "-" * len(header)]
    for placement, metrics in aggregate.items():
        cells = [f"{placement:<12}"]
        widths = {
            "across_time_variance": 20,
            "effective_rank": 16,
            "forecast_mae": 16,
            "forecast_mse": 16,
        }
        for metric in _METRICS:
            entry = metrics[metric]
            cells.append(f"{entry['mean']:.4f}+-{entry['std']:.4f}".rjust(widths[metric]))
        lines.append(" ".join(cells))
    return "\n".join(lines)
