"""Run the placement comparison: train each placement, then diagnose and probe it."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from chronojepa.data import TwoViewAugmentation, build_dataloaders, sliding_windows
from chronojepa.models import PatchTSTEncoder
from chronojepa.sigreg import make_sigreg
from chronojepa.train import train
from chronojepa.utils.devices import get_device
from chronojepa.utils.seed import set_seed

from .anomaly import MahalanobisScorer, inject_anomaly
from .collapse import collapse_report
from .model_selection import label_free_model_selection
from .probes import extract_features, forecast_linear_probe, linear_probe


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


def _train_placement(
    series: np.ndarray,
    placement: str,
    *,
    steps: int,
    window: int,
    stride: int,
    batch_size: int,
    d_model: int,
    num_slices: int,
    lam: float,
    seed: int,
    device: torch.device,
    pos_encoding: bool = True,
):
    """Train one placement and return the frozen encoder, scaler, splits, and training logger."""
    set_seed(seed)
    loaders, scaler, splits = build_dataloaders(
        series,
        window=window,
        stride=stride,
        batch_size=batch_size,
        augment=TwoViewAugmentation(jitter_sigma=0.1, scaling_sigma=0.1, mask_ratio=0.1),
        seed=seed,
    )
    encoder = PatchTSTEncoder(
        num_channels=series.shape[1],
        patch_len=16,
        stride=8,
        d_model=d_model,
        depth=2,
        n_heads=4,
        pos_encoding=pos_encoding,
    )
    logger = train(
        encoder,
        make_sigreg(placement, num_slices=num_slices),
        loaders["train"],
        steps=steps,
        lam=lam,
        device=device,
        seed=seed,
    )
    return encoder.to(device).eval(), scaler, splits, logger


def _anomaly_auroc_one_seed(
    series: np.ndarray,
    *,
    placements: tuple[str, ...],
    kinds: tuple[str, ...],
    steps: int,
    window: int,
    stride: int,
    batch_size: int,
    d_model: int,
    num_slices: int,
    lam: float,
    seed: int,
    strength: float,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, float]] = {}
    for placement in placements:
        encoder, scaler, splits, _ = _train_placement(
            series,
            placement,
            steps=steps,
            window=window,
            stride=stride,
            batch_size=batch_size,
            d_model=d_model,
            num_slices=num_slices,
            lam=lam,
            seed=seed,
            device=device,
        )
        normalized = scaler.transform(series)
        train_windows, _ = sliding_windows(normalized, *splits["train"], window, stride)
        test_windows, _ = sliding_windows(normalized, *splits["test"], window, stride)

        # Mahalanobis on flattened token features so temporal structure is available.
        scorer = MahalanobisScorer().fit(
            extract_features(encoder, torch.from_numpy(train_windows), device, pool=False)
        )
        normal_scores = scorer.score(
            extract_features(encoder, torch.from_numpy(test_windows), device, pool=False)
        )

        result[placement] = {}
        for kind in kinds:
            anomalous = inject_anomaly(test_windows, kind, rng, strength)
            anomalous_scores = scorer.score(
                extract_features(encoder, torch.from_numpy(anomalous), device, pool=False)
            )
            labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomalous_scores))])
            scores = np.concatenate([normal_scores, anomalous_scores])
            result[placement][kind] = float(roc_auc_score(labels, scores))
    return result


def run_anomaly_comparison(
    series: np.ndarray,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    placements: tuple[str, ...] = ("pooled", "dual"),
    kinds: tuple[str, ...] = ("spike", "shuffle", "block_shuffle"),
    steps: int = 500,
    window: int = 96,
    stride: int = 8,
    batch_size: int = 32,
    d_model: int = 32,
    num_slices: int = 32,
    lam: float = 0.5,
    strength: float = 4.0,
    device: torch.device | None = None,
    results_path: str | Path | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare placements on anomaly detection AUROC, aggregated over seeds.

    For each placement a frozen encoder is trained, a Mahalanobis scorer is fit on train
    token features, and AUROC is measured separating real test windows from windows with each
    injected anomaly. Returns ``{placement: {kind: {mean, std, values}}}``.
    """
    device = device or get_device()
    per_seed = [
        _anomaly_auroc_one_seed(
            series,
            placements=placements,
            kinds=kinds,
            steps=steps,
            window=window,
            stride=stride,
            batch_size=batch_size,
            d_model=d_model,
            num_slices=num_slices,
            lam=lam,
            seed=seed,
            strength=strength,
            device=device,
        )
        for seed in seeds
    ]

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for placement in placements:
        aggregate[placement] = {}
        for kind in kinds:
            values = [run[placement][kind] for run in per_seed]
            aggregate[placement][kind] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(v) for v in values],
            }

    if results_path is not None:
        path = Path(results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(aggregate, indent=2))
    return aggregate


def run_horizon_sweep(
    series: np.ndarray,
    *,
    horizons: tuple[int, ...] = (3, 6, 12, 24, 48),
    placements: tuple[str, ...] = ("pooled", "dual"),
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 500,
    window: int = 96,
    stride: int = 8,
    batch_size: int = 32,
    d_model: int = 32,
    num_slices: int = 32,
    lam: float = 0.5,
    device: torch.device | None = None,
    results_path: str | Path | None = None,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Trajectory-forecasting MAE and MSE per placement as the horizon grows.

    Each (placement, seed) encoder is trained once, then the trajectory probe is fit at every
    horizon on the frozen token features, so the only thing that varies across horizons is the
    task. Tests whether dual's preserved temporal structure helps more once short-horizon
    persistence breaks down. Returns ``{horizon: {placement: {mae|mse: {mean, std, values}}}}``.
    """
    device = device or get_device()
    raw = {h: {p: {"mae": [], "mse": []} for p in placements} for h in horizons}

    for seed in seeds:
        for placement in placements:
            encoder, scaler, splits, _ = _train_placement(
                series,
                placement,
                steps=steps,
                window=window,
                stride=stride,
                batch_size=batch_size,
                d_model=d_model,
                num_slices=num_slices,
                lam=lam,
                seed=seed,
                device=device,
            )
            normalized = scaler.transform(series)
            for horizon in horizons:
                x_train, y_train = _forecast_windows(
                    normalized, *splits["train"], window, horizon, stride, mode="trajectory"
                )
                x_val, y_val = _forecast_windows(
                    normalized, *splits["val"], window, horizon, stride, mode="trajectory"
                )
                forecast = forecast_linear_probe(
                    extract_features(encoder, torch.from_numpy(x_train), device, pool=False),
                    y_train,
                    extract_features(encoder, torch.from_numpy(x_val), device, pool=False),
                    y_val,
                )
                raw[horizon][placement]["mae"].append(forecast["mae"])
                raw[horizon][placement]["mse"].append(forecast["mse"])

    aggregate: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for horizon in horizons:
        aggregate[str(horizon)] = {}
        for placement in placements:
            aggregate[str(horizon)][placement] = {}
            for metric in ("mae", "mse"):
                values = raw[horizon][placement][metric]
                aggregate[str(horizon)][placement][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": [float(v) for v in values],
                }

    if results_path is not None:
        path = Path(results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(aggregate, indent=2))
    return aggregate


def format_horizon_table(aggregate: dict[str, dict[str, dict[str, dict[str, float]]]]) -> str:
    """Render horizon vs placement trajectory MAE (mean +- std) and the dual minus pooled gap."""
    placements = list(next(iter(aggregate.values())))
    header = (
        f"{'horizon':<8}"
        + "".join(f"{p + ' MAE':>20}" for p in placements)
        + f"{'dual-pooled':>14}"
    )
    lines = [header, "-" * len(header)]
    for horizon, per_placement in aggregate.items():
        cells = [f"{horizon:<8}"]
        means = {}
        for placement in placements:
            entry = per_placement[placement]["mae"]
            means[placement] = entry["mean"]
            cells.append(f"{entry['mean']:.4f}+-{entry['std']:.4f}".rjust(20))
        if "pooled" in means and "dual" in means:
            cells.append(f"{means['dual'] - means['pooled']:+.4f}".rjust(14))
        lines.append("".join(cells))
    return "\n".join(lines)


def format_anomaly_table(aggregate: dict[str, dict[str, dict[str, float]]]) -> str:
    """Render the anomaly comparison as placement vs anomaly-kind AUROC, mean plus or minus std."""
    kinds = list(next(iter(aggregate.values())))
    header = f"{'placement':<12}" + "".join(f"{kind + ' AUROC':>22}" for kind in kinds)
    lines = [header, "-" * len(header)]
    for placement, per_kind in aggregate.items():
        cells = [f"{placement:<12}"]
        for kind in kinds:
            entry = per_kind[kind]
            cells.append(f"{entry['mean']:.4f}+-{entry['std']:.4f}".rjust(22))
        lines.append("".join(cells))
    return "\n".join(lines)


_LAMBDA_METRICS = (
    "sigreg_loss",
    "across_time_variance",
    "effective_rank",
    "forecast_mae",
    "forecast_mse",
)


def run_lambda_sweep(
    series: np.ndarray,
    *,
    lambdas: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    placements: tuple[str, ...] = ("pooled", "dual"),
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 500,
    window: int = 96,
    horizon: int = 12,
    stride: int = 8,
    batch_size: int = 32,
    d_model: int = 32,
    num_slices: int = 32,
    final_window: int = 50,
    device: torch.device | None = None,
    results_path: str | Path | None = None,
) -> dict:
    """Sweep lambda across placements and report collapse, downstream, and label-free selection.

    For each (placement, lambda, seed) it records the final SIGReg loss (mean over the last
    ``final_window`` steps), the collapse metrics, and the trajectory-forecasting MAE and MSE.
    Then it ranks all configs by their label-free SIGReg loss and correlates that with the
    labeled downstream metric, testing whether SIGReg loss selects models without labels.
    Returns ``{"configs": {name: {metric: {mean, std}}}, "label_free": {...}}``.
    """
    device = device or get_device()
    raw: dict[str, dict[str, list[float]]] = {}

    for seed in seeds:
        for placement in placements:
            for lam in lambdas:
                name = f"{placement}|lam{lam}"
                encoder, scaler, splits, logger = _train_placement(
                    series,
                    placement,
                    steps=steps,
                    window=window,
                    stride=stride,
                    batch_size=batch_size,
                    d_model=d_model,
                    num_slices=num_slices,
                    lam=lam,
                    seed=seed,
                    device=device,
                )
                final_sigreg = float(
                    np.mean([record["sigreg"] for record in logger.history[-final_window:]])
                )
                normalized = scaler.transform(series)
                x_train, y_train = _forecast_windows(
                    normalized, *splits["train"], window, horizon, stride, mode="trajectory"
                )
                x_val, y_val = _forecast_windows(
                    normalized, *splits["val"], window, horizon, stride, mode="trajectory"
                )
                with torch.no_grad():
                    tokens, _ = encoder(torch.from_numpy(x_val).to(device))
                report = collapse_report(tokens.cpu())
                forecast = forecast_linear_probe(
                    extract_features(encoder, torch.from_numpy(x_train), device, pool=False),
                    y_train,
                    extract_features(encoder, torch.from_numpy(x_val), device, pool=False),
                    y_val,
                )
                record = raw.setdefault(name, {key: [] for key in _LAMBDA_METRICS})
                record["sigreg_loss"].append(final_sigreg)
                record["across_time_variance"].append(report["across_time_variance"])
                record["effective_rank"].append(report["effective_rank"])
                record["forecast_mae"].append(forecast["mae"])
                record["forecast_mse"].append(forecast["mse"])

    configs: dict[str, dict[str, dict[str, float]]] = {}
    for name, record in raw.items():
        configs[name] = {
            key: {"mean": float(np.mean(record[key])), "std": float(np.std(record[key]))}
            for key in _LAMBDA_METRICS
        }

    names = list(configs)
    label_free = label_free_model_selection(
        names,
        [configs[n]["sigreg_loss"]["mean"] for n in names],
        [configs[n]["forecast_mae"]["mean"] for n in names],
        lower_is_better=True,
    )
    out = {"configs": configs, "label_free": label_free}

    if results_path is not None:
        path = Path(results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
    return out


def format_lambda_table(out: dict) -> str:
    """Render the lambda sweep: per-config metrics plus the label-free selection summary."""
    configs = out["configs"]
    header = f"{'config':<16}{'sigreg':>12}{'across_time':>14}{'eff_rank':>10}{'fc_mae':>12}"
    lines = [header, "-" * len(header)]
    for name in sorted(configs):
        c = configs[name]
        lines.append(
            f"{name:<16}{c['sigreg_loss']['mean']:>12.4f}"
            f"{c['across_time_variance']['mean']:>14.4f}"
            f"{c['effective_rank']['mean']:>10.3f}{c['forecast_mae']['mean']:>12.4f}"
        )
    lf = out["label_free"]
    lines += [
        "",
        f"label-free selection: spearman(sigreg_loss, forecast_mae) = "
        f"{lf['spearman']:.3f} (p={lf['pvalue']:.3f})",
        f"label-free pick = {lf['label_free_pick']}   "
        f"label-based pick = {lf['label_based_pick']}   agree = {lf['agree']}",
    ]
    return "\n".join(lines)


def _classification_labels(
    windows: np.ndarray, kind: str, threshold: float | None = None
) -> tuple[np.ndarray, float | None]:
    """Build per-window binary labels from ``(N, C, T)`` windows.

    ``trend`` labels whether the second half's mean exceeds the first half's, a pure
    temporal-order property uncorrelated with the overall level. ``level`` labels whether the
    window mean is above a threshold (the train median, passed in for test to avoid look-ahead).
    """
    if kind == "trend":
        half = windows.shape[2] // 2
        first = windows[:, :, :half].mean(axis=(1, 2))
        second = windows[:, :, half:].mean(axis=(1, 2))
        return (second > first).astype(int), None
    if kind == "level":
        means = windows.mean(axis=(1, 2))
        cut = float(np.median(means)) if threshold is None else threshold
        return (means > cut).astype(int), cut
    raise ValueError(f"unknown label kind {kind!r}")


def _halfswap_accuracy(
    encoder, train_windows, test_windows, features_train, features_test, device, pool
) -> float:
    """Classify a window (label 0) against the same window with halves swapped (label 1).

    Swapping the two halves keeps the exact value multiset and changes only temporal
    position, so this isolates position. The original-window features are reused; only the
    swapped versions need a fresh forward pass.
    """
    half = train_windows.shape[2] // 2
    swapped_train = np.roll(train_windows, half, axis=2)
    swapped_test = np.roll(test_windows, half, axis=2)
    x_train = np.concatenate(
        [
            features_train,
            extract_features(encoder, torch.from_numpy(swapped_train), device, pool=pool),
        ]
    )
    x_test = np.concatenate(
        [
            features_test,
            extract_features(encoder, torch.from_numpy(swapped_test), device, pool=pool),
        ]
    )
    y_train = np.concatenate([np.zeros(len(features_train)), np.ones(len(swapped_train))]).astype(
        int
    )
    y_test = np.concatenate([np.zeros(len(features_test)), np.ones(len(swapped_test))]).astype(int)
    return linear_probe(x_train, y_train, x_test, y_test)


def run_classification_comparison(
    series: np.ndarray,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    placements: tuple[str, ...] = ("pooled", "dual"),
    label_kinds: tuple[str, ...] = ("trend", "level", "halfswap"),
    steps: int = 500,
    window: int = 96,
    stride: int = 8,
    batch_size: int = 32,
    d_model: int = 32,
    num_slices: int = 32,
    lam: float = 0.5,
    pool: bool = False,
    pos_encoding: bool = True,
    device: torch.device | None = None,
    results_path: str | Path | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Linear-probe classification accuracy per placement for temporal and level labels.

    ``trend`` needs temporal order, so it is where dual's preserved per-timestep structure
    should help if it helps anywhere; ``level`` is the control a collapsed representation can
    do. Features come from the frozen encoder (token features by default). Returns
    ``{placement: {kind: {mean, std, values}}}``.
    """
    device = device or get_device()
    raw = {p: {k: [] for k in label_kinds} for p in placements}

    for seed in seeds:
        for placement in placements:
            encoder, scaler, splits, _ = _train_placement(
                series,
                placement,
                steps=steps,
                window=window,
                stride=stride,
                batch_size=batch_size,
                d_model=d_model,
                num_slices=num_slices,
                lam=lam,
                seed=seed,
                device=device,
                pos_encoding=pos_encoding,
            )
            normalized = scaler.transform(series)
            train_windows, _ = sliding_windows(normalized, *splits["train"], window, stride)
            test_windows, _ = sliding_windows(normalized, *splits["test"], window, stride)
            features_train = extract_features(
                encoder, torch.from_numpy(train_windows), device, pool=pool
            )
            features_test = extract_features(
                encoder, torch.from_numpy(test_windows), device, pool=pool
            )
            for kind in label_kinds:
                if kind == "halfswap":
                    accuracy = _halfswap_accuracy(
                        encoder,
                        train_windows,
                        test_windows,
                        features_train,
                        features_test,
                        device,
                        pool,
                    )
                else:
                    y_train, cut = _classification_labels(train_windows, kind)
                    y_test, _ = _classification_labels(test_windows, kind, threshold=cut)
                    accuracy = linear_probe(features_train, y_train, features_test, y_test)
                raw[placement][kind].append(accuracy)

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for placement in placements:
        aggregate[placement] = {}
        for kind in label_kinds:
            values = raw[placement][kind]
            aggregate[placement][kind] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": [float(v) for v in values],
            }

    if results_path is not None:
        path = Path(results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(aggregate, indent=2))
    return aggregate


def format_classification_table(aggregate: dict[str, dict[str, dict[str, float]]]) -> str:
    """Render placement vs label-kind classification accuracy, mean plus or minus std."""
    kinds = list(next(iter(aggregate.values())))
    header = f"{'placement':<12}" + "".join(f"{kind + ' acc':>22}" for kind in kinds)
    lines = [header, "-" * len(header)]
    for placement, per_kind in aggregate.items():
        cells = [f"{placement:<12}"]
        for kind in kinds:
            entry = per_kind[kind]
            cells.append(f"{entry['mean']:.4f}+-{entry['std']:.4f}".rjust(22))
        lines.append("".join(cells))
    return "\n".join(lines)
