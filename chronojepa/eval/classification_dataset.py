"""SSL pretrain-then-probe evaluation on a labeled sequence-classification dataset.

Pretrains an encoder self-supervised on the (unlabeled) train sequences, then linear- and
MLP-probes the frozen features against the labels. Tests whether the dual placement's richer
representation helps a real, order-dependent classification task, and whether the conclusion
depends on the feature (pooled vs token) or the probe (linear vs nonlinear).
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from chronojepa.data import TwoViewAugmentation, WindowDataset
from chronojepa.models import BagOfPatchesEncoder, PatchTSTEncoder, TCNEncoder
from chronojepa.sigreg import make_sigreg
from chronojepa.train import train
from chronojepa.utils.devices import get_device
from chronojepa.utils.seed import set_seed

from .collapse import across_time_variance
from .probes import extract_features, linear_probe, mlp_probe

_KEYS = ("linear_pooled", "mlp_pooled", "linear_token", "mlp_token", "across_time_variance")


def _build_encoder(
    architecture: str,
    channels: int,
    d_model: int,
    patch_len: int,
    stride: int,
    depth: int,
    n_heads: int,
):
    if architecture == "positional":
        return PatchTSTEncoder(
            num_channels=channels,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
        )
    if architecture == "tcn":
        return TCNEncoder(num_channels=channels, d_model=d_model, kernel_size=3, num_layers=3)
    if architecture == "bagofpatches":
        return BagOfPatchesEncoder(num_channels=channels, patch_len=patch_len, d_model=d_model)
    raise ValueError(f"unknown architecture {architecture!r}")


def run_ssl_classification(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    placements: tuple[str, ...] = ("pooled", "dual"),
    architecture: str = "positional",
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 500,
    d_model: int = 32,
    num_slices: int = 32,
    lam: float = 0.5,
    patch_len: int = 16,
    stride: int = 8,
    depth: int = 2,
    n_heads: int = 4,
    batch_size: int = 64,
    device: torch.device | None = None,
    results_path: str | Path | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-placement classification accuracy from linear and MLP probes on pooled and token
    features, plus the across-time variance, for one encoder ``architecture`` (positional, tcn,
    or bagofpatches). Returns ``{placement: {key: {mean, std, values}}}``."""
    device = device or get_device()
    channels = x_train.shape[1]
    raw = {placement: {key: [] for key in _KEYS} for placement in placements}

    for seed in seeds:
        for placement in placements:
            set_seed(seed)
            dataset = WindowDataset(
                x_train,
                augment=TwoViewAugmentation(jitter_sigma=0.1, scaling_sigma=0.1, mask_ratio=0.1),
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            encoder = _build_encoder(
                architecture, channels, d_model, patch_len, stride, depth, n_heads
            )
            train(
                encoder,
                make_sigreg(placement, num_slices=num_slices),
                loader,
                steps=steps,
                lam=lam,
                device=device,
                seed=seed,
            )
            encoder = encoder.to(device).eval()
            with torch.no_grad():
                test_tokens, _ = encoder(torch.from_numpy(x_test).to(device))
            raw[placement]["across_time_variance"].append(across_time_variance(test_tokens.cpu()))
            for feature, pool in (("pooled", True), ("token", False)):
                features_train = extract_features(
                    encoder, torch.from_numpy(x_train), device, pool=pool
                )
                features_test = extract_features(
                    encoder, torch.from_numpy(x_test), device, pool=pool
                )
                raw[placement][f"linear_{feature}"].append(
                    linear_probe(features_train, y_train, features_test, y_test)
                )
                raw[placement][f"mlp_{feature}"].append(
                    mlp_probe(features_train, y_train, features_test, y_test, seed=seed)
                )

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for placement in placements:
        aggregate[placement] = {
            key: {
                "mean": float(np.mean(raw[placement][key])),
                "std": float(np.std(raw[placement][key])),
                "values": [float(v) for v in raw[placement][key]],
            }
            for key in _KEYS
        }

    if results_path is not None:
        path = Path(results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(aggregate, indent=2))
    return aggregate


def format_ssl_classification_table(aggregate: dict[str, dict[str, dict[str, float]]]) -> str:
    """Render placement vs probe-and-feature classification accuracy, mean plus or minus std."""
    columns = ("linear_pooled", "mlp_pooled", "linear_token", "mlp_token", "across_time_variance")
    header = f"{'placement':<10}" + "".join(f"{c:>20}" for c in columns)
    lines = [header, "-" * len(header)]
    for placement, metrics in aggregate.items():
        cells = [f"{placement:<10}"]
        for column in columns:
            entry = metrics[column]
            cells.append(f"{entry['mean']:.4f}+-{entry['std']:.4f}".rjust(20))
        lines.append("".join(cells))
    return "\n".join(lines)
