"""Tests for the HAR loader and the SSL classification runner."""

import numpy as np
import torch

from chronojepa.data import load_har
from chronojepa.eval import format_ssl_classification_table, run_ssl_classification

_SIGNALS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)


def _write_mini_har(root) -> None:
    for split, n in (("train", 3), ("test", 2)):
        signals = root / split / "Inertial Signals"
        signals.mkdir(parents=True)
        for s in _SIGNALS:
            rows = "\n".join(" ".join(f"{i + 0.1 * j:.4f}" for j in range(4)) for i in range(n))
            (signals / f"{s}_{split}.txt").write_text(rows + "\n")
        (root / split / f"y_{split}.txt").write_text("\n".join("1 2 1"[: 2 * n].split()) + "\n")


def test_load_har_shapes(tmp_path) -> None:
    _write_mini_har(tmp_path)
    x_train, y_train, x_test, y_test = load_har(tmp_path)
    assert x_train.shape == (3, 9, 4)
    assert x_test.shape == (2, 9, 4)
    assert y_train.shape == (3,) and x_train.dtype == np.float32


def test_run_ssl_classification_structure() -> None:
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((40, 3, 48)).astype(np.float32)
    x_test = rng.standard_normal((20, 3, 48)).astype(np.float32)
    y_train = np.array([0, 1] * 20)
    y_test = np.array([0, 1] * 10)
    agg = run_ssl_classification(
        x_train,
        y_train,
        x_test,
        y_test,
        placements=("pooled", "dual"),
        seeds=(0,),
        steps=3,
        d_model=16,
        num_slices=16,
        patch_len=16,
        stride=8,
        device=torch.device("cpu"),
    )
    for placement in ("pooled", "dual"):
        for key in (
            "linear_pooled",
            "mlp_pooled",
            "linear_token",
            "mlp_token",
            "across_time_variance",
        ):
            entry = agg[placement][key]
            assert {"mean", "std", "values"} <= set(entry)
            assert np.isfinite(entry["mean"])
    assert "placement" in format_ssl_classification_table(agg)
