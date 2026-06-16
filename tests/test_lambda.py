"""Tests for the lambda sweep and its label-free selection analysis."""

import numpy as np
import torch

from chronojepa.eval import format_lambda_table, run_lambda_sweep


def _series(num_steps: int = 800, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, num_steps)[:, None]
    return (np.sin(2.0 * np.pi * 4.0 * t) + rng.standard_normal((num_steps, channels))).astype(
        np.float32
    )


def test_run_lambda_sweep_structure() -> None:
    out = run_lambda_sweep(
        _series(),
        lambdas=(0.3, 0.7),
        placements=("pooled", "dual"),
        seeds=(0,),
        steps=5,
        window=48,
        horizon=8,
        stride=8,
        batch_size=16,
        d_model=16,
        num_slices=16,
        final_window=5,
        device=torch.device("cpu"),
    )
    assert set(out) == {"configs", "label_free"}
    assert len(out["configs"]) == 4  # 2 placements x 2 lambdas
    for config in out["configs"].values():
        for key in (
            "sigreg_loss",
            "across_time_variance",
            "effective_rank",
            "forecast_mae",
            "forecast_mse",
        ):
            assert {"mean", "std"} <= set(config[key])
            assert np.isfinite(config[key]["mean"])
    assert -1.0 <= out["label_free"]["spearman"] <= 1.0
    assert "spearman" in format_lambda_table(out)
