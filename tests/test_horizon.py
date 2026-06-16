"""Tests for the forecasting-vs-horizon sweep."""

import numpy as np
import torch

from chronojepa.eval import format_horizon_table, run_horizon_sweep


def _series(num_steps: int = 800, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, num_steps)[:, None]
    return (np.sin(2.0 * np.pi * 4.0 * t) + rng.standard_normal((num_steps, channels))).astype(
        np.float32
    )


def test_run_horizon_sweep_structure() -> None:
    out = run_horizon_sweep(
        _series(),
        horizons=(4, 8),
        placements=("pooled", "dual"),
        seeds=(0,),
        steps=5,
        window=48,
        stride=8,
        batch_size=16,
        d_model=16,
        num_slices=16,
        device=torch.device("cpu"),
    )
    assert set(out) == {"4", "8"}
    for horizon in ("4", "8"):
        for placement in ("pooled", "dual"):
            for metric in ("mae", "mse"):
                entry = out[horizon][placement][metric]
                assert set(entry) == {"mean", "std", "values"}
                assert len(entry["values"]) == 1
                assert np.isfinite(entry["mean"])
    assert "horizon" in format_horizon_table(out)
