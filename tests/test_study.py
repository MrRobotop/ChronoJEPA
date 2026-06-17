"""Test the architecture x placement factorial study structure."""

import numpy as np
import torch

from chronojepa.eval import format_architecture_study_table, run_architecture_study


def _series(num_steps: int = 800, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, num_steps)[:, None]
    return (np.sin(2.0 * np.pi * 4.0 * t) + rng.standard_normal((num_steps, channels))).astype(
        np.float32
    )


def test_run_architecture_study_structure() -> None:
    agg = run_architecture_study(
        _series(),
        seeds=(0,),
        placements=("pooled", "dual"),
        steps=5,
        window=48,
        stride=16,
        batch_size=16,
        d_model=16,
        num_slices=16,
        device=torch.device("cpu"),
    )
    assert set(agg) == {
        "positional|pooled",
        "positional|dual",
        "bagofpatches|pooled",
        "bagofpatches|dual",
    }
    for metrics in agg.values():
        for metric in (
            "across_time_variance",
            "effective_rank",
            "halfswap_token",
            "halfswap_pooled",
            "trend_token",
            "trend_pooled",
        ):
            assert {"mean", "std", "values"} <= set(metrics[metric])
            assert np.isfinite(metrics[metric]["mean"])
    assert "arch" in format_architecture_study_table(agg)
