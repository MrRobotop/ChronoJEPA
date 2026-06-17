"""Tests for the temporal-structure classification comparison."""

import numpy as np
import torch

from chronojepa.eval import format_classification_table, run_classification_comparison


def _series(num_steps: int = 800, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, num_steps)[:, None]
    return (np.sin(2.0 * np.pi * 4.0 * t) + rng.standard_normal((num_steps, channels))).astype(
        np.float32
    )


def test_run_classification_comparison_structure() -> None:
    agg = run_classification_comparison(
        _series(),
        seeds=(0,),
        placements=("pooled", "dual"),
        label_kinds=("trend", "level"),
        steps=5,
        window=48,
        stride=8,
        batch_size=16,
        d_model=16,
        num_slices=16,
        device=torch.device("cpu"),
    )
    for placement in ("pooled", "dual"):
        for kind in ("trend", "level"):
            entry = agg[placement][kind]
            assert set(entry) == {"mean", "std", "values"}
            assert 0.0 <= entry["mean"] <= 1.0
            assert len(entry["values"]) == 1
    assert "trend" in format_classification_table(agg)
