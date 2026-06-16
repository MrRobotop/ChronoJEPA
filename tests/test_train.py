"""Tests for the SIGReg placements, the placement factory, and the training loop."""

import numpy as np
import pytest
import torch

from chronojepa.data import TwoViewAugmentation, build_dataloaders
from chronojepa.models import MLPPredictor, PatchTSTEncoder
from chronojepa.sigreg import (
    DualSIGReg,
    PooledSIGReg,
    StructuredSIGReg,
    make_sigreg,
)
from chronojepa.train import train

PLACEMENTS = ["pooled", "dual", "structured"]


@pytest.mark.parametrize("name", PLACEMENTS)
def test_placement_returns_scalar_and_backprops(name: str) -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(8, 12, 16, requires_grad=True)
    loss = make_sigreg(name, num_slices=16)(embeddings)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0


def test_factory_switches_placement_by_name() -> None:
    assert isinstance(make_sigreg("pooled"), PooledSIGReg)
    assert isinstance(make_sigreg("dual"), DualSIGReg)
    assert isinstance(make_sigreg("structured"), StructuredSIGReg)
    with pytest.raises(ValueError):
        make_sigreg("does-not-exist")


def _series(num_steps: int = 400, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, num_steps)[:, None]
    return (np.sin(2.0 * np.pi * 4.0 * t) + rng.standard_normal((num_steps, channels))).astype(
        np.float32
    )


@pytest.mark.parametrize("name", PLACEMENTS)
def test_smoke_training_run_logs_finite_losses(name: str) -> None:
    torch.manual_seed(0)
    loaders, _, _ = build_dataloaders(
        _series(),
        window=48,
        stride=12,
        batch_size=16,
        augment=TwoViewAugmentation(jitter_sigma=0.1, scaling_sigma=0.1, mask_ratio=0.1),
        seed=0,
    )
    encoder = PatchTSTEncoder(
        num_channels=3, patch_len=16, stride=8, d_model=32, depth=2, n_heads=4
    )
    sigreg = make_sigreg(name, num_slices=16)

    logger = train(
        encoder,
        sigreg,
        loaders["train"],
        steps=50,
        lr=1e-3,
        lam=0.5,
        warmup=5,
        device=torch.device("cpu"),
        seed=0,
    )

    assert len(logger.history) == 50
    last = logger.history[-1]
    for key in ("loss", "sigreg", "invariance", "lr"):
        assert np.isfinite(last[key])


def test_smoke_training_run_with_predictor() -> None:
    torch.manual_seed(0)
    loaders, _, _ = build_dataloaders(
        _series(), window=48, stride=12, batch_size=16, augment=TwoViewAugmentation(), seed=0
    )
    encoder = PatchTSTEncoder(
        num_channels=3, patch_len=16, stride=8, d_model=32, depth=2, n_heads=4
    )
    predictor = MLPPredictor(dim=32)

    logger = train(
        encoder,
        make_sigreg("dual", num_slices=16),
        loaders["train"],
        steps=10,
        predictor=predictor,
        device=torch.device("cpu"),
        seed=0,
    )

    assert len(logger.history) == 10
    assert np.isfinite(logger.history[-1]["loss"])
