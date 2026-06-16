"""Ground-truth tests for the SIGReg core.

The Epps-Pulley statistic compares the empirical characteristic function of the
projected embeddings to the standard normal characteristic function. We do not
standardize the samples, because SIGReg regularizes toward N(0, I): a shift or a
scale is a deviation we want to detect, not normalize away.
"""

import numpy as np
import pytest
import torch
from scipy import integrate

from chronojepa.sigreg import PooledSIGReg, epps_pulley_statistic, sliced_sigreg


def _reference_statistic(samples: np.ndarray, t_max: float) -> float:
    """Independent Epps-Pulley value via scipy adaptive quadrature over [0, t_max]."""

    def integrand(t: float) -> float:
        cos_mean = np.mean(np.cos(t * samples))
        sin_mean = np.mean(np.sin(t * samples))
        normal_cf = np.exp(-0.5 * t * t)
        return (cos_mean - normal_cf) ** 2 + sin_mean**2

    value, _ = integrate.quad(integrand, 0.0, t_max)
    return 2.0 * value


def test_epps_pulley_matches_scipy_quadrature() -> None:
    samples = np.array([-2.3, -1.1, -0.2, 0.4, 0.9, 1.7, 2.5, 3.1])
    t_max = 5.0
    reference = _reference_statistic(samples, t_max)

    got = epps_pulley_statistic(
        torch.tensor(samples, dtype=torch.float64), t_max=t_max, num_points=2048
    ).item()

    assert abs(got - reference) < 1e-3 * (abs(reference) + 1.0)


def test_univariate_statistic_detects_nongaussian() -> None:
    torch.manual_seed(0)
    n = 8192
    gaussian = torch.randn(n)
    uniform = (torch.rand(n) - 0.5) * (2.0 * 3.0**0.5)  # mean 0, var 1
    shifted = torch.randn(n) + 2.0
    scaled = torch.randn(n) * 2.0

    baseline = epps_pulley_statistic(gaussian).item()
    assert baseline < 2e-2
    assert baseline * 10 < epps_pulley_statistic(uniform).item()
    assert baseline * 10 < epps_pulley_statistic(shifted).item()
    assert baseline * 10 < epps_pulley_statistic(scaled).item()


def test_sliced_loss_near_zero_for_gaussian_and_larger_for_shift_scale() -> None:
    # Random projection gaussianizes a sum of independent coordinates, so shift and
    # scale (which survive projection) are the deviations the sliced loss must catch.
    torch.manual_seed(0)
    n, dim = 8192, 8

    gaussian = torch.randn(n, dim)
    shifted = torch.randn(n, dim) + 2.0
    scaled = torch.randn(n, dim) * 2.0

    def loss(x: torch.Tensor) -> float:
        g = torch.Generator().manual_seed(1)  # same directions for every input
        return sliced_sigreg(x, num_slices=32, generator=g).item()

    gaussian_loss = loss(gaussian)
    assert gaussian_loss < 2e-2
    assert gaussian_loss * 10 < loss(shifted)
    assert gaussian_loss * 10 < loss(scaled)


def test_gradients_flow_to_input() -> None:
    torch.manual_seed(0)
    x = torch.randn(256, 4, requires_grad=True)
    loss = PooledSIGReg(num_slices=16)(x)

    loss.backward()

    assert loss.ndim == 0
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_pooled_reduces_time_axis() -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(64, 20, 8)  # (batch, time, dim)
    loss = PooledSIGReg(num_slices=16)(embeddings)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device not available")
def test_cpu_and_mps_agree() -> None:
    torch.manual_seed(0)
    x = torch.randn(512, 8)
    directions = torch.randn(8, 32)
    directions = directions / directions.norm(dim=0, keepdim=True)

    cpu_loss = sliced_sigreg(x, directions=directions).item()
    mps_loss = sliced_sigreg(x.to("mps"), directions=directions.to("mps")).item()

    assert abs(cpu_loss - mps_loss) < 1e-3
