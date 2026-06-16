"""Random-slicing wrapper that lifts the univariate test to many dimensions."""

import torch
from torch import Generator, Tensor

from .univariate import epps_pulley_statistic


def random_directions(
    dim: int,
    num_slices: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    generator: Generator | None = None,
) -> Tensor:
    """Sample ``num_slices`` unit-norm Gaussian directions in ``dim`` dimensions."""
    weights = torch.randn(dim, num_slices, device=device, dtype=dtype, generator=generator)
    return weights / weights.norm(dim=0, keepdim=True)


def sliced_sigreg(
    x: Tensor,
    *,
    num_slices: int = 64,
    t_max: float = 5.0,
    num_points: int = 256,
    reduction: str = "mean",
    directions: Tensor | None = None,
    generator: Generator | None = None,
) -> Tensor:
    """Project ``(N, D)`` embeddings onto random unit directions, then aggregate the
    per-slice Epps-Pulley statistic with ``mean`` or ``sum``."""
    if directions is None:
        directions = random_directions(
            x.shape[1], num_slices, device=x.device, dtype=x.dtype, generator=generator
        )
    projections = x @ directions  # (N, K)
    stats = epps_pulley_statistic(projections, t_max=t_max, num_points=num_points)  # (K,)

    if reduction == "mean":
        return stats.mean()
    if reduction == "sum":
        return stats.sum()
    raise ValueError(f"unknown reduction: {reduction!r}")
