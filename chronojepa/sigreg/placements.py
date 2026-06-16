"""SIGReg placement variants and the placement factory."""

from torch import Generator, Tensor, nn

from .slicing import random_directions, sliced_sigreg
from .univariate import epps_pulley_statistic


class PooledSIGReg(nn.Module):
    """SIGReg over the time-pooled sequence embedding.

    Pools ``(N, L, D)`` to ``(N, D)`` by averaging across time, then applies the sliced
    Epps-Pulley objective. This is the baseline placement expected to collapse the time
    axis to a constant per-sequence vector.
    """

    def __init__(
        self,
        num_slices: int = 64,
        t_max: float = 5.0,
        num_points: int = 256,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.t_max = t_max
        self.num_points = num_points
        self.reduction = reduction

    def forward(
        self,
        embeddings: Tensor,
        *,
        directions: Tensor | None = None,
        generator: Generator | None = None,
    ) -> Tensor:
        if embeddings.dim() == 3:
            pooled = embeddings.mean(dim=1)
        elif embeddings.dim() == 2:
            pooled = embeddings
        else:
            raise ValueError(
                f"expected (N, L, D) or (N, D) embeddings, got shape {tuple(embeddings.shape)}"
            )
        return sliced_sigreg(
            pooled,
            num_slices=self.num_slices,
            t_max=self.t_max,
            num_points=self.num_points,
            reduction=self.reduction,
            directions=directions,
            generator=generator,
        )


class DualSIGReg(nn.Module):
    """SIGReg within each sequence across time, plus across samples in the batch.

    The within-sequence term asks the L per-timestep embeddings of each sequence to look
    like N(0, I); a sequence that collapses to a constant vector along time has near-zero
    variance there and is penalized heavily, which is the mechanism that fights the
    time-axis collapse. The between-sample term shapes the pooled batch distribution.
    """

    def __init__(
        self,
        num_slices: int = 64,
        t_max: float = 5.0,
        num_points: int = 256,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.t_max = t_max
        self.num_points = num_points
        self.reduction = reduction

    def forward(
        self,
        embeddings: Tensor,
        *,
        directions: Tensor | None = None,
        generator: Generator | None = None,
    ) -> Tensor:
        if embeddings.dim() != 3:
            raise ValueError(
                f"DualSIGReg expects (N, L, D) embeddings, got shape {tuple(embeddings.shape)}"
            )
        batch, length, dim = embeddings.shape
        if directions is None:
            directions = random_directions(
                dim,
                self.num_slices,
                device=embeddings.device,
                dtype=embeddings.dtype,
                generator=generator,
            )
        projections = embeddings @ directions  # (N, L, K)

        # Within-sequence: each (sample, slice) column uses its L timesteps as samples.
        within_columns = projections.permute(1, 0, 2).reshape(length, batch * self.num_slices)
        within = epps_pulley_statistic(within_columns, self.t_max, self.num_points).mean()

        # Between-sample: pool over time, then SIGReg across the batch.
        pooled_projections = projections.mean(dim=1)  # (N, K)
        between = epps_pulley_statistic(pooled_projections, self.t_max, self.num_points)
        between = between.sum() if self.reduction == "sum" else between.mean()

        return within + between


class StructuredSIGReg(nn.Module):
    """Joint placement: regularize all per-timestep embeddings ``(N * L, D)`` to N(0, I).

    An initial concrete instantiation of the open structured formulation. It treats every
    timestep of every sample as one shared set of D-dimensional points.
    """

    def __init__(
        self,
        num_slices: int = 64,
        t_max: float = 5.0,
        num_points: int = 256,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.t_max = t_max
        self.num_points = num_points
        self.reduction = reduction

    def forward(
        self,
        embeddings: Tensor,
        *,
        directions: Tensor | None = None,
        generator: Generator | None = None,
    ) -> Tensor:
        if embeddings.dim() != 3:
            raise ValueError(
                f"StructuredSIGReg expects (N, L, D), got shape {tuple(embeddings.shape)}"
            )
        joint = embeddings.reshape(-1, embeddings.shape[-1])  # (N * L, D)
        return sliced_sigreg(
            joint,
            num_slices=self.num_slices,
            t_max=self.t_max,
            num_points=self.num_points,
            reduction=self.reduction,
            directions=directions,
            generator=generator,
        )


_PLACEMENTS = {
    "pooled": PooledSIGReg,
    "dual": DualSIGReg,
    "structured": StructuredSIGReg,
}


def make_sigreg(name: str, **kwargs) -> nn.Module:
    """Build a SIGReg placement by config name: ``pooled``, ``dual``, or ``structured``."""
    if name not in _PLACEMENTS:
        raise ValueError(f"unknown placement {name!r}; choose from {sorted(_PLACEMENTS)}")
    return _PLACEMENTS[name](**kwargs)
