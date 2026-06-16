"""SIGReg placement variants. Phase 1 ships the pooled baseline."""

from torch import Generator, Tensor, nn

from .slicing import sliced_sigreg


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
