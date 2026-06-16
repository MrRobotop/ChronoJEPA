"""Optional MLP predictor for the predictive variant of the invariance term."""

from torch import Tensor, nn


class MLPPredictor(nn.Module):
    """Two-layer MLP mapping a pooled embedding to the prediction space."""

    def __init__(self, dim: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or 2 * dim
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
