"""The combined LeJEPA objective: SIGReg plus a single-lambda invariance term."""

import torch.nn.functional as F
from torch import Tensor, nn


def invariance_loss(z1: Tensor, z2: Tensor, predictor: nn.Module | None = None) -> Tensor:
    """Pull the two views together. With a predictor this becomes the predictive variant.

    There is no stop-gradient, no teacher, and no EMA, matching the LeJEPA philosophy.
    """
    if predictor is None:
        return F.mse_loss(z1, z2)
    return 0.5 * (F.mse_loss(predictor(z1), z2) + F.mse_loss(predictor(z2), z1))
