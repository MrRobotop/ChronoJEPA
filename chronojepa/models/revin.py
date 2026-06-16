"""RevIN: reversible instance normalization for forecasting under distribution shift."""

import torch
from torch import Tensor, nn


class RevIN(nn.Module):
    """Per-instance, per-channel normalization that can be exactly reversed.

    ``normalize`` stores the statistics of the current batch so ``denormalize`` can undo
    the transform later, for example to map a forecast back to the original scale. Inputs
    are ``(B, C, T)`` and statistics are computed over the time axis.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        self._mean: Tensor | None = None
        self._std: Tensor | None = None

    def normalize(self, x: Tensor) -> Tensor:
        self._mean = x.mean(dim=-1, keepdim=True).detach()
        var = x.var(dim=-1, keepdim=True, unbiased=False).detach()
        self._std = torch.sqrt(var + self.eps)
        out = (x - self._mean) / self._std
        if self.affine:
            out = out * self.weight[None, :, None] + self.bias[None, :, None]
        return out

    def denormalize(self, x: Tensor) -> Tensor:
        if self._mean is None or self._std is None:
            raise RuntimeError("call normalize before denormalize")
        out = x
        if self.affine:
            out = (out - self.bias[None, :, None]) / (self.weight[None, :, None] + self.eps)
        return out * self._std + self._mean
