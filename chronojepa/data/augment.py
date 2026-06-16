"""Config-driven two-view augmentation for self-supervised time series."""

import torch
from torch import Tensor


class TwoViewAugmentation:
    """Produce two augmented views of a ``(C, T)`` window.

    Strengths are configured, not hardcoded. Crop length is shared across views so the
    two views always have matching shapes, which the contrastive objective relies on.
    """

    def __init__(
        self,
        jitter_sigma: float = 0.1,
        scaling_sigma: float = 0.1,
        mask_ratio: float = 0.1,
        crop_ratio: float = 1.0,
    ) -> None:
        self.jitter_sigma = jitter_sigma
        self.scaling_sigma = scaling_sigma
        self.mask_ratio = mask_ratio
        self.crop_ratio = crop_ratio

    def _view(self, window: Tensor, crop_len: int, crop_start: int) -> Tensor:
        out = window[:, crop_start : crop_start + crop_len]
        if self.scaling_sigma > 0:
            out = out * (1.0 + torch.randn(out.shape[0], 1) * self.scaling_sigma)
        if self.jitter_sigma > 0:
            out = out + torch.randn_like(out) * self.jitter_sigma
        if self.mask_ratio > 0:
            keep = (torch.rand(out.shape[1]) >= self.mask_ratio).to(out.dtype)
            out = out * keep
        return out

    def __call__(self, window: Tensor) -> tuple[Tensor, Tensor]:
        length = window.shape[1]
        crop_len = max(1, int(length * self.crop_ratio))
        high = length - crop_len + 1
        start1 = int(torch.randint(0, high, ()).item())
        start2 = int(torch.randint(0, high, ()).item())
        return self._view(window, crop_len, start1), self._view(window, crop_len, start2)
