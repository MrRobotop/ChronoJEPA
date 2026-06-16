"""PatchTST-style transformer encoder with channel independence."""

import math

import torch
from torch import Tensor, nn


def _sinusoidal_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Standard fixed sinusoidal positional encoding of shape ``(length, dim)``."""
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    scale = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * scale)
    encoding[:, 1::2] = torch.cos(position * scale)
    return encoding


class PatchTSTEncoder(nn.Module):
    """Patch the time axis, embed each channel independently with a shared transformer,
    then mean-pool channels to a time-indexed sequence of embeddings.

    Returns ``(tokens, pooled)`` with ``tokens`` of shape ``(B, L, D)`` over time patches
    and ``pooled`` of shape ``(B, D)``.
    """

    def __init__(
        self,
        num_channels: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        depth: int = 3,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.patch_embed = nn.Linear(patch_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, _ = x.shape
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        num_patches = patches.shape[2]
        tokens = self.patch_embed(patches)  # (B, C, num_patches, d_model)
        tokens = tokens + _sinusoidal_encoding(num_patches, self.d_model, x.device, x.dtype)
        tokens = tokens.reshape(batch * channels, num_patches, self.d_model)
        tokens = self.transformer(tokens)
        tokens = tokens.reshape(batch, channels, num_patches, self.d_model)
        tokens = tokens.mean(dim=1)  # mean over channels -> (B, num_patches, d_model)
        pooled = tokens.mean(dim=1)  # mean over time -> (B, d_model)
        return tokens, pooled
