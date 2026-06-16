"""A small dilated TCN encoder used as a baseline, sharing the encoder contract."""

from torch import Tensor, nn


class _TCNBlock(nn.Module):
    """Dilated causal convolution with a residual connection."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation
        )
        self.act = nn.ReLU()
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)[..., : x.shape[-1]]  # causal crop of right padding
        out = self.act(out)
        residual = x if self.downsample is None else self.downsample(x)
        return out + residual


class TCNEncoder(nn.Module):
    """Stacked dilated causal convolutions over time.

    Returns ``(tokens, pooled)`` with ``tokens`` of shape ``(B, L, D)`` over time steps
    and ``pooled`` of shape ``(B, D)``.
    """

    def __init__(
        self,
        num_channels: int,
        d_model: int = 64,
        kernel_size: int = 3,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = num_channels
        for layer in range(num_layers):
            blocks.append(_TCNBlock(in_channels, d_model, kernel_size, dilation=2**layer))
            in_channels = d_model
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.blocks(x)  # (B, d_model, T)
        tokens = hidden.transpose(1, 2)  # (B, T, d_model)
        pooled = tokens.mean(dim=1)
        return tokens, pooled
