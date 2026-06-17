"""A position-free bag-of-patches encoder, for isolating the role of positional structure.

Non-overlapping patches embedded by a shared MLP, with no positional encoding and no
cross-patch attention. Each token depends only on its own patch, so the encoder is
permutation-equivariant over patches and the pooled (time-mean) feature is permutation
invariant: order is available only as position in the token sequence, never in the pooled
feature. This is the deliberate opposite of a positional transformer.
"""

from torch import Tensor, nn


class BagOfPatchesEncoder(nn.Module):
    """Embed each non-overlapping patch independently. Shares the ``(tokens, pooled)`` contract.

    ``tokens`` is ``(B, L, D)`` over patches and ``pooled`` is ``(B, D)``.
    """

    def __init__(self, num_channels: int, patch_len: int = 16, d_model: int = 64) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.embed = nn.Sequential(
            nn.Linear(patch_len, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.patch_len)
        tokens = self.embed(patches)  # (B, C, num_patches, d_model), each patch independent
        tokens = tokens.mean(dim=1)  # mean over channels -> (B, num_patches, d_model)
        pooled = tokens.mean(dim=1)  # mean over time -> (B, d_model)
        return tokens, pooled
