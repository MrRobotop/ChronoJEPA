"""Shape, contract, and device tests for the encoders and RevIN.

Shared contract: encoders map ``(B, C, T)`` to ``(tokens, pooled)`` where ``tokens`` is
``(B, L, D)`` indexed by time and ``pooled`` is ``(B, D)``.
"""

import pytest
import torch

from chronojepa.models import BagOfPatchesEncoder, PatchTSTEncoder, RevIN, TCNEncoder

B, C, T, D = 4, 3, 96, 32


def _patchtst() -> PatchTSTEncoder:
    return PatchTSTEncoder(num_channels=C, patch_len=16, stride=8, d_model=D, depth=2, n_heads=4)


def _tcn() -> TCNEncoder:
    return TCNEncoder(num_channels=C, d_model=D, kernel_size=3, num_layers=3)


def test_patchtst_output_contract() -> None:
    tokens, pooled = _patchtst()(torch.randn(B, C, T))
    assert tokens.ndim == 3
    assert tokens.shape[0] == B and tokens.shape[2] == D
    assert pooled.shape == (B, D)
    assert torch.isfinite(tokens).all() and torch.isfinite(pooled).all()


def test_tcn_output_contract() -> None:
    tokens, pooled = _tcn()(torch.randn(B, C, T))
    assert tokens.ndim == 3
    assert tokens.shape[0] == B and tokens.shape[2] == D
    assert pooled.shape == (B, D)
    assert torch.isfinite(tokens).all() and torch.isfinite(pooled).all()


def _permute_patches(x: torch.Tensor, perm: list[int], patch: int) -> torch.Tensor:
    num = x.shape[2] // patch
    return x.reshape(x.shape[0], x.shape[1], num, patch)[:, :, perm, :].reshape(x.shape)


def test_patchtst_without_pos_encoding_is_patch_permutation_equivariant() -> None:
    torch.manual_seed(0)
    encoder = PatchTSTEncoder(
        num_channels=1, patch_len=8, stride=8, d_model=16, depth=2, n_heads=4, pos_encoding=False
    ).eval()
    x = torch.randn(1, 1, 32)
    perm = [3, 1, 2, 0]
    with torch.no_grad():
        tokens, _ = encoder(x)
        permuted, _ = encoder(_permute_patches(x, perm, patch=8))
    # No positional encoding: permuting input patches permutes the output tokens identically.
    assert torch.allclose(tokens[:, perm, :], permuted, atol=1e-5)


def test_patchtst_with_pos_encoding_breaks_permutation_equivariance() -> None:
    torch.manual_seed(0)
    encoder = PatchTSTEncoder(
        num_channels=1, patch_len=8, stride=8, d_model=16, depth=2, n_heads=4, pos_encoding=True
    ).eval()
    x = torch.randn(1, 1, 32)
    perm = [3, 1, 2, 0]
    with torch.no_grad():
        tokens, _ = encoder(x)
        permuted, _ = encoder(_permute_patches(x, perm, patch=8))
    assert not torch.allclose(tokens[:, perm, :], permuted, atol=1e-5)


def test_bagofpatches_pooled_feature_is_permutation_invariant() -> None:
    torch.manual_seed(0)
    encoder = BagOfPatchesEncoder(num_channels=1, patch_len=8, d_model=16).eval()
    x = torch.randn(1, 1, 32)
    perm = [3, 1, 2, 0]
    with torch.no_grad():
        tokens, pooled = encoder(x)
        permuted_tokens, permuted_pooled = encoder(_permute_patches(x, perm, patch=8))
    # Position-free: tokens permute with the patches, and the pooled mean is invariant.
    assert torch.allclose(tokens[:, perm, :], permuted_tokens, atol=1e-5)
    assert torch.allclose(pooled, permuted_pooled, atol=1e-5)


def test_bagofpatches_output_contract() -> None:
    tokens, pooled = BagOfPatchesEncoder(num_channels=C, patch_len=16, d_model=D)(
        torch.randn(B, C, T)
    )
    assert tokens.ndim == 3 and tokens.shape[0] == B and tokens.shape[2] == D
    assert pooled.shape == (B, D)


def test_pooled_is_mean_over_time_tokens() -> None:
    tokens, pooled = _tcn()(torch.randn(B, C, T))
    assert torch.allclose(pooled, tokens.mean(dim=1), atol=1e-5)


def test_revin_roundtrip_is_identity() -> None:
    revin = RevIN(num_channels=C)
    x = torch.randn(B, C, T) * 5.0 + 2.0
    recon = revin.denormalize(revin.normalize(x))
    assert torch.allclose(recon, x, atol=1e-4)


def test_revin_normalizes_per_channel() -> None:
    revin = RevIN(num_channels=C, affine=False)
    x = torch.randn(B, C, T) * 5.0 + 2.0
    norm = revin.normalize(x)
    assert norm.mean(dim=-1).abs().max() < 1e-4
    assert (norm.std(dim=-1, unbiased=False) - 1.0).abs().max() < 1e-3


@pytest.mark.parametrize("encoder_factory", [_patchtst, _tcn])
def test_forward_and_backward_cpu(encoder_factory) -> None:
    encoder = encoder_factory()
    x = torch.randn(B, C, T, requires_grad=True)
    _, pooled = encoder(x)
    pooled.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device not available")
@pytest.mark.parametrize("encoder_factory", [_patchtst, _tcn])
def test_forward_and_backward_mps(encoder_factory) -> None:
    encoder = encoder_factory().to("mps")
    x = torch.randn(B, C, T, device="mps", requires_grad=True)
    tokens, pooled = encoder(x)
    pooled.sum().backward()
    assert tokens.shape[2] == D
    assert x.grad is not None and torch.isfinite(x.grad).all()
