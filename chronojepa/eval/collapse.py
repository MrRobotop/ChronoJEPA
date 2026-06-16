"""Collapse diagnostics: across-time variance and effective rank."""

import torch
from torch import Tensor


def across_time_variance(tokens: Tensor) -> float:
    """Mean variance of the embedding along the time axis of ``(B, L, D)`` tokens.

    Near zero means each sequence is nearly constant over time, the signature of the
    time-axis collapse.
    """
    return tokens.var(dim=1, unbiased=False).mean().item()


def effective_rank(matrix: Tensor) -> float:
    """Effective rank ``exp(entropy(normalized singular values))`` of ``(N, D)``.

    Drops toward 1 when the representation occupies a degenerate subspace.
    """
    singular = torch.linalg.svdvals(matrix)
    total = singular.sum()
    if total <= 0:
        return 0.0
    probs = singular / total
    entropy = -(probs * torch.log(probs + 1e-12)).sum()
    return torch.exp(entropy).item()


def collapse_report(tokens: Tensor) -> dict[str, float]:
    """Diagnostic report for ``(B, L, D)`` tokens: across-time variance and effective rank."""
    pooled = tokens.mean(dim=1)  # (B, D)
    return {
        "across_time_variance": across_time_variance(tokens),
        "effective_rank": effective_rank(pooled),
    }
