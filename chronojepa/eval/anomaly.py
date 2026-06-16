"""Mahalanobis anomaly scoring, principled because SIGReg pushes features to N(0, I)."""

import numpy as np


def inject_anomaly(
    windows: np.ndarray,
    kind: str,
    rng: np.random.Generator,
    strength: float = 4.0,
    block: int | None = None,
) -> np.ndarray:
    """Return a copy of ``(N, C, T)`` windows with synthetic anomalies injected.

    ``spike`` adds transient bumps at a few timesteps, an amplitude anomaly any representation
    can see. ``shuffle`` permutes the whole time axis, which also reshuffles values across
    patches, so it changes content as well as order. ``block_shuffle`` permutes contiguous
    blocks while preserving each block's internal content, changing only the order of the
    blocks: a collapsed (constant-over-time) representation is blind to it, while one that
    encodes temporal structure can detect it.
    """
    out = windows.copy()
    num, _, length = out.shape
    if kind == "spike":
        count = max(1, length // 20)
        for i in range(num):
            positions = rng.choice(length, size=count, replace=False)
            out[i][:, positions] += strength
        return out
    if kind == "shuffle":
        for i in range(num):
            out[i] = out[i][:, rng.permutation(length)]
        return out
    if kind == "block_shuffle":
        block = block or max(2, length // 8)
        num_blocks = length // block
        for i in range(num):
            order = rng.permutation(num_blocks)
            reordered = np.concatenate(
                [out[i][:, b * block : (b + 1) * block] for b in order], axis=1
            )
            out[i][:, : num_blocks * block] = reordered
        return out
    raise ValueError(f"unknown anomaly kind {kind!r}")


class MahalanobisScorer:
    """Fit a Gaussian on training features, then score by Mahalanobis distance."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "MahalanobisScorer":
        self.mean_ = features.mean(axis=0)
        covariance = np.cov(features, rowvar=False)
        covariance = covariance + self.eps * np.eye(covariance.shape[0])
        self.precision_ = np.linalg.pinv(covariance)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.precision_ is None:
            raise RuntimeError("call fit before score")
        centered = features - self.mean_
        distances = np.einsum("ni,ij,nj->n", centered, self.precision_, centered)
        return np.sqrt(np.clip(distances, 0.0, None))
