"""Mahalanobis anomaly scoring, principled because SIGReg pushes features to N(0, I)."""

import numpy as np


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
