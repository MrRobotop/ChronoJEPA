"""Frozen-encoder feature extraction and linear, kNN, and forecasting probes."""

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor, nn


@torch.no_grad()
def extract_features(
    encoder: nn.Module, windows: Tensor, device: torch.device, batch_size: int = 256
) -> np.ndarray:
    """Run a frozen encoder over ``(N, C, T)`` windows and return pooled features.

    The encoder is set to eval and wrapped in no_grad, so no gradients reach the backbone.
    """
    encoder = encoder.to(device).eval()
    outputs = []
    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start : start + batch_size].to(device)
        _, pooled = encoder(batch)
        outputs.append(pooled.cpu())
    return torch.cat(outputs, dim=0).numpy()


def linear_probe(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Accuracy of a logistic-regression probe on frozen features."""
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    return float(accuracy_score(y_test, model.predict(x_test)))


def knn_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_neighbors: int = 5,
) -> float:
    """Accuracy of a kNN probe on frozen features."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors).fit(x_train, y_train)
    return float(accuracy_score(y_test, model.predict(x_test)))


def forecast_linear_probe(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    """Fit a linear head on frozen features and report forecasting MAE and MSE."""
    model = LinearRegression().fit(x_train, y_train)
    prediction = model.predict(x_test)
    return {
        "mae": float(mean_absolute_error(y_test, prediction)),
        "mse": float(mean_squared_error(y_test, prediction)),
    }
