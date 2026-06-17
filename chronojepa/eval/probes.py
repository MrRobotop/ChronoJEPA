"""Frozen-encoder feature extraction and linear, kNN, and forecasting probes."""

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    windows: Tensor,
    device: torch.device,
    batch_size: int = 256,
    pool: bool = True,
) -> np.ndarray:
    """Run a frozen encoder over ``(N, C, T)`` windows and return features.

    With ``pool`` the pooled embedding ``(N, D)`` is returned; otherwise the flattened token
    sequence ``(N, L * D)``, which keeps the temporal structure a collapsed encoder loses.
    The encoder is set to eval and wrapped in no_grad, so no gradients reach the backbone.
    """
    encoder = encoder.to(device).eval()
    outputs = []
    for start in range(0, windows.shape[0], batch_size):
        batch = windows[start : start + batch_size].to(device)
        tokens, pooled = encoder(batch)
        feature = pooled if pool else tokens.reshape(tokens.shape[0], -1)
        outputs.append(feature.cpu())
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


def mlp_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    hidden: tuple[int, ...] = (128,),
    max_iter: int = 300,
    seed: int = 0,
) -> float:
    """Accuracy of a nonlinear MLP probe on frozen features (standardized inputs).

    A nonlinear readout checks whether conclusions drawn from the linear probe survive when
    the probe can use nonlinear structure in the features.
    """
    scaler = StandardScaler().fit(x_train)
    model = MLPClassifier(hidden_layer_sizes=hidden, max_iter=max_iter, random_state=seed)
    model.fit(scaler.transform(x_train), y_train)
    return float(accuracy_score(y_test, model.predict(scaler.transform(x_test))))


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
