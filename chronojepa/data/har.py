"""UCI HAR loader: human activity recognition from smartphone inertial sensors."""

from pathlib import Path

import numpy as np
import pandas as pd

_SIGNALS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)


def _load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    base = root / split / "Inertial Signals"
    channels = [
        pd.read_csv(base / f"{signal}_{split}.txt", sep=r"\s+", header=None).to_numpy()
        for signal in _SIGNALS
    ]
    x = np.stack(channels, axis=1).astype(np.float32)  # (n, 9, length)
    y = pd.read_csv(root / split / f"y_{split}.txt", header=None).to_numpy().ravel().astype(int)
    return x, y


def load_har(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load UCI HAR into ``(X_train, y_train, X_test, y_test)``.

    ``X`` is ``(n, 9, length)`` (nine inertial channels) and ``y`` is the activity label in
    1 to 6. ``root`` is the extracted ``UCI HAR Dataset`` directory; download it yourself.
    """
    root = Path(root)
    x_train, y_train = _load_split(root, "train")
    x_test, y_test = _load_split(root, "test")
    return x_train, y_train, x_test, y_test
