"""Time-ordered splits, sliding windows, and train-only normalization."""

import numpy as np


def time_split(
    num_steps: int, train_frac: float = 0.7, val_frac: float = 0.1
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Split a length into contiguous, time-ordered (train, val, test) index ranges."""
    # Add integer lengths rather than float fractions: 0.7 + 0.1 is 0.79999... in
    # floating point, which would round the validation boundary down by one step.
    train_end = int(num_steps * train_frac)
    val_end = train_end + int(num_steps * val_frac)
    return (0, train_end), (train_end, val_end), (val_end, num_steps)


def sliding_windows(
    series: np.ndarray, start: int, end: int, window: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(num, channels, window)`` windows from ``series[start:end]``.

    Windows never cross ``[start, end)``, so windows built per split stay within that
    split. Also returns each window's start index for time-overlap checks.
    """
    starts = np.arange(start, end - window + 1, stride)
    windows = (
        np.stack([series[s : s + window].T for s in starts])
        if starts.size
        else np.empty((0, series.shape[1], window), dtype=series.dtype)
    )
    return windows, starts


class StandardScaler:
    """Per-channel standardization fit on training data only, applied forward."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0) + self.eps
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("call fit before transform")
        return (x - self.mean_) / self.std_
