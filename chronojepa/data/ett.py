"""ETT loader. The Electricity Transformer Temperature CSV, downloaded by the user."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_ett(path: str | Path) -> np.ndarray:
    """Load an ETT ``.csv`` into a ``(time, channels)`` float32 array.

    ETT ships as a date column followed by seven sensor channels (six load features and the
    oil temperature OT). The date column is dropped. Download the file yourself and pass the
    path; this function never fetches anything, to keep data provenance explicit.
    """
    frame = pd.read_csv(Path(path))
    values = frame.drop(columns=["date"]).to_numpy()
    return np.ascontiguousarray(values, dtype=np.float32)
