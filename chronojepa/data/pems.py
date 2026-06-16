"""PEMS loader. The dataset is downloaded by the user; this only reads it from disk."""

from pathlib import Path

import numpy as np


def load_pems(path: str | Path, key: str = "data") -> np.ndarray:
    """Load a PEMS ``.npz`` file into a ``(time, channels)`` float32 array.

    PEMS ships as ``(time, nodes, features)``. We keep the first feature per node, which
    yields one channel per sensor. Download the dataset yourself and pass the path; this
    function never fetches anything, to keep data provenance explicit.
    """
    array = np.load(Path(path))[key]
    if array.ndim == 3:
        array = array[:, :, 0]
    return np.ascontiguousarray(array, dtype=np.float32)
