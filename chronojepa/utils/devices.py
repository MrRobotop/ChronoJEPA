"""Device selection: resolve the compute device once and reuse it across a run."""

import torch


def get_device() -> torch.device:
    """Resolve the compute device once, preferring CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
