"""Device selection: resolve the compute device once and reuse it across a run."""

import torch


def get_device() -> torch.device:
    """Resolve the compute device once, preferring CUDA, then Apple MPS, then CPU."""
    raise NotImplementedError
