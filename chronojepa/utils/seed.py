"""Reproducibility: seed Python, numpy, and torch from a single entry point."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, numpy, and torch (including CUDA) so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
