"""Window dataset and the DataLoader factory that wires splits, scaling, and views."""

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .augment import TwoViewAugmentation
from .windowing import StandardScaler, sliding_windows, time_split


class WindowDataset(Dataset):
    """Yields ``(view1, view2, clean)`` per window. Without augmentation the two views
    are the clean window itself, which is what probing and forecasting want."""

    def __init__(self, windows: np.ndarray, augment: TwoViewAugmentation | None = None) -> None:
        self.windows = torch.as_tensor(windows, dtype=torch.float32)
        self.augment = augment

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        clean = self.windows[index]
        if self.augment is None:
            return clean, clean, clean
        view1, view2 = self.augment(clean)
        return view1, view2, clean


def build_dataloaders(
    series: np.ndarray,
    *,
    window: int,
    stride: int,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    batch_size: int = 32,
    augment: TwoViewAugmentation | None = None,
    seed: int = 0,
) -> tuple[dict[str, DataLoader], StandardScaler, dict[str, tuple[int, int]]]:
    """Split by time, fit the scaler on train only, then build per-split loaders.

    Augmentation is applied to the train loader only; val and test return clean windows.
    """
    num_steps = series.shape[0]
    train, val, test = time_split(num_steps, train_frac, val_frac)
    scaler = StandardScaler().fit(series[train[0] : train[1]])
    normalized = scaler.transform(series)

    generator = torch.Generator().manual_seed(seed)
    loaders: dict[str, DataLoader] = {}
    splits = {"train": train, "val": val, "test": test}
    for name, (start, end) in splits.items():
        windows, _ = sliding_windows(normalized, start, end, window, stride)
        dataset = WindowDataset(windows, augment=augment if name == "train" else None)
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            generator=generator if name == "train" else None,
        )
    return loaders, scaler, splits
