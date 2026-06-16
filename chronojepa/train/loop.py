"""A minimal, device-agnostic, seeded training loop for the LeJEPA objective."""

import math
from itertools import cycle

import torch
from torch import nn
from torch.utils.data import DataLoader

from chronojepa.utils.devices import get_device
from chronojepa.utils.logging import RunLogger
from chronojepa.utils.seed import set_seed

from .objective import invariance_loss


def _cosine_warmup(step: int, total: int, warmup: int) -> float:
    """Linear warmup to 1.0, then cosine decay to 0.0 over the remaining steps."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train(
    encoder: nn.Module,
    sigreg: nn.Module,
    dataloader: DataLoader,
    *,
    steps: int,
    lr: float = 1e-3,
    lam: float = 0.5,
    warmup: int = 10,
    weight_decay: float = 0.0,
    predictor: nn.Module | None = None,
    device: torch.device | None = None,
    seed: int = 0,
    logger: RunLogger | None = None,
) -> RunLogger:
    """Train ``encoder`` (and optional ``predictor``) on two augmented views.

    The objective is ``lam * sigreg + (1 - lam) * invariance``, a single tradeoff knob.
    bf16 autocast is used on CUDA where supported; CPU and MPS run in float32.
    """
    set_seed(seed)
    device = device or get_device()
    logger = logger or RunLogger()

    encoder = encoder.to(device)
    parameters = list(encoder.parameters())
    if predictor is not None:
        predictor = predictor.to(device)
        parameters += list(predictor.parameters())

    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _cosine_warmup(step, steps, warmup)
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    encoder.train()
    if predictor is not None:
        predictor.train()

    batches = cycle(dataloader)
    for step in range(steps):
        view1, view2, _ = next(batches)
        view1 = view1.to(device)
        view2 = view2.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            tokens1, pooled1 = encoder(view1)
            tokens2, pooled2 = encoder(view2)
            sigreg_term = 0.5 * (sigreg(tokens1) + sigreg(tokens2))
            invariance_term = invariance_loss(pooled1, pooled2, predictor)
            loss = lam * sigreg_term + (1.0 - lam) * invariance_term

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        logger.log(
            {
                "loss": loss.item(),
                "sigreg": sigreg_term.item(),
                "invariance": invariance_term.item(),
                "lr": scheduler.get_last_lr()[0],
            },
            step=step,
        )

    logger.finish()
    return logger
