"""Config-driven experiment runner. All knobs come from the config, none from code."""

from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from chronojepa.data import TwoViewAugmentation, build_dataloaders, load_pems
from chronojepa.models import PatchTSTEncoder, TCNEncoder
from chronojepa.sigreg import make_sigreg
from chronojepa.utils.devices import get_device
from chronojepa.utils.logging import RunLogger, WandbLogger
from chronojepa.utils.seed import set_seed

from .loop import train


def _build_series(cfg: DictConfig) -> np.ndarray:
    if cfg.data.name == "synthetic":
        rng = np.random.default_rng(cfg.seed)
        t = np.linspace(0.0, 8.0, cfg.data.num_steps)[:, None]
        columns = [
            np.sin(2.0 * np.pi * (k + 2) * t) + 0.3 * rng.standard_normal((cfg.data.num_steps, 1))
            for k in range(cfg.data.channels)
        ]
        return np.concatenate(columns, axis=1).astype(np.float32)
    if cfg.data.name == "pems":
        return load_pems(cfg.data.path)
    raise ValueError(f"unknown data source {cfg.data.name!r}")


def _build_encoder(cfg: DictConfig, channels: int):
    if cfg.model.name == "patchtst":
        return PatchTSTEncoder(
            num_channels=channels,
            patch_len=cfg.model.patch_len,
            stride=cfg.model.stride,
            d_model=cfg.model.d_model,
            depth=cfg.model.depth,
            n_heads=cfg.model.n_heads,
        )
    if cfg.model.name == "tcn":
        return TCNEncoder(
            num_channels=channels,
            d_model=cfg.model.d_model,
            kernel_size=cfg.model.kernel_size,
            num_layers=cfg.model.num_layers,
        )
    raise ValueError(f"unknown model {cfg.model.name!r}")


def run_experiment(cfg: DictConfig, output_dir: str | Path | None = None) -> RunLogger:
    """Build everything from ``cfg``, train, and save the resolved config for the run."""
    set_seed(cfg.seed)
    device = get_device()
    series = _build_series(cfg)

    loaders, _, _ = build_dataloaders(
        series,
        window=cfg.window,
        stride=cfg.stride,
        batch_size=cfg.batch_size,
        augment=TwoViewAugmentation(**cfg.augment),
        seed=cfg.seed,
    )
    encoder = _build_encoder(cfg, series.shape[1])
    sigreg = make_sigreg(cfg.placement, num_slices=cfg.num_slices)

    if cfg.wandb.enabled:
        logger: RunLogger = WandbLogger(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        logger = RunLogger()

    print(f"run: placement={cfg.placement} model={cfg.model.name} seed={cfg.seed} device={device}")
    train(
        encoder,
        sigreg,
        loaders["train"],
        steps=cfg.steps,
        lr=cfg.optimizer.lr,
        lam=cfg.lam,
        warmup=cfg.warmup,
        weight_decay=cfg.optimizer.weight_decay,
        device=device,
        seed=cfg.seed,
        logger=logger,
    )

    out = Path(output_dir) if output_dir is not None else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "resolved_config.yaml")
    return logger
