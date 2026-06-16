"""Single config-driven entry point. Run with, for example:

    uv run python scripts/train.py +experiment=smoke
    uv run python scripts/train.py +experiment=pems_dual data.path=/path/to/pems.npz

Sweep over the placements and lambda with Hydra multirun:

    uv run python scripts/train.py -m placement=pooled,dual,structured lam=0.1,0.5,0.9
"""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from chronojepa.train import run_experiment


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # hydra.job.chdir defaults to False, so resolve the per-run output dir explicitly
    # instead of relying on the current working directory.
    run_experiment(cfg, output_dir=HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
