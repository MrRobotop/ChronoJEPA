"""Tests that the Hydra configs compose and that a smoke experiment runs end to end."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from chronojepa.train import run_experiment

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


@pytest.mark.parametrize(
    ("experiment", "placement", "data_name"),
    [
        ("smoke", "dual", "synthetic"),
        ("pems_pooled", "pooled", "pems"),
        ("pems_dual", "dual", "pems"),
        ("pems_structured", "structured", "pems"),
    ],
)
def test_named_experiments_compose(experiment: str, placement: str, data_name: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=[f"+experiment={experiment}"])
    assert cfg.placement == placement
    assert cfg.data.name == data_name


def test_smoke_experiment_runs_and_saves_resolved_config(tmp_path) -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=smoke", "steps=5", "batch_size=8"],
        )
    logger = run_experiment(cfg, output_dir=tmp_path)
    assert len(logger.history) == 5
    assert (tmp_path / "resolved_config.yaml").exists()
