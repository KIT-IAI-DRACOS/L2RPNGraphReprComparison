"""Compose the project's Hydra configs (configs/rllib/) in tests, as the experiment scripts would see them."""
from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "rllib")


def compose_config(config_name: str, overrides: list[str]) -> DictConfig:
    """The config experiments/train.py (`config`) or experiments/pretrain_hpo.py (`pretrain_hpo`) would see.

    Args:
        config_name: Top-level config file in configs/rllib/, without `.yaml`.
        overrides: Hydra command-line overrides.
    """
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=config_name, overrides=overrides)
