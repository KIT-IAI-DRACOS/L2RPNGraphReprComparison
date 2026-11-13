"""
One imitation pretraining run from the Hydra config (`configs/rllib/config.yaml` with
`imitation=default`): build the train/val/test sets, the policy network and the optimisation
settings, train, and write the run directory.

A run directory contains:
    policy_weights.pt   state_dict of the RL_POLICY model -> `imitation.weights_path` for RL
    history.csv         per-epoch train loss and validation metrics, rewritten after every epoch
    val_predictions.npz per-state predictions of the best model on the val split (`imitation.pretrain.predict`)
                        with n_actions and the good-action tolerance they use
    test_metrics.json   metrics of the best model on the test split
    config.yaml         the resolved Hydra config (used by experiments/imitation_rollout.py)

Shared by experiments/pretrain_policy.py (one run), experiments/pretrain_hpo.py (one run per Optuna
trial) and experiments/imitation_rollout.py (rebuilds the network of a finished run).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from grid2op.gym_compat import GymEnv
from omegaconf import DictConfig, OmegaConf
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from core.rllib_config import build_model_config
from grid2op_env.observation_converter import make_observation_converter
from grid2op_env.utils import make_g2op_env
from imitation.dataset import ImitationData, load_split, split_indices
from imitation.pretrain import (
    GOOD_ACTION_TOLERANCE,
    EpochCallback,
    PretrainConfig,
    build_policy_model,
    evaluate,
    predict,
    train,
)
from imitation.torch_dataset import ImitationTorchDataset, load_or_convert

WEIGHTS_FILE = "policy_weights.pt"
HISTORY_FILE = "history.csv"
VAL_PREDICTIONS_FILE = "val_predictions.npz"
TEST_METRICS_FILE = "test_metrics.json"
CONFIG_FILE = "config.yaml"

logger = logging.getLogger(__name__)


@dataclass
class ImitationSets:
    """Train/val/test samples of one pretraining run.

    Attributes:
        train: Samples of the train chronics (optionally subsampled, see `imitation.max_train_samples`).
        val: Samples of the val chronics, for early stopping and model selection.
        test: Samples of the test chronics, for the final report only.
        n_actions: Policy actions incl. do-nothing.
    """

    train: ImitationTorchDataset
    val: ImitationTorchDataset
    test: ImitationTorchDataset
    n_actions: int


def check_pretraining_config(cfg: DictConfig, env_config: dict[str, Any]) -> None:
    """Raise SystemExit if the config cannot be used for pretraining.

    Args:
        cfg: Full Hydra config.
        env_config: Env config of the train split (`core.rllib_config.build_env_config`).
    """
    if "dataset" not in cfg.imitation:
        raise SystemExit("Select the pretraining settings with imitation=default")
    if env_config["use_forecasts"] and env_config["observation_space"] != "FlatObsSpace":
        raise SystemExit("Imitation data has no forecasts: set obs_space.use_forecasts=false")


def policy_model_config(cfg: DictConfig) -> dict:
    """RLlib model dict of the PPO RL_POLICY, as `_build_policies` in experiments/train.py sets it."""
    return build_model_config(cfg)


def converter_observation_space(env_config: dict[str, Any]) -> gym.spaces.Space:
    """The RL agent's observation space, i.e. the observation converter's."""
    g2op_env = make_g2op_env(env_config)
    try:
        return make_observation_converter(GymEnv(g2op_env, with_forecast=True), env_config, backend=g2op_env.backend).observation_space
    finally:
        g2op_env.close()


def build_model(cfg: DictConfig, obs_space: gym.spaces.Space, n_actions: int) -> TorchModelV2:
    """The RL_POLICY network described by `cfg.model`.

    Args:
        cfg: Full Hydra config.
        obs_space: Observation space, as from `converter_observation_space`.
        n_actions: Policy actions incl. do-nothing.

    Returns:
        The untrained torch model.
    """
    return build_policy_model(obs_space, n_actions, policy_model_config(cfg))


def pretrain_config(cfg: DictConfig) -> PretrainConfig:
    """Optimisation settings from `cfg.imitation` and the experiment seed."""
    im = cfg.imitation
    return PretrainConfig(epochs=im.epochs, batch_size=im.batch_size, lr=im.lr, weight_decay=im.weight_decay,
                          patience=im.patience, device=im.device, seed=cfg.experiment.seed,
                          monitor=im.get("monitor", "kl"))


def build_datasets(cfg: DictConfig, env_config: dict[str, Any]) -> ImitationSets:
    """Load (and convert on first use) the dataset and split it by chronic.

    Train samples come from `imitation.dataset`, val/test samples from `imitation.eval_dataset`
    (default: the same dataset). The split file assigns whole chronics, so datasets built from the
    same Tutor run get the same states in each split.

    Args:
        cfg: Full Hydra config.
        env_config: Env config of the train split; defines the observation converter.

    Returns:
        The three subsets.
    """
    im = cfg.imitation
    seed = cfg.experiment.seed
    split = load_split(Path(im.split))
    data = ImitationData.load(Path(im.dataset))
    eval_path = Path(im.eval_dataset or im.dataset)
    eval_data = data if eval_path == Path(im.dataset) else ImitationData.load(eval_path)
    if data.policy_target.shape[1] != eval_data.policy_target.shape[1]:
        raise SystemExit("dataset and eval_dataset have different action spaces")

    obs = load_or_convert(Path(im.dataset), data, env_config, Path(im.cache_dir), im.warmup_states, seed)
    eval_obs = obs if eval_data is data else load_or_convert(
        eval_path, eval_data, env_config, Path(im.cache_dir), im.warmup_states, seed)

    train_idx = split_indices(data.chronic, split, "train")
    if im.max_train_samples is not None and len(train_idx) > im.max_train_samples:
        rng = np.random.default_rng(seed)
        train_idx = np.sort(rng.choice(train_idx, size=im.max_train_samples, replace=False))
    sets = ImitationSets(
        train=ImitationTorchDataset(obs, data, train_idx),
        val=ImitationTorchDataset(eval_obs, eval_data, split_indices(eval_data.chronic, split, "val")),
        test=ImitationTorchDataset(eval_obs, eval_data, split_indices(eval_data.chronic, split, "test")),
        n_actions=data.policy_target.shape[1],
    )
    logger.info(f"Samples: train {len(sets.train)} ({im.dataset}), val {len(sets.val)}, test {len(sets.test)} ({eval_path})")
    return sets


def run_pretraining(
    cfg: DictConfig,
    sets: ImitationSets,
    obs_space: gym.spaces.Space,
    out_dir: Path,
    epoch_callback: EpochCallback | None = None,
) -> tuple[list[dict], dict[str, float]]:
    """Train the network of `cfg` on `sets` and write the run directory (see module docstring).

    The config is written first and the history after every epoch, so a run stopped by
    `epoch_callback` (e.g. a pruned Optuna trial) or the walltime still documents its settings and curves.

    Args:
        cfg: Full Hydra config; `cfg.model` and `cfg.imitation` define network and optimisation.
        sets: Train/val/test samples, as from `build_datasets`.
        obs_space: Observation space, as from `converter_observation_space`.
        out_dir: Run directory to write.
        epoch_callback: Passed to `imitation.pretrain.train`.

    Returns:
        `(history, test_metrics)`: per-epoch metrics and the best model's test metrics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / CONFIG_FILE, resolve=True)
    model = build_model(cfg, obs_space, sets.n_actions)
    config = pretrain_config(cfg)
    rows = []

    def write_history(epoch: int, row: dict[str, float]) -> None:
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / HISTORY_FILE, index=False)
        if epoch_callback is not None:
            epoch_callback(epoch, row)

    best_state, history = train(model, sets.train, sets.val, config, write_history)

    torch.save(best_state, out_dir / WEIGHTS_FILE)
    device = torch.device(config.device)
    np.savez_compressed(out_dir / VAL_PREDICTIONS_FILE, **predict(model, sets.val, config.batch_size, device),
                        n_actions=sets.n_actions, good_action_tolerance=GOOD_ACTION_TOLERANCE)
    test_metrics = evaluate(model, sets.test, config.batch_size, device)
    (out_dir / TEST_METRICS_FILE).write_text(json.dumps({"n_train": len(sets.train), **test_metrics}, indent=1))
    logger.info(f"Test: {test_metrics}. Wrote {out_dir / WEIGHTS_FILE}")
    return history, test_metrics
