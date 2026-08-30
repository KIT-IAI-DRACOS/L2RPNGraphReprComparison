"""
Trains an RL agent using Hydra for config composition.

Usage examples
--------------
# Default GNN/PPO run:
  python train.py

# Quick test run (minimal timesteps, local Ray mode):
  python train.py experiment=test_minimal training=ppo_test

# MLP baseline:
  python train.py model=mlp obs_space=flat

# Optuna hyperparameter search:
  python train.py optimization=optuna

# Single-key override via CLI:
  python train.py experiment.nb_timesteps=500000 training.lr=3e-4
"""

import logging
import os
from typing import Any

import grid2op
import hydra
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.algorithms.registry import POLICIES
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

from core.constants import DO_NOTHING_POLICY, RL_POLICY, HIGH_LEVEL_POLICY, set_seed
from core.train import run_training
from grid2op_env import policy_mapping_fn
from grid2op_env.env import CustomizedGrid2OpEnvironment
from grid2op_env.multi_agent_policies.do_nothing_policy import DoNothingPolicy
from grid2op_env.multi_agent_policies.select_agent_policy import SelectAgentPolicy
from rl.gnn_ppo_model import GNNModel

logger = logging.getLogger(__name__)

ModelCatalog.register_custom_model("gnn_model", GNNModel)

POLICIES[DO_NOTHING_POLICY] = DoNothingPolicy
POLICIES[HIGH_LEVEL_POLICY] = SelectAgentPolicy

_ALGORITHM_CONFIG_CLS = {
    "ppo": ppo.PPOConfig,
}


# ---------------------------------------------------------------------------
# Config → RLLib translation
# ---------------------------------------------------------------------------

def _build_env_config(cfg: DictConfig, split: str) -> dict[str, Any]:
    """Build an env_config dict for the given split ('train' or 'val')."""
    env = cfg.env
    obs = cfg.obs_space
    reward_instance = get_class(cfg.reward._target_)()

    env_config = {
        "env_name": f"{env.env_name}_{split}",
        "action_space": env.action_space,
        "observation_space": obs.observation_space,
        "g2op_input": obs.get("g2op_input", []),
        "custom_input": obs.get("custom_input", []),
        "lib_dir": os.getcwd(),
        "grid2op_kwargs": {"reward_class": reward_instance},
        "seed": cfg.experiment.seed,
        "rho_threshold": env.rho_threshold,
        "reset_topo": env.reset_topo,
        "line_reco": env.line_reco,
        "line_disc": env.line_disc,
        "penalty_game_over": env.penalty_game_over,
        "reward_finish": env.reward_finish,
        "curriculum_training": env.curriculum_training,
        "curriculum_thresholds": list(env.curriculum_thresholds),
        "chronics_dir": env.get("chronics_dir", None),
        "use_chronics_cache": env.get("use_chronics_cache", False),
    }

    if cfg.opponent.enabled:
        opp = cfg.opponent
        env_config["grid2op_kwargs"].update(
            {
                "opponent_attack_cooldown": opp.opponent_attack_cooldown,
                "opponent_attack_duration": opp.opponent_attack_duration,
                "opponent_budget_per_ts": opp.opponent_budget_per_ts,
                "opponent_init_budget": opp.opponent_init_budget,
                "opponent_action_class": get_class(opp.opponent_action_class),
                "opponent_class": get_class(opp.opponent_class),
                "opponent_budget_class": get_class(opp.opponent_budget_class),
                "kwargs_opponent": OmegaConf.to_container(opp.kwargs_opponent, resolve=True),
            }
        )

    if split == "val":
        overrides = OmegaConf.to_container(cfg.evaluation.eval_env_overrides, resolve=True)
        env_config.update(overrides)

    return env_config


def _build_model_config(cfg: DictConfig) -> dict[str, Any]:
    """Build the RLLib model dict from cfg.model."""
    model = cfg.model
    custom_model_config: dict[str, Any] = {}

    if model.encoder is not None:
        custom_model_config["encoder"] = OmegaConf.to_container(model.encoder, resolve=True)

    if model.gnn is not None:
        custom_model_config["gnn"] = OmegaConf.to_container(model.gnn, resolve=True)

    return {
        "fcnet_hiddens": list(model.fcnet_hiddens),
        "fcnet_activation": model.fcnet_activation,
        "post_fcnet_hiddens": list(model.post_fcnet_hiddens),
        "custom_model": model.custom_model if model.custom_model != "null" else None,
        "custom_model_config": custom_model_config,
    }


def _build_policies(cfg: DictConfig, algorithm: str) -> dict:
    """Build the multi-agent policies dict."""
    custom_model = cfg.model.custom_model
    model_override = {}

    policy_class = PPOTorchPolicy
    if custom_model == "gnn_model":
        model_override = {"model": {"custom_model": "gnn_model"}}
    elif custom_model is None or custom_model == "mlp_model":
        model_override = {}
    else:
        raise ValueError(f"Unsupported algorithm-model combination '{algorithm}'+'{custom_model}")

    logger.info(f"Using algorithm {policy_class}")
    logger.info(f"Using model {model_override.get('model', {}).get('custom_model', {}) or 'default model'}")

    return {
        HIGH_LEVEL_POLICY: PolicySpec(
            policy_class=SelectAgentPolicy,
            config=(
                AlgorithmConfig()
                .training(model={"custom_model_config": {"rho_threshold": cfg.env.rho_threshold}})
                .rollouts(preprocessor_pref=None)
            ),
        ),
        RL_POLICY: PolicySpec(
            policy_class=policy_class,
            config=model_override,
        ),
        DO_NOTHING_POLICY: PolicySpec(
            policy_class=DoNothingPolicy,
            config=AlgorithmConfig(),
        ),
    }


def build_rllib_config(cfg: DictConfig) -> dict[str, Any]:
    """Translate the assembled Hydra config into a flat RLLib algorithm config dict."""
    # Extract algorithm name before merging (it is a meta-key, not an RLLib param).
    training = OmegaConf.to_container(cfg.training, resolve=True)
    algorithm = training.pop("algorithm", "ppo")

    algo_config_cls = _ALGORITHM_CONFIG_CLS.get(algorithm)
    if algo_config_cls is None:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose from: {list(_ALGORITHM_CONFIG_CLS)}")

    rllib_cfg = algo_config_cls().to_dict()
    rllib_cfg["_disable_preprocessor_api"] = True
    rllib_cfg["seed"] = cfg.experiment.seed

    # --- Training hyperparameters ---
    rllib_cfg.update(training)

    # --- Model ---
    rllib_cfg["model"] = _build_model_config(cfg)
    rllib_cfg["_enable_learner_api"] = cfg.model._enable_learner_api
    rllib_cfg["_enable_rl_module_api"] = False

    # --- Environment ---
    train_env_config = _build_env_config(cfg, "train")
    val_env_config = _build_env_config(cfg, "val")
    rllib_cfg["env_config"] = train_env_config
    rllib_cfg["env"] = CustomizedGrid2OpEnvironment

    # --- Evaluation ---
    eval_cfg = cfg.evaluation
    rllib_cfg["evaluation_interval"] = eval_cfg.evaluation_interval
    rllib_cfg["evaluation_duration_unit"] = eval_cfg.evaluation_duration_unit
    rllib_cfg["always_attach_evaluation_results"] = True
    rllib_cfg["evaluation_num_workers"] = eval_cfg.evaluation_num_workers
    rllib_cfg["evaluation_config"] = {"env_config": val_env_config, "explore": False}

    # Number of eval episodes = number of available chronics for the val env
    val_env_name = val_env_config["env_name"]
    chronics_path = os.path.join(grid2op.get_current_local_dir(), val_env_name, "chronics")
    if os.path.exists(chronics_path):
        rllib_cfg["evaluation_duration"] = len(os.listdir(chronics_path))
    else:
        rllib_cfg["evaluation_duration"] = 50
        logger.warning("Val chronics path not found (%s); defaulting evaluation_duration=50", chronics_path)

    # --- Rollouts / resources ---
    rollouts = cfg.rollouts
    rllib_cfg["num_rollout_workers"] = rollouts.num_rollout_workers
    rllib_cfg["num_learner_workers"] = rollouts.num_learner_workers
    rllib_cfg["num_gpus_per_learner_worker"] = rollouts.num_gpus_per_learner_worker
    rllib_cfg["batch_mode"] = rollouts.batch_mode
    # This project runs the old RLlib API stack (_enable_learner_api=False), which has
    # no separate learner-worker actors: training happens on the single driver process
    # via multi_gpu_train_one_step(). `num_learner_workers`/`num_gpus_per_learner_worker`
    # are new-API-stack fields that RLlib's old-API resource request ignores entirely
    # (see Algorithm.default_resource_request: driver GPU = config.num_gpus). The only
    # knob that actually grants the driver a GPU is `rollouts.num_gpus` below.
    rllib_cfg["num_gpus"] = rollouts.num_gpus
    rllib_cfg["count_steps_by"] = training.get("count_steps_by", rollouts.count_steps_by)
    rllib_cfg["keep_per_episode_custom_metrics"] = rollouts.keep_per_episode_custom_metrics
    rllib_cfg["framework"] = rollouts.framework
    rllib_cfg["exploration_config"] = OmegaConf.to_container(rollouts.exploration_config, resolve=True)

    # --- Callbacks ---
    callback_classes = [get_class(cb["_target_"]) for cb in cfg.callbacks.callbacks]
    rllib_cfg["callbacks"] = make_multi_callbacks(callback_classes)

    # --- Multi-agent ---
    rllib_cfg["policies"] = _build_policies(cfg, algorithm)
    rllib_cfg["policy_mapping_fn"] = policy_mapping_fn
    rllib_cfg["policies_to_train"] = [RL_POLICY]

    # --- Misc ---
    rllib_cfg["my_log_level"] = cfg.experiment.my_log_level
    rllib_cfg["trial_info"] = "trial_id"
    rllib_cfg["total_timesteps"] = cfg.experiment.nb_timesteps

    return rllib_cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs/rllib", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # fail fast on missing values

    set_seed(cfg.experiment.seed)

    rllib_cfg = build_rllib_config(cfg)
    run_training(rllib_cfg, cfg)


if __name__ == "__main__":
    logging.getLogger("pandapower").setLevel(logging.WARNING)
    main()
