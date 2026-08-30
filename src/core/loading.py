"""
Load configs and agents from checkpoints for evaluation.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional, Tuple

from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment

from ray.rllib.models import ModelCatalog

from agents import RllibAgent
from core.constants import RL_POLICY
from grid2op_env import CustomizedGrid2OpEnvironment

logger = logging.getLogger(__name__)


def _config_from_experiment_state(trial_dir: Path) -> dict | None:
    """Extract the trial config from a Ray Tune experiment_state-*.json in the parent dir.

    First tries to match by logdir. If no exact match is found, falls back to
    the first valid config in the same experiment folder — all trials in a
    multi-seed group share the same config structure (only the seed differs,
    which is irrelevant for inference).
    """
    fallback_config: dict | None = None

    for state_file in sorted(trial_dir.parent.glob("experiment_state-*.json")):
        try:
            content = state_file.read_text().strip()
            if not content:
                continue
            d = json.loads(content)
        except Exception:
            continue
        for td in d.get("trial_data", []):
            if not isinstance(td, list):
                continue
            for item in td:
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except Exception:
                        continue
                if not isinstance(item, dict):
                    continue
                cfg = item.get("config")
                if not cfg:
                    continue
                logdir = item.get("relative_logdir") or item.get("logdir", "")
                if Path(logdir).name == trial_dir.name:
                    return cfg           # exact match
                if fallback_config is None and "env_config" in cfg:
                    fallback_config = cfg  # keep first valid config as fallback

    if fallback_config is not None:
        logger.info(
            f"No exact experiment_state match for {trial_dir.name}; "
            "using sibling trial config (same experiment, seeds share config structure)."
        )
    return fallback_config


def load_config(checkpoint_path: Path) -> dict:
    """
    Find and load the algorithm config for a trial directory.

    Search order:
    1. params.json in the trial directory.
    2. params.json in the parent directory (algo-level fallback).
    3. experiment_state-*.json in the parent directory (re-run trials that Ray
       Tune did not serialise params.json for).

    :param checkpoint_path: Path to the trial directory.
    :return: Raw params dictionary.
    """
    params_path = checkpoint_path / "params.json"
    if not params_path.exists():
        params_path = checkpoint_path.parent / "params.json"
        if params_path.exists():
            logger.info(f"Found params.json in parent directory: {params_path}")
        else:
            cfg = _config_from_experiment_state(checkpoint_path)
            if cfg is not None:
                logger.info(f"Loaded config from experiment_state for {checkpoint_path.name}")
                return cfg
            raise FileNotFoundError(
                f"params.json not found at {checkpoint_path / 'params.json'} or "
                f"{params_path}, and no matching entry in experiment_state-*.json"
            )

    with open(params_path, 'r') as f:
        return json.load(f)


def preprocess_config(params: dict) -> dict:
    """
    Preprocess a raw params dictionary loaded from params.json.

    Removes serialized object strings that cannot be used directly (they will be
    recreated by the environment). Sets lib_dir to the current working directory,
    syncs the observation_space from the training config into the evaluation config,
    and optionally overrides the environment name.

    :param params: Raw params dictionary as returned by load_config
    :return: Preprocessed params dictionary
    """
    if "env_config" not in params:
        raise ValueError("No env_config found in params.json")

    def _remove_serialized_objects(_env_config: dict) -> None:
        if "grid2op_kwargs" not in _env_config:
            return
        grid2op_kwargs = _env_config["grid2op_kwargs"]
        keys_to_remove = [
            key for key, value in grid2op_kwargs.items()
            if (isinstance(value, str) and value.startswith("<"))
            or (isinstance(value, dict) and value.get("_type") == "CLOUDPICKLE_FALLBACK")
        ]
        for key in keys_to_remove:
            del grid2op_kwargs[key]
            logger.debug(f"Removing serialized object: {key}")
        logger.debug(f"Cleaned {len(keys_to_remove)} serialized objects from grid2op_kwargs")

    env_config = params["env_config"]
    evaluation_env_config = params["evaluation_config"]["env_config"]

    _remove_serialized_objects(env_config)
    _remove_serialized_objects(evaluation_env_config)

    env_config["lib_dir"] = os.getcwd()
    evaluation_env_config["lib_dir"] = os.getcwd()
    evaluation_env_config["observation_space"] = env_config["observation_space"]

    return params


def _policy_from_checkpoint_with_conv_type(policy_ckpt_dir: str, conv_type: str):
    """Load a policy, patching conv_type into the embedded model config.

    ``Policy.from_checkpoint`` rebuilds the model from config serialized inside
    the checkpoint pickle.  Old checkpoints lack a ``conv_type`` key, so we
    load the pickle, inject the value, and call ``Policy.from_state`` directly.

    :param policy_ckpt_dir: Path to ``policies/<policy_name>/`` inside a checkpoint.
    :param conv_type: ``"gcn"`` or ``"gin"``.
    :return: Restored RLlib Policy.
    """
    from ray.rllib import Policy

    ckpt_json = os.path.join(policy_ckpt_dir, "rllib_checkpoint.json")
    with open(ckpt_json) as f:
        ckpt_info = json.load(f)

    state_file = os.path.join(policy_ckpt_dir, ckpt_info["state_file"])
    with open(state_file, "rb") as f:
        state = pickle.load(f)

    try:
        gnn_cfg = state["policy_spec"]["config"]["model"]["custom_model_config"]["gnn"]
        gnn_cfg["conv_type"] = conv_type
        logger.info("Patched checkpoint config: conv_type=%s", conv_type)
    except (KeyError, TypeError) as exc:
        logger.warning("Could not patch conv_type in checkpoint config: %s", exc)

    return Policy.from_state(state)


def load_rllib_agent(
        checkpoint_path: str,
        policy_name: str,
        checkpoint_name: str,
        env_name: str,
        env_config: dict,
        conv_type: Optional[str] = None,
):
    """
    Load an RLlib agent from a checkpoint.

    :param checkpoint_path: Path to the experiment directory containing checkpoints
    :param policy_name: Name of the policy (defaults to RL_POLICY, specified in core/constants.py)
    :param checkpoint_name: Name of the checkpoint folder (e.g., "checkpoint_000000")
    :param env_name: Name of the Grid2Op environment to evaluate on
    :param env_config: Environment configuration dictionary
    :param conv_type: Override GNN conv type in the checkpoint config (``"gin"`` for
        checkpoints trained before the GCNConv revert). ``None`` = use embedded config.
    :return: Tuple of (RllibAgent, Grid2Op Environment, gym_wrapper)
    """
    # Add env_name to config for CustomizedGrid2OpEnvironment
    env_config["env_name"] = env_name

    # Create the CustomizedGrid2OpEnvironment to get proper observation/action spaces
    gym_wrapper = CustomizedGrid2OpEnvironment(env_config)

    # Get the underlying Grid2Op environment for evaluation
    g2op_env = gym_wrapper.env_gym.init_env

    # Re-register custom models (ray.shutdown() clears the ModelCatalog registry)
    from rl.gnn_ppo_model import GNNModel
    ModelCatalog.register_custom_model("gnn_model", GNNModel)

    # Load the RLlib agent
    agent = RllibAgent(
        action_space=g2op_env.action_space,
        env_config=env_config,
        file_path=checkpoint_path,
        policy_name=policy_name,
        checkpoint_name=checkpoint_name,
        gym_wrapper=gym_wrapper,
        conv_type=conv_type,
    )

    # Return gym_wrapper to keep it alive and prevent premature cleanup
    return agent, g2op_env, gym_wrapper


class AgentSpec:
    def __init__(self, name: str, load_path: Path, checkpoint_name: str, policy_name: str = RL_POLICY):
        self.name = name
        self.checkpoint_name = checkpoint_name
        self.policy_name = policy_name
        self.load_path = load_path


def load_agent_from_spec(agent_spec: AgentSpec, env_name: str = "l2rpn_case14_sandbox_val") -> Tuple[BaseAgent, Environment, CustomizedGrid2OpEnvironment]:
    params = load_config(agent_spec.load_path)
    params = preprocess_config(params)
    env_config = params["evaluation_config"]["env_config"]
    return load_rllib_agent(
        checkpoint_path=agent_spec.load_path,
        policy_name=agent_spec.policy_name,
        checkpoint_name=agent_spec.checkpoint_name,
        env_name=env_name,
        env_config=env_config
    )
