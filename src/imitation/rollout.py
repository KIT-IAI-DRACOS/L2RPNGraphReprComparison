"""
Rollout evaluation of a policy outside RLlib, with the RL env's control logic: do-nothing while
`rho.max() <= rho_threshold`, the policy's action above it, every action enriched by
`reconnection_rule` (env `line_reco`), and the observation converter updated on every step.

The policy is any `ActionChooser`: the greedy (argmax) action of a policy network
(`greedy_chooser`), the Tutor's rule over the network's k highest-ranked actions (`top_k_chooser`,
which simulates them and keeps do-nothing unless one beats it), or always do-nothing
(`do_nothing_chooser`) as the reference without topology actions. Each chronic is seeded with
`grid2op_env.utils.chronic_seed`, so every agent faces exactly the maintenance schedule the Tutor
faced on that chronic and survival can be compared per chronic.
Results use the shared episode table of `analysis.episodes`.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from analysis.episodes import make_episodes
from core.heuristic_actions import reconnection_rule
from grid2op_env.observation_converter import ObservationConverter
from grid2op_env.utils import chronic_seed
from imitation.pretrain import policy_logits
from imitation.tutor import DO_NOTHING_INDEX, Tutor

logger = logging.getLogger(__name__)

# Maps the Grid2Op observation and its converted form (None without converter) to a policy action index.
ActionChooser = Callable[[BaseObservation, dict | None], int]


def resolve_chronic_ids(env: BaseEnv, names: list[str]) -> list[str]:
    """Map chronic names (as in split files and episode tables) to the env's chronic paths."""
    by_name = {Path(p).name: p for p in env.chronics_handler.subpaths}
    missing = [n for n in names if Path(n).name not in by_name]
    if missing:
        raise KeyError(f"{len(missing)} chronics not in {env.env_name}, e.g. {missing[:3]}")
    return [by_name[Path(n).name] for n in names]


@torch.no_grad()
def action_logits(model: TorchModelV2, gym_obs: dict, device: torch.device) -> torch.Tensor:
    """Action logits [A] of the policy for one converted observation."""
    obs = {k: torch.as_tensor(np.asarray(v)).unsqueeze(0).to(device) for k, v in gym_obs.items()}
    return policy_logits(model, obs)[0]


def choose_action(model: TorchModelV2, gym_obs: dict, device: torch.device) -> int:
    """Greedy (argmax) action of the policy for one converted observation."""
    return int(action_logits(model, gym_obs, device).argmax().item())


def greedy_chooser(model: TorchModelV2, device: torch.device) -> ActionChooser:
    """Chooser that takes the argmax action of `model` (switched to eval mode)."""
    model.eval()
    return lambda obs, gym_obs: choose_action(model, gym_obs, device)


def top_k_chooser(model: TorchModelV2, device: torch.device, tutor: Tutor, k: int) -> ActionChooser:
    """Chooser that simulates the k highest-ranked actions of `model` and applies the Tutor's rule.

    The best simulated action is taken only if it beats simulated do-nothing by more than the
    Tutor's margin, otherwise do-nothing (see `imitation.tutor.select_action`). With k = number of
    actions this is the Tutor itself, whatever the ranking.
    """
    model.eval()

    def choose(obs: BaseObservation, gym_obs: dict) -> int:
        logits = action_logits(model, gym_obs, device)
        top = logits.topk(min(k, logits.shape[-1])).indices.tolist()
        return tutor.evaluate(obs, top)[2]

    return choose


def do_nothing_chooser(obs: BaseObservation, gym_obs: dict | None) -> int:
    """Chooser that always takes do-nothing; needs no converter."""
    return DO_NOTHING_INDEX


def run_policy_chronic(
    env: BaseEnv,
    converter: ObservationConverter | None,
    chooser: ActionChooser,
    actions: list[BaseAction],
    chronic_id: str,
    base_seed: int,
    rho_threshold: float,
    line_reco: bool,
) -> dict:
    """Roll out one chronic and return its episode row (without `agent`).

    Args:
        env: Grid2Op env containing the chronic.
        converter: Observation converter feeding `chooser`; None if the chooser ignores observations.
        chooser: Policy above `rho_threshold`.
        actions: Policy actions, do-nothing at index 0.
        chronic_id: Chronic path, as from `resolve_chronic_ids`.
        base_seed: Base seed of `chronic_seed`.
        rho_threshold: The policy acts only above this max rho.
        line_reco: Enrich every action with `reconnection_rule`.

    Returns:
        Episode row with chronic, seed, steps_survived, max_step, n_decisions, n_actions (decisions
        whose topology action was executed) and n_illegal (decisions whose topology action the env
        rejected as illegal).
    """
    seed = chronic_seed(base_seed, chronic_id)
    env.seed(seed)
    env.set_id(chronic_id)
    obs = env.reset()
    gym_obs = converter.to_gym(obs) if converter is not None else None
    n_decisions = n_actions = n_illegal = 0
    done = False
    while not done:
        chosen = DO_NOTHING_INDEX
        if obs.rho.max() > rho_threshold:
            chosen = chooser(obs, gym_obs)
            n_decisions += 1
        action = actions[chosen]
        if line_reco:
            action = reconnection_rule(obs, action, env.action_space)
        obs, _, done, info = env.step(action)
        if chosen != DO_NOTHING_INDEX:
            illegal = bool(info.get("is_illegal", False))
            n_illegal += illegal
            n_actions += not illegal
        gym_obs = converter.to_gym(obs) if converter is not None else None
    return {
        "chronic": Path(chronic_id).name,
        "seed": seed,
        "steps_survived": int(obs.current_step),
        "max_step": int(obs.max_step),
        "n_decisions": n_decisions,
        "n_actions": n_actions,
        "n_illegal": n_illegal,
    }


def evaluate_policy(
    env: BaseEnv,
    converter: ObservationConverter | None,
    chooser: ActionChooser,
    actions: list[BaseAction],
    chronic_names: list[str],
    agent: str,
    base_seed: int,
    rho_threshold: float,
    line_reco: bool,
) -> pd.DataFrame:
    """Roll out several chronics (see `run_policy_chronic`).

    Returns:
        Their episode table (`analysis.episodes`), with `agent` as agent column.
    """
    rows = []
    for chronic_id in resolve_chronic_ids(env, chronic_names):
        row = run_policy_chronic(env, converter, chooser, actions, chronic_id, base_seed, rho_threshold, line_reco)
        logger.info(f"{row['chronic']}: survived {row['steps_survived']}/{row['max_step']} ({row['n_decisions']} decisions, {row['n_actions']} actions)")
        rows.append({"agent": agent, **row})
    return make_episodes(rows)
