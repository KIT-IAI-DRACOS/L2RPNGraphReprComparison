"""
Greedy Tutor that records the simulated outcome of every action in the RL policy's action space.

Differences to `curriculumagent`'s GeneralTutor:
  - The candidate set is exactly the policy's: do-nothing at index 0, then the action-space JSON in
    file order (`CustomizedGrid2OpEnvironment` builds its discrete actions the same way).
  - Actions are compared against the *simulated* do-nothing outcome, not the current rho. The
    GeneralTutor simulates do-nothing but never uses the result, and drops every state where no
    action improved on the current rho, so its dataset contains no do-nothing labels.
  - Every state the RL agent would be asked to act in (`rho.max() > rho_threshold`) is recorded,
    whatever the chosen action.
  - Each candidate is enriched by `reconnection_rule` before simulating, as the RL env does with
    `line_reco: true` before executing an action, so the recorded max rho is the outcome of what
    the policy's action actually executes.

Each recorded state stores the max rho and validity of all candidates; the label rule applied
during the rollout (`select_action`) only decides the trajectory and can be redone offline.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import grid2op
import numpy as np
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation
from lightsim2grid import LightSimBackend

from core.heuristic_actions import reconnection_rule
from grid2op_env.action_converters import load_actions
from grid2op_env.utils import chronic_seed

logger = logging.getLogger(__name__)

DO_NOTHING_INDEX = 0

# Matches `rho_threshold` in configs/rllib/env/case36.yaml; SelectAgentPolicy hands control to the
# RL agent when `rho.max() > rho_threshold`.
RHO_THRESHOLD = 0.95

# A topology action must beat simulated do-nothing by more than this (in rho) to be chosen.
# Without a margin, numerical noise between equal outcomes would decide for topology changes
# that buy nothing but trigger substation cooldowns.
DO_NOTHING_MARGIN = 0.005


@dataclass
class TutorConfig:
    """Configuration for one Tutor run.

    Attributes:
        env_name: Grid2Op environment name or path (e.g. "l2rpn_wcci_2020_train").
        action_space_path: Action-space JSON, the same file the RL env loads.
        save_dir: Directory for the per-chronic `<chronic_name>.npz` files.
        seed: Base seed. Each chronic is seeded with `chronic_seed(seed, chronic)`, so its random
            maintenance schedule is reproducible regardless of which task runs it, but differs
            between chronics.
        rho_threshold: States with `rho.max()` above this are recorded and acted on.
        do_nothing_margin: See `DO_NOTHING_MARGIN`.
        line_reco: Enrich every action with `reconnection_rule`, as the RL env's `line_reco`.
        max_steps: Optional cap on steps per chronic, for smoke tests.
    """

    env_name: str
    action_space_path: Path
    save_dir: Path
    seed: int = 42
    rho_threshold: float = RHO_THRESHOLD
    do_nothing_margin: float = DO_NOTHING_MARGIN
    line_reco: bool = True
    max_steps: int | None = None


def select_action(max_rho: np.ndarray, valid: np.ndarray, margin: float) -> int:
    """Pick the candidate the Tutor executes: the best valid one, preferring do-nothing on near-ties.

    Args:
        max_rho: Simulated max rho per candidate, shape [n_actions]; index 0 is do-nothing.
        valid: Whether each candidate's simulation was legal and did not end the episode.
        margin: A topology action is only chosen if it beats do-nothing by more than this.

    Returns:
        The chosen candidate index. Do-nothing if no candidate is valid.
    """
    topo = np.where(valid, max_rho, np.inf)
    topo[DO_NOTHING_INDEX] = np.inf
    best = int(np.argmin(topo))
    if not np.isfinite(topo[best]):
        return DO_NOTHING_INDEX
    if not valid[DO_NOTHING_INDEX] or topo[best] < max_rho[DO_NOTHING_INDEX] - margin:
        return best
    return DO_NOTHING_INDEX


def simulate_candidate(obs: BaseObservation, action: BaseAction) -> tuple[float, bool]:
    """Simulate one candidate action for the next step.

    An action is invalid if the simulation raises, ends the episode, is illegal (e.g. substation
    cooldown — grid2op then silently simulates do-nothing instead) or is ambiguous.

    Args:
        obs: Current observation.
        action: Candidate action, already enriched by any heuristics.

    Returns:
        `(max_rho, valid)`; `max_rho` is NaN when invalid.
    """
    try:
        sim_obs, _, done, info = obs.simulate(action)
    except Exception as exc:  # noqa: BLE001 - backend divergence must not kill a multi-hour task
        logger.debug(f"Simulation raised: {exc}")
        return float("nan"), False
    invalid = done or info.get("is_illegal", False) or info.get("is_ambiguous", False) or bool(info.get("exception"))
    if invalid:
        return float("nan"), False
    return float(sim_obs.rho.max()), True


class Tutor:
    """Evaluates the candidates of the policy's action space on a state (all, or a subset).

    Args:
        env: The Grid2Op environment the candidates belong to.
        actions: Policy action list, do-nothing at index 0.
        line_reco: Enrich candidates with `reconnection_rule` before simulating.
        margin: See `select_action`.
    """

    def __init__(self, env: BaseEnv, actions: list[BaseAction], line_reco: bool, margin: float):
        self.env = env
        self.actions = actions
        self.line_reco = line_reco
        self.margin = margin

    def enrich(self, obs: BaseObservation, action: BaseAction) -> BaseAction:
        """Apply the RL env's pre-execution heuristics to an action."""
        if self.line_reco:
            return reconnection_rule(obs, action, self.env.action_space)
        return action

    def evaluate(
        self, obs: BaseObservation, candidates: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int, BaseAction]:
        """Simulate candidates and choose one with `select_action`.

        Args:
            obs: Current observation.
            candidates: Indices to simulate; default all. Do-nothing is always simulated, as the
                reference the others must beat. The rest count as invalid, so the choice is the
                Tutor's rule restricted to these candidates.

        Returns:
            `(max_rho, valid, chosen_index, chosen_enriched_action)`, arrays over all actions.
        """
        n = len(self.actions)
        indices = range(n) if candidates is None else sorted({DO_NOTHING_INDEX, *candidates})
        max_rho = np.full(n, np.nan, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        enriched = {}
        for i in indices:
            enriched[i] = self.enrich(obs, self.actions[i])
            max_rho[i], valid[i] = simulate_candidate(obs, enriched[i])
        chosen = select_action(max_rho, valid, self.margin)
        return max_rho, valid, chosen, enriched[chosen]


def load_policy_actions(env: BaseEnv, action_space_path: Path) -> list[BaseAction]:
    """Build the policy's action list as `CustomizedGrid2OpEnvironment` does: do-nothing + JSON."""
    actions = load_actions(str(action_space_path), env)
    actions.insert(DO_NOTHING_INDEX, env.action_space({}))
    return actions


def run_chronic(env: BaseEnv, tutor: Tutor, chronic_id: str, config: TutorConfig) -> dict[str, np.ndarray]:
    """Roll out one chronic with the Tutor and record every state above the threshold.

    Below the threshold the env executes do-nothing (enriched by the heuristics), as the RL env's
    DO_NOTHING_AGENT does.

    Args:
        env: Grid2Op environment.
        tutor: Configured Tutor.
        chronic_id: Chronic path or name, as from `list_chronic_ids`.
        config: Tutor configuration.

    Returns:
        Arrays to save, see `run_tutor_chronics`.
    """
    seed = chronic_seed(config.seed, chronic_id)
    env.seed(seed)
    env.set_id(chronic_id)
    obs = env.reset()
    do_nothing = tutor.actions[DO_NOTHING_INDEX]

    records: dict[str, list] = {k: [] for k in ("obs", "step", "max_rho", "valid", "action",
                                                "load_p_forecast", "load_q_forecast",
                                                "prod_p_forecast", "prod_v_forecast")}
    done = False
    while not done:
        if config.max_steps is not None and obs.current_step >= config.max_steps:
            break
        if obs.rho.max() > config.rho_threshold:
            max_rho, valid, chosen, action = tutor.evaluate(obs)
            # Next-step forecasts are not part of `to_vect()`; kept so forecast features can be
            # compared later without another rollout.
            load_p, load_q, prod_p, prod_v, _ = obs.get_forecast_arrays()
            records["obs"].append(obs.to_vect().astype(np.float32))
            records["step"].append(obs.current_step)
            records["max_rho"].append(max_rho)
            records["valid"].append(valid)
            records["action"].append(chosen)
            records["load_p_forecast"].append(load_p[1])
            records["load_q_forecast"].append(load_q[1])
            records["prod_p_forecast"].append(prod_p[1])
            records["prod_v_forecast"].append(prod_v[1])
        else:
            action = tutor.enrich(obs, do_nothing)
        obs, _, done, _ = env.step(action)

    n_actions = len(tutor.actions)
    obs_dim = env.observation_space.size()
    arrays = {
        "obs": np.asarray(records["obs"], dtype=np.float32).reshape(-1, obs_dim),
        "step": np.asarray(records["step"], dtype=np.int32),
        "max_rho": np.asarray(records["max_rho"], dtype=np.float32).reshape(-1, n_actions),
        "valid": np.asarray(records["valid"], dtype=bool).reshape(-1, n_actions),
        "action": np.asarray(records["action"], dtype=np.int16),
        "load_p_forecast": np.asarray(records["load_p_forecast"], dtype=np.float32).reshape(-1, env.n_load),
        "load_q_forecast": np.asarray(records["load_q_forecast"], dtype=np.float32).reshape(-1, env.n_load),
        "prod_p_forecast": np.asarray(records["prod_p_forecast"], dtype=np.float32).reshape(-1, env.n_gen),
        "prod_v_forecast": np.asarray(records["prod_v_forecast"], dtype=np.float32).reshape(-1, env.n_gen),
        # The episode survived iff `steps_survived == max_step` (and `max_steps` was not set).
        "steps_survived": np.int32(obs.current_step),
        "max_step": np.int32(obs.max_step),
        "chronic": np.str_(Path(chronic_id).name),
        "seed": np.int64(seed),
        "base_seed": np.int64(config.seed),
        "rho_threshold": np.float32(config.rho_threshold),
        "do_nothing_margin": np.float32(config.do_nothing_margin),
        "action_space": np.str_(Path(config.action_space_path).name),
    }
    return arrays


def list_chronic_ids(env_name: str) -> list[str]:
    """List chronic subpaths for an environment, in grid2op's (stable) order."""
    env = grid2op.make(env_name, backend=LightSimBackend())
    try:
        return list(env.chronics_handler.subpaths)
    finally:
        env.close()


def run_tutor_chronics(config: TutorConfig, chronic_ids: list[str]) -> None:
    """Run the Tutor over a batch of chronics, one `<chronic_name>.npz` each.

    Files are written atomically (temp file + rename), so a task killed at its walltime leaves
    no truncated file, and chronics whose file exists are skipped on resubmission.

    Per-state arrays (n = recorded states, A = len(actions) incl. do-nothing):
        obs [n, obs_dim] `obs.to_vect()`; step [n]; max_rho [n, A] (NaN where invalid);
        valid [n, A]; action [n] executed candidate; {load,prod}_{p,q|v}_forecast [n, n_load|n_gen].
    Per-chronic scalars: steps_survived, max_step, chronic, seed (the chronic's), base_seed, rho_threshold,
        do_nothing_margin, action_space.

    Args:
        config: Tutor configuration.
        chronic_ids: Chronic paths or names to process in order.
    """
    config.save_dir.mkdir(parents=True, exist_ok=True)
    env = grid2op.make(config.env_name, backend=LightSimBackend())
    try:
        actions = load_policy_actions(env, config.action_space_path)
        tutor = Tutor(env, actions, line_reco=config.line_reco, margin=config.do_nothing_margin)
        logger.info(f"Tutor over {len(actions)} candidates (do-nothing + {len(actions) - 1} actions)")

        for chronic_id in chronic_ids:
            out_path = config.save_dir / f"{Path(chronic_id).name}.npz"
            if out_path.exists():
                logger.info(f"Skipping chronic {chronic_id}: {out_path} already exists")
                continue

            t0 = time.perf_counter()
            arrays = run_chronic(env, tutor, chronic_id, config)
            tmp_path = out_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp_path, **arrays)
            os.replace(tmp_path, out_path)

            n = len(arrays["action"])
            n_dn = int((arrays["action"] == DO_NOTHING_INDEX).sum())
            logger.info(
                f"Saved {n} states ({n_dn} do-nothing) for {Path(chronic_id).name}: survived "
                f"{arrays['steps_survived']}/{arrays['max_step']} steps in {time.perf_counter() - t0:.0f} s"
            )
    finally:
        env.close()
