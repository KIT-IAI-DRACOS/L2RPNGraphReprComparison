"""
Callbacks for RLlib
"""
import logging
import time
from typing import Dict, Optional, List, Any

import grid2op
import numpy as np
from ray._private.dict import unflattened_lookup
from ray.rllib import RolloutWorker, BaseEnv, Policy
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.tune.experiment import Trial
from ray.tune.experimental.output import (
    TuneReporterBase,
    get_air_verbosity,
    _get_time_str,
    _current_best_trial,
)
from tabulate import tabulate

from core.constants import RL_POLICY, HIGH_LEVEL_AGENT
from grid2op_env.observation_converter import GraphObservationConverter
from visualization import get_node_styles

_POLICY_ID = "default_policy"
logger = logging.getLogger(__name__)


def _get_policy(algorithm: Algorithm, policy_id: str = _POLICY_ID):
    """Try multiple policy IDs to handle different algorithm setups."""
    for pid in (policy_id, RL_POLICY, _POLICY_ID):
        p = algorithm.get_policy(pid)
        if p is not None:
            return p
    return None


class TuneCallback(TuneReporterBase):
    def __init__(
            self,
            log_level: int,
            metric: str,
            mode: str = "max",
            heartbeat_freq: int = 30,
            eval_freq: int = 1,
    ):
        super().__init__(get_air_verbosity(0))
        self._start_end_verbosity = 1
        self._heartbeat_freq = heartbeat_freq
        self.log_level = log_level
        self._eval_freq = eval_freq
        self._metric = metric
        self._mode = mode
        self._best_trial = None

    def print_heartbeat(self, trials, *args, force: bool = False):
        if force or time.time() - self._last_heartbeat_time >= self._heartbeat_freq:
            self._print_heartbeat(trials, *args, force=force)
            self._last_heartbeat_time = time.time()

    def _print_heartbeat(self, trials, *sys_args, force: bool = False):
        current_best_trial, metric_val = _current_best_trial(trials, self._metric, self._mode)

        rows = [
            ["Trial status", self._get_overall_trial_progress_str(trials)],
            ["Current time", self._time_heartbeat_str],
            ["Logical resource usage", ", ".join(sys_args) if sys_args else "N/A"],
        ]
        if current_best_trial:
            best_str = (
                f"{current_best_trial.trial_id} | score: {unflattened_lookup(self._metric, current_best_trial.last_result)}"
                f" at timestep: {current_best_trial.last_result['timesteps_total']}"
            )
            rows.append(["Current best trial", best_str])

        print(tabulate(rows, tablefmt="heavy_outline"))

    def on_trial_result(
            self,
            iteration: int,
            trials: List[Trial],
            trial: Trial,
            result: Dict,
            **info,
    ):
        self.print_heartbeat(trials)
        if self.log_level:
            # start printing after first evaluation
            if result['training_iteration'] % self._eval_freq == 0:
                # print intermediate results for trial:
                self._print_result(trial, result)

    def _print_result(self, trial: Trial, result: Optional[Dict] = None, force: bool = False):
        result = result or trial.last_result
        trial_id = str(trial)
        eval_res = result.get("evaluation", {})
        train_res = result.get("custom_metrics", {})
        # Print the table
        headers = ["trial name", "iter", "total time", "total steps", "total interactions",
                   "eval survival mean", "eval survival std", "eval return",
                   "train survival mean", "train survival std", "train return", "episodes this iteration"]

        eval_custom = eval_res.get("custom_metrics", {})
        table = [[trial_id,
                  result.get('training_iteration', "N/A"),
                  _get_time_str(self._start_time, time.time())[1],
                  result.get("timesteps_total", "N/A"),
                  result.get("custom_metrics", {}).get("total_agent_interact", "N/A"),
                  eval_custom.get("grid2op_end_mean", "N/A"),
                  eval_custom.get("grid2op_end_std", "N/A"),
                  eval_res.get("episode_reward_mean", "N/A"),
                  train_res.get("grid2op_end_mean", "N/A"),
                  train_res.get("grid2op_end_std", "N/A"),
                  result.get("episode_reward_mean", "N/A"),
                  result.get("episodes_this_iter", "N/A"), ]]

        print(tabulate(table, headers, tablefmt="rounded_grid", floatfmt=".3f"))


class CustomMetricsCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None, graph_viz_freq: int = 10):
        super().__init__(legacy_callbacks_dict)
        self.curr_level = 0
        self.node_styles = None
        self.powerline_edge_index = None
        self._graph_viz_freq = graph_viz_freq

    def on_algorithm_init(
            self,
            *,
            algorithm: Algorithm,
            **kwargs,
    ) -> None:
        self.curr_level = 0
        # get node styles and edge index once
        env_name = algorithm.config.env_config["env_name"]
        env = grid2op.make(env_name)
        self.node_styles = get_node_styles(env, GraphObservationConverter)
        obs_space = GraphObservationConverter(env.observation_space)
        self.powerline_edge_index = obs_space._get_edge_index(env.reset())
        policy = _get_policy(algorithm)
        if policy is not None and hasattr(policy, "model"):
            model_device = next(policy.model.parameters()).device
            logger.info(
                f"Instantiated model class: {type(policy.model).__name__} "
                f"on device={model_device} (config.num_gpus={algorithm.config.num_gpus})"
            )

        if hasattr(algorithm, "curriculum_training") and algorithm.curriculum_training:
            print(f"Start with curriculum level {self.curr_level}")

    def on_episode_end(
            self,
            *,
            episode: EpisodeV2,
            worker: Optional[RolloutWorker] = None,
            base_env: Optional[BaseEnv] = None,
            policies: Optional[Policy] = None,
            env_index: Optional[int] = None,
            **kwargs: Dict[str, Any],
    ) -> None:
        """
        Collect extra metrics such as:
         - grid2op episode length - RLlib counts extra steps because of high level agent.
         - chronic id.
        """
        agents_steps = {k: len(v) for k, v in episode._agent_reward_history.items()}

        episode.custom_metrics["corrected_ep_len"] = agents_steps.get(HIGH_LEVEL_AGENT, 0)
        if base_env is not None:
            envs = base_env.get_sub_environments()

            # New extra metrics:
            interact_count = np.array([env.interact_count for env in envs]).mean()
            active_dn_count = np.array([env.active_dn_count for env in envs]).mean()
            reconnect_count = np.array([env.reconnect_count for env in envs]).mean()
            disconnect_count = np.array([env.disconnect_count for env in envs]).mean()
            reset_count = np.array([env.reset_count for env in envs]).mean()
            grid2op_end = np.array([env.env_g2op.current_obs.current_step for env in envs]).mean()

            episode.custom_metrics["interact_count"] = interact_count
            episode.custom_metrics["active_dn_count"] = active_dn_count
            episode.custom_metrics["reconnect_count"] = reconnect_count
            episode.custom_metrics["disconnect_count"] = disconnect_count
            episode.custom_metrics["reset_count"] = reset_count
            episode.custom_metrics["grid2op_end"] = grid2op_end

    def on_evaluate_end(
            self,
            *,
            algorithm: "Algorithm",
            evaluation_metrics: dict,
            **kwargs,
    ) -> None:
        data = evaluation_metrics.get("evaluation", {})
        custom_metrics = data.get("custom_metrics", {})
        # Save summarized results
        grid2op_end = custom_metrics.get("grid2op_end", -1)
        custom_metrics["grid2op_end_min"] = int(np.min(grid2op_end))
        custom_metrics["grid2op_end_mean"] = int(np.mean(grid2op_end))
        custom_metrics["grid2op_end_max"] = int(np.max(grid2op_end))
        custom_metrics["grid2op_end_std"] = np.std(grid2op_end)
        # Extra metrics:
        interact_count = custom_metrics.get("interact_count", -1)
        custom_metrics["mean_interact_count"] = np.mean(interact_count)
        custom_metrics["total_agent_interact"] = np.sum(interact_count)
        custom_metrics["mean_active_dn_count"] = np.mean(custom_metrics.get("active_dn_count", -1))
        custom_metrics["mean_reconnect_count"] = np.mean(custom_metrics.get("reconnect_count", -1))
        custom_metrics["mean_disconnect_count"] = np.mean(custom_metrics.get("disconnect_count", -1))
        custom_metrics["mean_reset_count"] = np.mean(custom_metrics.get("reset_count", -1))

    def on_train_result(
            self,
            *,
            algorithm: "Algorithm",
            result: dict,
            **kwargs,
    ) -> None:
        custom = result.get("custom_metrics", {})  # old metrics should be replaced
        updated = {  # new metrics to replace the old ones
            "grid2op_end_mean": int(np.mean(custom.get("grid2op_end", -1))),
            "grid2op_end_std": np.var(custom.get("grid2op_end", -1)),
            "corrected_ep_len_mean": int(np.mean(custom.get("corrected_ep_len", -1))),
            "mean_interact_count": np.mean(custom.get("interact_count", -1)),
            "total_agent_interact": np.sum(custom.get("interact_count", -1)),
            "active_dn_count": np.mean(custom.get("active_dn_count", -1)),
            "mean_reconnect_count": np.mean(custom.get("reconnect_count", -1)),
            "disconnect_count": np.mean(custom.get("disconnect_count", -1)),
            "reset_count": np.mean(custom.get("reset_count", -1))
        }

        if len(custom) > 0:
            result["custom_metrics"] = updated
        # TBXLoggerCallback can't serialize a list-of-strings; drop it from the
        # train-result dict. on_evaluate_end reads chronic_id from a separate
        # evaluation_metrics path, so this only affects the training logger.
        episode_media = result.get("episode_media", {})
        if "chronic_id" in episode_media:
            del result["episode_media"]["chronic_id"]

        if algorithm.curriculum_training:
            if self.curr_level < len(algorithm.curriculum_threshold) and \
                    result['timesteps_total'] > algorithm.curriculum_threshold[self.curr_level]:
                self.curr_level += 1
                algorithm.workers.foreach_worker(
                    lambda ev: ev.foreach_env(
                        lambda env: env.set_curriculum(self.curr_level)
                    )
                )
                print(f"Curriculum level increased to {self.curr_level}")


class TimerCallback(DefaultCallbacks):
    """
    Collects per-step wall-clock timings from the environment and model and
    logs them to TensorBoard as custom metrics.

    Environment timings (per-episode means across all sub-environments):
      g2op_step_ms, heuristic_ms, obs_conversion_ms, forecast_ms, edge_index_ms

    Model timings (last learning forward pass, read from local policy):
      edge_prep_ms, graph_data_cache_ms, graph_data_compute_ms, gnn_ms, fcn_ms

    Note: model timings reflect the *learning* forward pass, not inference.
    They are only populated for models that expose a ``_timings`` dict.
    """

    _ENV_TIMER_KEYS = [
        "g2op_step_ms",
        "heuristic_ms",
        "obs_conversion_ms",
        "forecast_ms",
        "edge_index_ms",
    ]

    def on_episode_end(
        self,
        *,
        episode: EpisodeV2,
        worker: Optional[RolloutWorker] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Policy] = None,
        env_index: Optional[int] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        if base_env is None:
            return
        envs = base_env.get_sub_environments()
        for key in self._ENV_TIMER_KEYS:
            values = [
                env._timing_sum.get(key, 0.0) / max(env._timing_count, 1)
                for env in envs
                if hasattr(env, "_timing_sum")
            ]
            if values:
                episode.custom_metrics[key] = np.mean(values)

    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: dict,
        **kwargs,
    ) -> None:
        policy = _get_policy(algorithm)
        if policy is None or not hasattr(policy, "model"):
            return
        model = policy.model
        if not hasattr(model, "_timings") or not model._timings:
            return
        custom = result.setdefault("custom_metrics", {})
        for key, val in model._timings.items():
            custom[key] = val


