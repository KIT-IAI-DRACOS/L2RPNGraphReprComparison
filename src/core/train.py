"""
Utilities in the grid2op experiments.
"""
import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Dict

import grid2op
import ray
from omegaconf import DictConfig, OmegaConf
from ray import air, tune
from ray.tune.experiment import Trial
from ray.tune.logger import TBXLoggerCallback
from ray.tune.result_grid import ResultGrid
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper.stopper import Stopper
from tabulate import tabulate

from algorithms.custom_ppo import CustomPPO
from algorithms.optuna_search import MyOptunaSearch
from core.callback import TuneCallback
from core.constants import RL_POLICY, Style
from core.loading import load_rllib_agent
from core.utils import delete_nested_key, set_nested_key

# Configure logging
logger = logging.getLogger(__name__)

_TRAINABLE_MAP = {
    "ppo": CustomPPO,
}


def get_num_available_episodes(env_name: str) -> int:
    """
    Get the number of available episodes for a given environment.

    :param env_name: Name of the Grid2Op environment
    :return: Number of available episodes
    """
    try:
        chronics_path = os.path.join(
            f"{grid2op.get_current_local_dir()}",
            env_name,
            "chronics"
        )
        if os.path.exists(chronics_path):
            num_episodes = len(os.listdir(chronics_path))
            logger.info(f"Found {num_episodes} available episodes for environment {env_name}")
            return num_episodes
        else:
            logger.warning(f"Chronics path not found: {chronics_path}. Defaulting to 50 episodes.")
            return 50
    except Exception as e:
        logger.warning(f"Error counting episodes: {e}. Defaulting to 50 episodes.")
        return 50


class MaxCustomMetricStopper(Stopper):
    """Stop trials after reaching a maximum value for the custom metric

    Args:
        metric: Metric to use.
        max_value: If custom metric reaches this value stop trials
    """

    def __init__(self, metric: str, max_value: int):
        self._max_value = max_value
        self._custom_Metric = metric

    def __call__(self, trial_id: str, result: Dict):
        logger.info("current value custom metric: ", result["custom_metric"][self._custom_Metric])
        return result["custom_metric"][self._custom_Metric] >= self._max_value

    def stop_all(self):
        return False


class TimeStopper(Stopper):
    def __init__(self, deadline):
        self._start = time()
        if isinstance(deadline, str):
            self._deadline = int(deadline.split(":")[0]) * 3600 + int(deadline.split(":")[1]) * 60
            logger.info("Run training for ", deadline, " hours.")
        else:
            self._deadline = deadline * 60  # Stop all trials after deadline minutes
            logger.info("Run training for ", deadline, " minutes.")

    def __call__(self, trial_id, result):
        return False

    def stop_all(self):
        return time() - self._start > self._deadline


def _run_post_training_analysis(
    trial_path: Path,
    checkpoint_name: str,
    test_env_name: str,
    num_episodes: int,
    env_config: dict,
) -> None:
    """Load the best checkpoint and run the full post-training analysis suite."""
    from analysis.runner import PostTrainingRunner
    from analysis.analyzers import SurvivalAnalyzer, CongestionProfileAnalyzer, TopologyActionAnalyzer

    agent, env, _ = load_rllib_agent(
        checkpoint_path=trial_path,
        policy_name=RL_POLICY,
        checkpoint_name=checkpoint_name,
        env_name=test_env_name,
        env_config=env_config,
    )

    out_dir = trial_path / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Analysis output directory: %s", out_dir)

    analyzers = [
        SurvivalAnalyzer(),
        CongestionProfileAnalyzer(env_name=test_env_name),
        TopologyActionAnalyzer(env_name=test_env_name),
    ]

    runner = PostTrainingRunner(agent, env, analyzers)
    runner.run(num_episodes=num_episodes)
    runner.save_all(out_dir)

    # Copy TensorBoard event files so the analysis folder is self-contained
    import shutil
    for tf_file in trial_path.glob("events.out.tfevents.*"):
        shutil.copy2(tf_file, out_dir / tf_file.name)
    logger.info("Analysis complete. Results in: %s", out_dir)


def _inject_search_space(rllib_cfg: dict[str, Any], search_space: dict) -> None:
    """Inject Optuna search space entries into the RLLib config as Ray Tune samplers.

    Reads the declarative search space from the optimization config and writes
    tune.uniform / tune.loguniform / tune.randint / tune.choice samplers into
    the corresponding nested positions of rllib_cfg.  Keys use '/' as a path
    separator (e.g. 'model/custom_model_config/encoder/hidden_dim').

    Args:
        rllib_cfg:    The assembled RLLib config dict (mutated in-place).
        search_space: Dict mapping '/'-separated key paths to spec dicts with
                      keys 'type' ('float'|'int'|'categorical'), and either
                      'low'+'high' (+ optional 'log': true) or 'choices'.
    """
    from ray import tune as ray_tune

    _TYPE_BUILDERS = {
        "float": lambda s: ray_tune.loguniform(s["low"], s["high"]) if s.get("log") else ray_tune.uniform(s["low"], s["high"]),
        "int": lambda s: ray_tune.randint(s["low"], s["high"]),
        "categorical": lambda s: ray_tune.choice(list(s["choices"])),
    }

    for path, spec in search_space.items():
        kind = spec.get("type")
        builder = _TYPE_BUILDERS.get(kind)
        if builder is None:
            raise ValueError(f"Unknown search space type '{kind}' for key '{path}'. Use float, int, or categorical.")
        set_nested_key(rllib_cfg, path, builder(spec))
        logger.info(f"HPO search space: {path} = {spec}")


def run_training(rllib_cfg: dict[str, Any], cfg: DictConfig) -> ResultGrid:
    """Run RLLib PPO training driven by the Hydra config.

    Args:
        rllib_cfg: Flat RLLib algorithm config dict (built by build_rllib_config).
        cfg:       Full assembled Hydra DictConfig (cfg.experiment, cfg.optimization, …).
    """
    exp = cfg.experiment
    opt = cfg.optimization

    _setup_ray(exp)

    # --- Optuna search (optional) ---
    algo = None
    asha = None
    if opt.enable:
        search_space = OmegaConf.to_container(opt.search_space, resolve=True)
        _inject_search_space(rllib_cfg, search_space)

        algo = MyOptunaSearch(
            metric=opt.score_metric,
            mode=opt.mode,
            points_to_evaluate=[opt.points_to_evaluate] if opt.points_to_evaluate is not None else None,
        )
        if opt.load_from is not None:
            logger.info("Retrieving previous Optuna results from: ", opt.load_from)
            algo.restore_from_dir(opt.load_from)
            for key in algo._space.keys():
                if '/' in key:
                    delete_nested_key(rllib_cfg, key)
                else:
                    rllib_cfg.pop(key, None)

        asha = ASHAScheduler(
            time_attr="timesteps_total",
            max_t=exp.nb_timesteps,
            grace_period=max(1, exp.nb_timesteps // 2),
            reduction_factor=5,
        )

    dur = get_duration(exp)
    time_budget = int(dur * 0.9) if (opt.enable and dur) else None
    if time_budget:
        logger.info(f"Optimization time budget: {time_budget}s ({time_budget / 3600:.2f}h, 10% buffer for cleanup)")

    job_id = _get_job_id(cfg)

    storage_path = os.path.abspath(os.path.join(os.getcwd(), "results"))
    os.makedirs(storage_path, exist_ok=True)

    algorithm = OmegaConf.select(cfg, "training.algorithm", default="ppo")
    trainable_cls = _TRAINABLE_MAP.get(algorithm)
    if trainable_cls is None:
        raise ValueError(f"No trainable class registered for algorithm '{algorithm}'.")

    # --- Build TuneConfig ---
    if opt.enable:
        tune_config = tune.TuneConfig(
            trial_name_creator=lambda t: trial_str_creator(t, job_id),
            trial_dirname_creator=trial_dir_name,
            search_alg=algo,
            scheduler=asha,
            metric=opt.score_metric,
            mode=opt.mode,
            num_samples=opt.num_trials or -1,
            time_budget_s=time_budget,
        )
    else:
        tune_config = tune.TuneConfig(
            trial_name_creator=lambda t: trial_str_creator(t, job_id),
            trial_dirname_creator=trial_dir_name,
        )

    # --- Build Tuner ---
    tuner = tune.Tuner(
        trainable=trainable_cls,
        param_space=rllib_cfg,
        run_config=air.RunConfig(
            name=exp.name,
            storage_path=storage_path,
            stop={"timesteps_total": exp.nb_timesteps},
            callbacks=[
                TBXLoggerCallback(),
                TuneCallback(
                    exp.my_log_level,
                    opt.score_metric,
                    eval_freq=rllib_cfg["evaluation_interval"],
                    heartbeat_freq=60,
                ),
            ],
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=exp.checkpoint_freq,
                checkpoint_at_end=True,
                checkpoint_score_attribute=opt.score_metric,
                num_to_keep=3,
            ),
            verbose=exp.verbose,
        ),
        tune_config=tune_config,
    )

    print_details(rllib_cfg)

    # --- Launch ---
    try:
        logger.info(f"Starting training for experiment '{exp.name}' with algorithm '{algorithm}'")
        result_grid = tuner.fit()
    finally:
        ray.shutdown()

    # --- Print checkpoint summary for each trial ---
    for i, result in enumerate(result_grid):
        if not result.error:
            checkpoints_to_json = {
                os.path.basename(checkpoint.path): {
                    k: v for k, v in metrics.get('evaluation', {}).get('custom_metrics', {}).items()
                    if not isinstance(v, list)
                }
                for checkpoint, metrics in result.best_checkpoints
            }
            with open(os.path.join(result.path, "checkpoint_results.json"), "w") as f:
                json.dump(checkpoints_to_json, f)
            try:
                eval_metrics = result.metrics.get('evaluation', {}).get('custom_metrics', {})
                if eval_metrics:
                    print(
                        Style.BOLD + f" *---- Trial {i} finished successfully ---*\n" + Style.END +
                        tabulate(
                            [[k] + list(v.values()) for k, v in checkpoints_to_json.items() if v],
                            headers=['checkpoint'] + list(eval_metrics.keys()),
                            tablefmt='rounded_grid',
                        )
                    )
                else:
                    print(Style.BOLD + f" *---- Trial {i} finished successfully (no evaluation metrics) ---*" + Style.END)
            except Exception as e:
                print("Could not print checkpoint results table: ", e)
        else:
            print(f"Trial {i} failed with error {result.error}.")

    # --- Find best result ---
    if opt.enable:
        try:
            best_result = result_grid.get_best_result(metric=opt.score_metric, mode=opt.mode)
        except RuntimeError as e:
            print(f"\n{Style.BOLD}{Style.RED}{'=' * 80}{Style.END}")
            print(f"{Style.BOLD}{Style.RED}ERROR: Could not find best trial for metric '{opt.score_metric}'{Style.END}")
            print(f"{Style.RED}Original error: {str(e)}{Style.END}")
            print(f"{Style.BOLD}{Style.RED}{'=' * 80}{Style.END}\n")
            return result_grid

        # Print best hyperparameters
        best_model_cfg = best_result.config["model"]["custom_model_config"]
        rows = [
            [f"model.custom_model_config.{k}", v]
            for k, v in best_model_cfg.items()
            if not isinstance(v, dict)
        ]
        print(f"\n{Style.BOLD}{'=' * 80}{Style.END}")
        print(f"{Style.BOLD}Best hyperparameters found:{Style.END}")
        print(tabulate(rows, headers=["Parameter", "Value"], tablefmt="rounded_grid", floatfmt=".3f"))
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            print("Corresponding checkpoint: ", checkpoint_dir)

        # Save Optuna study
        if algo is not None:
            optuna_path = os.path.join(storage_path, exp.experiment_name, f"optuna_results_{job_id}")
            study_name = f"{exp.experiment_name}_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            db_path = algo.save_study(optuna_path, study_name)
            tune_path = os.path.join(storage_path, exp.experiment_name, "tune_results")
            os.makedirs(tune_path, exist_ok=True)
            algo.save_to_dir(tune_path, f"tune_checkpoint_{job_id}")
            print(f"\n{Style.BOLD}Optuna study saved to: {db_path}{Style.END}")
            print(f"  optuna-dashboard sqlite:///{db_path}")
            print(f"{Style.BOLD}{'=' * 80}{Style.END}\n")
    else:
        try:
            best_result = result_grid.get_best_result(metric="episode_reward_mean", mode="max")
        except RuntimeError as e:
            print(f"\n{Style.BOLD}{Style.RED}ERROR: Could not find best trial — {e}{Style.END}\n")
            return result_grid
        if best_result.checkpoint is None:
            print(f"{Style.BOLD}No checkpoint available (training may have been too short to trigger evaluation).{Style.END}")
            return result_grid
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            print("Best checkpoint: ", checkpoint_dir)

    # --- Post-training analysis ---
    post_eval = exp.post_training_evaluation
    test_env_name = rllib_cfg["env_config"]["env_name"].removesuffix("_train") + "_test"
    if post_eval.enabled:
        if best_result.checkpoint is None:
            print(f"{Style.BOLD}Skipping post-training analysis: no checkpoint available.{Style.END}")
            return result_grid
        print(f"\n{Style.BOLD}{'=' * 80}{Style.END}")
        print(f"{Style.BOLD}Running post-training analysis...{Style.END}")
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            trial_path = Path(checkpoint_dir).parent
            checkpoint_name = os.path.basename(checkpoint_dir)
            num_episodes = (
                get_num_available_episodes(test_env_name)
                if post_eval.num_episodes in ("all", None)
                else int(post_eval.num_episodes)
            )
            print(f"Environment: {test_env_name}  |  Episodes: {num_episodes}")
            try:
                _run_post_training_analysis(
                    trial_path=trial_path,
                    checkpoint_name=checkpoint_name,
                    test_env_name=test_env_name,
                    num_episodes=num_episodes,
                    env_config=rllib_cfg["env_config"],
                )
                print(f"{Style.BOLD}Post-training analysis completed!{Style.END}")
            except Exception as e:
                print(f"{Style.BOLD}Warning: Post-training analysis failed: {e}{Style.END}")
                traceback.print_exc()
        print(f"{Style.BOLD}{'=' * 80}{Style.END}\n")

    return result_grid


def _get_job_id(cfg: DictConfig) -> str:
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    model_id = {None: "MLP", "gnn_model": "GNN"}.get(cfg.model.custom_model, "unknown_model")
    job_id = f"{model_id}_{slurm_job_id}"
    return job_id


def get_duration(experiment_cfg: DictConfig) -> int | None:
    """Return the wall-clock training budget in seconds, or None to run until nb_timesteps."""
    duration = experiment_cfg.duration
    if duration is None:
        logger.info(f"Run until {experiment_cfg.nb_timesteps} agent time steps.")
        return None
    if isinstance(duration, str):
        h, m = duration.split(":")
        seconds = int(h) * 3600 + int(m) * 60
    else:
        seconds = int(duration) * 60
    if seconds == 0:
        logger.info(f"Run until {experiment_cfg.nb_timesteps} agent time steps.")
        return None
    logger.info(f"Run training for {seconds}s.")
    return seconds


def trial_str_creator(trial: Trial, job_id=""):
    # Don't modify trial.trial_id as it breaks Optuna's internal tracking!
    # Just create a custom display name
    base_id = trial.trial_id.split("_")[0]
    if job_id:
        custom_id = "{}_{}".format(job_id, base_id)
    else:
        custom_id = base_id
    logger.info(f'Creating trial with ID: {custom_id}')
    return "{}_{}".format(trial.trainable_name, custom_id)


def trial_dir_name(trial: Trial):
    logger.info(f"Trial name is: {trial.custom_trial_name}")
    return "{}_{}".format(trial.custom_trial_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

def print_details(rllib_cfg: Dict[str, Any]) -> None:
    chronics_dir = rllib_cfg["env_config"].get("chronics_dir")
    if chronics_dir is None:
        chronics_dir = grid2op.get_current_local_dir()

    logger.info(f'Using reward function:      {rllib_cfg["env_config"]["grid2op_kwargs"]["reward_class"].__class__.__name__}')
    logger.info(f'Using action space:         {rllib_cfg["env_config"]["action_space"]}')
    logger.info(f'Using observation space:    {rllib_cfg["env_config"]["observation_space"]}')
    logger.info(f'Chronics are accessed from: {chronics_dir}')
    logger.info(f'Chronics are cached:        {rllib_cfg["env_config"].get("use_chronics_cache", False)}')


def _setup_ray(exp):
    # --- Init Ray ---
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"  # suppress result.json + progress.csv
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONWARNINGS"] = "ignore:Could not parse CUBLAS_WORKSPACE_CONFIG:UserWarning"
    ray.init(local_mode=exp.ray_local_mode)
    logger.info(f"Ray initialized in {'local' if exp.ray_local_mode else 'cluster'} mode.")
    logger.info(f"Ray sees GPUs: {ray.available_resources().get('GPU', 0)}")
