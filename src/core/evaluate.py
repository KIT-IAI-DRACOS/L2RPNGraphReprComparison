"""
Script to evaluate from agent on Grid2Op environment.
"""
import json
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Environment import Environment
from grid2op.Runner import Runner
from grid2op.Runner.runner import runner_returned_type

from core import constants
from core.constants import RL_POLICY
from core.loading import load_config, preprocess_config, load_rllib_agent

from visualization import get_evaluation_metrics, visualize_agent_survival

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_rllib_checkpoint(
        checkpoint_path: Path,
        policy_name: str = RL_POLICY,
        checkpoint_name: str = "checkpoint_000000",
        env_name_override: str = None,
        num_episodes: int = 50,
        max_episode_length: Optional[int] = None,
        visualize: bool = True,
        save_to_path: Optional[Path] = None,
        seed: Optional[int] = None,
):
    """
    Evaluate an RLlib checkpoint on a Grid2Op environment.

    :param checkpoint_path: Path to the experiment directory containing checkpoints
    :param policy_name: Name of the policy (defaults to RL_POLICY specified in core/constants.py)
    :param checkpoint_name: Name of the checkpoint folder (default: "checkpoint_000000")
    :param env_name_override: Override environment name for evaluation (default: None, uses params.json)
    :param num_episodes: Number of evaluation episodes (default: 50)
    :param visualize: Whether to show visualization after evaluation (default: True)
    :param max_episode_length: the maximum length up to which episodes are played
    :param save_to_path: Optional path to save results (default: None, saves in checkpoint directory)
    :param seed: Seed used for env_seeds. If None, falls back to constants.SEED at call time.
    :return: Path to results directory
    """
    params = load_config(checkpoint_path)
    params = preprocess_config(params)

    env_config = params["env_config"]

    # Results path
    results_path = Path(checkpoint_path) / "evaluations" if save_to_path is None else save_to_path
    results_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading agent from: {checkpoint_path}")
    logger.info(f"Checkpoint: {checkpoint_name}")
    logger.info(f"Policy: {policy_name}")

    # Load the agent and keep gym_wrapper alive to prevent double-close
    agent, g2op_env, gym_wrapper = load_rllib_agent(
        checkpoint_path=checkpoint_path,
        policy_name=policy_name,
        checkpoint_name=checkpoint_name,
        env_name=env_name_override if env_name_override else env_config["env_name"],
        env_config=env_config
    )

    logger.info(f"Agent loaded successfully!")

    # Evaluate the agent
    evaluate_agent(
        agent=agent,
        env=g2op_env,
        path_results=results_path,
        num_episodes=num_episodes,
        max_episode_length=max_episode_length,
        verbose=True,
        seed=seed,
    )

    logger.info("Evaluation completed!")

    # Load and visualize results
    if visualize:
        metrics = get_evaluation_metrics(results_path, "RLlib Agent")
        visualize_agent_survival([metrics], show=True)

    logger.info(f"Results saved to: {results_path}")

    return results_path


def evaluate_agent(agent: BaseAgent, env: Environment, path_results: Path, num_episodes: int,
                   max_episode_length: Optional[int] = None, verbose=True,
                   seed: Optional[int] = None) -> List[runner_returned_type]:
    """
    Runs an agent on an environment for evaluation.
    :param agent: The agent
    :param env: the environment
    :param num_episodes: the number of episodes to run
    :param path_results: where to store the results
    :param max_episode_length: the maximum number of steps to take per episode
    :param verbose: print extra explanatory or diagnostic information
    :param seed: Seed used for env_seeds. If None, falls back to constants.SEED at call time.
    :return: the evaluation results from the runner
    """
    logging.getLogger("grid2op.Environment.baseEnv.grid2op_Runner").disabled = True
    runner = Runner(**env.get_params_for_runner(), agentInstance=agent, agentClass=None)
    path_results.mkdir(exist_ok=True, parents=True)
    env_seed = seed if seed is not None else constants.SEED
    res = runner.run(
        nb_episode=num_episodes,
        max_iter=max_episode_length,
        path_save=path_results,
        add_detailed_output=True,
        pbar=verbose,
        env_seeds=[env_seed] * num_episodes,
    )

    if verbose:
        # print results
        _print_runner_results(res)

    # Compute and store summary metrics across episodes
    _store_summary_metrics(res=res, path_results=path_results, verbose=verbose)

    if verbose:
        logger.info(f"Evaluation results are stored in: {path_results}")

    return res


def _print_runner_results(res: List[runner_returned_type]):
    for _, chron_id, cum_reward, nb_time_step, max_ts, data in res:
        logger.info(f"Chronics: '{chron_id}', Return: {cum_reward:.2f}, "
                    f"Survival Duration: {nb_time_step:.0f} / {max_ts:.0f}, "
                    f"Per-step-reward: {np.nan_to_num(data.rewards).mean():.2f} ± {np.nan_to_num(data.rewards).std():.2f}")


def _store_summary_metrics(res: List[runner_returned_type], path_results: Path, verbose: bool = True) -> None:
    """
    Compute averaged Completed Episodes % and Survived Steps % across episodes and store them in a summary JSON.

    Completed Episodes %: fraction of episodes that reached their max allowed steps (nb_time_step == max_ts) * 100.
    Survived Steps %: average over episodes of (nb_time_step / max_ts) * 100.

    :param res: list of runner returns
    :param path_results: the folder in which to store the summary_metrics.json
    :param verbose: print extra explanatory or diagnostic information
    """
    # only include episodes with positive max_timesteps
    res = [ep_info for ep_info in res if ep_info[4] > 0]

    if not len(res):
        return

    completed_flags = []
    survived_ratios = []

    for _, _, _, nb_time_step, max_ts, _ in res:
        completed_flags.append(1.0 if nb_time_step >= max_ts else 0.0)
        survived_ratios.append(float(nb_time_step) / float(max_ts))

    completed_episodes_frac = (sum(completed_flags) / float(len(completed_flags)))
    survived_steps_frac = float(np.mean(survived_ratios))

    summary = {
        "episodes": len(res),
        "completed_episodes_pct": round(completed_episodes_frac * 100, 4),
        "survived_steps_pct": round(survived_steps_frac * 100, 4)
    }

    # Persist a summary file in the root result folder
    path_results.mkdir(parents=True, exist_ok=True)
    summary_path = Path(path_results, "summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    # Log a brief summary
    if verbose:
        logger.info(
            f"Summary metrics: Completed Episodes: {completed_episodes_frac * 100:.2f}%, Survived Steps: {survived_steps_frac * 100:.2f}%")


def main():
    """Main evaluation script when running as standalone."""
    import argparse
    from core.train import get_num_available_episodes

    parser = argparse.ArgumentParser(description="Evaluate an RLlib checkpoint on a Grid2Op environment.")
    parser.add_argument("--checkpoint-path", type=Path, required=True,
                        help="Path to the trial directory containing checkpoint_* subdirectories.")
    parser.add_argument("--checkpoint-name", type=str, default=None,
                        help="Checkpoint folder name (e.g. checkpoint_000002). Defaults to the latest found.")
    parser.add_argument("--env-name", type=str, required=True,
                        help="Grid2Op environment name for evaluation (e.g. l2rpn_wcci_2020_test).")
    parser.add_argument("--num-episodes", default="all",
                        help="Number of evaluation episodes, or 'all' to use every available chronic. Default: all.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed passed to the runner.")
    parser.add_argument("--save-to", type=Path, default=None,
                        help="Directory to save evaluation results (default: <checkpoint_path>/evaluations).")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path

    checkpoint_name = args.checkpoint_name
    if checkpoint_name is None:
        candidates = sorted(checkpoint_path.glob("checkpoint_*"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint_* directories found in {checkpoint_path}")
        checkpoint_name = candidates[-1].name
        logger.info(f"Auto-selected latest checkpoint: {checkpoint_name}")

    num_episodes = (
        get_num_available_episodes(args.env_name)
        if args.num_episodes == "all"
        else int(args.num_episodes)
    )

    evaluate_rllib_checkpoint(
        checkpoint_path=checkpoint_path,
        policy_name=RL_POLICY,
        checkpoint_name=checkpoint_name,
        env_name_override=args.env_name,
        num_episodes=num_episodes,
        visualize=False,
        seed=args.seed,
        save_to_path=args.save_to,
    )


if __name__ == "__main__":
    main()
