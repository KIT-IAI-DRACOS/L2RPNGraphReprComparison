"""
Graph ablation experiment.

Each method is evaluated twice with the same activation threshold:

  baseline  — original graph observations
  ablated   — src/dst node indices of valid edges independently permuted
              per episode, breaking graph structure while preserving node
              features and edge count

Both conditions use a low activation threshold (default 0.0) so the RL
agent acts at every step, making the GNN's structural dependence visible
in survival statistics.  At the default training threshold (~0.95) the
heuristic dominates and the GNN is called too rarely to affect outcome.

Note: evaluating at threshold=0.0 is out-of-distribution for agents
trained with threshold > 0.  Label results accordingly in the paper.

Results are saved under experiments/survival/observation_spaces/ablation/.
"""

# ── Project root ───────────────────────────────────────────────────────────────
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import argparse
import json
import grid2op
import gymnasium as gym
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from grid2op.Observation import BaseObservation
from lightsim2grid import LightSimBackend
from tqdm import tqdm

from core.evaluate import evaluate_agent
from core.loading import AgentSpec, load_agent_from_spec
from grid2op_env.observation_converter import (
    ObservationConverter, T,
    EDGE_INDEX, EDGE_MASK, NODES,
)

RESULTS = ROOT / "experiments" / "survival" / "observation_spaces" / "ablation_gat_conv"
RESULTS.mkdir(exist_ok=True, parents=True)

# Canonical mapping: display name → obs-space directory name (as used in training).
_OBS_SPACE_DIRS = {
    "Default":          "default",
    "Elements":         "elements",
    "Elements + LODF":  "elements_lodf",
    "Heterogeneous":    "heterogenous",
    "LODF":             "lodf",
    "PTDF":             "ptdf",
    "Substation":       "substation",
    "Substation + PTDF":"substation_ptdf",
    "Substation + Zbus":"substation_zbus",
    "Zbus":             "zbus",
}


def _build_methods(experiment: str) -> dict:
    """Build the METHODS dict for a given experiment folder name.

    :param experiment: Name of the results sub-directory, e.g.
        ``"2026_08_17_compare_graph_obs_spaces_IEEE14"``.
    :return: Dict mapping display name → config dict with a ``"glob"`` key.
    """
    return {
        name: {"glob": f"results/{experiment}/{obs_dir}/CustomPPO*/"}
        for name, obs_dir in _OBS_SPACE_DIRS.items()
    }


class GraphAblationWrapper(ObservationConverter):
    """Wraps a graph observation converter and breaks graph connectivity.

    Node features and edge count are preserved. Each episode, two independent
    node-index permutations are drawn and applied to the source and destination
    rows of EDGE_INDEX respectively. This maps every message to a random
    (wrong) node pair, breaking the graph signal while keeping the feature
    distribution intact.

    Args:
        graph_observation_converter: the underlying converter to wrap
        seed: RNG seed for reproducible shuffling
    """

    def __init__(self, graph_observation_converter: ObservationConverter, seed: int = 42):
        self.converter = graph_observation_converter
        self.rng = np.random.default_rng(seed=seed)
        num_nodes = graph_observation_converter.observation_space[NODES].shape[0]
        self._perm_src = np.arange(num_nodes)
        self._perm_dst = np.arange(num_nodes)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.converter.observation_space

    def reset_obs(self) -> None:
        self.converter.reset_obs()
        num_nodes = len(self._perm_src)
        self._perm_src = self.rng.permutation(num_nodes)
        self._perm_dst = self.rng.permutation(num_nodes)
        pass  # permutations logged at debug level if needed

    def to_gym(self, g2op_obs: BaseObservation) -> T:
        """Convert observation and apply independent per-episode node permutations.

        Source and destination node indices of valid (non-padded) edges are
        remapped through two independent random permutations of [0, N-1].
        Padding slots (EDGE_MASK == 0) are left untouched. This sends every
        message to a wrong (src, dst) pair while preserving edge count,
        EDGE_MASK, and node features — a clean test of whether the policy
        relies on graph structure.

        The same pair of permutations is held for the entire episode (drawn in
        reset_obs) so the rewired topology is consistent across timesteps.

        Args:
            g2op_obs: raw Grid2Op observation

        Returns:
            gym observation with scrambled edge connectivity
        """
        obs = self.converter.to_gym(g2op_obs)
        valid = obs[EDGE_MASK].astype(bool)  # [E_max] — True for non-padded edges
        ei = obs[EDGE_INDEX].copy()          # [2, E_max]
        ei[0, valid] = self._perm_src[ei[0, valid]]   # remap src of valid edges only
        ei[1, valid] = self._perm_dst[ei[1, valid]]   # remap dst of valid edges only
        obs[EDGE_INDEX] = ei
        return obs

    def normalize(self, gym_obs: T) -> T:
        return self.converter.normalize(gym_obs)

    def close(self) -> None:
        self.converter.close()


def best_checkpoint(path: Path, metric: str = "grid2op_end_mean") -> tuple[str, float]:
    """Return the checkpoint name with the highest value for `metric`.

    Args:
        path: path to checkpoint_results.json
        metric: metric key to maximise

    Returns:
        (checkpoint_name, metric_value) for the best checkpoint
    """
    data = json.loads(path.read_text())
    return max(
        ((name, stats[metric]) for name, stats in data.items()),
        key=lambda item: item[1],
    )


def evaluate_condition(
    method: str,
    cfg: dict,
    condition: str,
    activation_threshold: float,
    num_episodes: int,
    seed: int,
) -> None:
    """Evaluate one (method, condition) pair.

    Args:
        method: display name of the method
        cfg: config dict with 'glob' key
        condition: 'baseline' (no ablation) or 'ablated' (shuffled edges)
        activation_threshold: rho threshold for RL activation; 0.0 forces
            the RL agent to act every step
        num_episodes: number of evaluation episodes
        seed: RNG seed passed to GraphAblationWrapper (ablated only)
    """
    dirs = sorted(ROOT.glob(cfg["glob"]))
    if not dirs:
        print(f"No checkpoints found for {method}")
        return
    d = dirs[0]
    try:
        agent, env, gym_env = load_agent_from_spec(
            AgentSpec(
                name=method,
                load_path=d,
                checkpoint_name=best_checkpoint(d / "checkpoint_results.json")[0],
            )
        )
        if not env.name.endswith("_test"):
            env = grid2op.make(
                env.name.removesuffix("_val").removesuffix("_train") + "_test",
                backend=LightSimBackend()
            )

        agent.activation_thresh = activation_threshold

        if condition == "ablated":
            gym_env.observation_converter = GraphAblationWrapper(
                gym_env.observation_converter, seed=seed
            )

        evaluate_agent(
            agent=agent,
            env=env,
            path_results=RESULTS / f"{method}_{condition}",
            num_episodes=num_episodes,
            max_episode_length=8064,
            verbose=False,
        )
    except Exception as e:
        print(f"Error evaluating {method} ({condition}):\n{e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", type=str, required=True,
        help=(
            "Results sub-directory name, e.g. "
            "'2026_08_17_compare_graph_obs_spaces_IEEE14'. "
            "Checkpoints are expected at results/<experiment>/<obs_space>/CustomPPO*/"
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=len(_OBS_SPACE_DIRS) * 2,
        help="Number of parallel worker processes (default: one per condition)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="RL activation threshold (default: 0.0 — act every step)",
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Episodes per condition (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for GraphAblationWrapper (default: 42)",
    )
    args = parser.parse_args()

    METHODS = _build_methods(args.experiment)

    tasks = [
        (method, cfg, condition)
        for method, cfg in METHODS.items()
        for condition in ("baseline", "ablated")
    ]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                evaluate_condition, method, cfg, condition,
                args.threshold, args.episodes, args.seed,
            ): (method, condition)
            for method, cfg, condition in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Conditions"):
            method, condition = futures[future]
            exc = future.exception()
            if exc:
                print(f"Error in {method} ({condition}): {exc}")


if __name__ == "__main__":
    main()
