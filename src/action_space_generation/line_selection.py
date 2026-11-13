"""
Pick which power lines the N-1 Teacher should treat as contingencies (`lines_to_attack`).

`curriculumagent.teacher.teachers.teacher_n_minus_1.NMinusOneTeacher` defaults `lines_to_attack`
to a hardcoded list of line indices from `l2rpn_neurips_2020_track1_small` — meaningless on other
grids. This module replaces that default with a grid-specific estimate of how damaging each
line's outage is: run a do-nothing rollout over a handful of chronics and, at sampled
timesteps, simulate every single-line outage (N-1 contingency analysis via lightsim2grid).
Each line is scored by how far its outage pushes the *remaining* lines above their thermal
limit, summed over time:

    score[l] = Σ_t max(0, max_rho_after_outage[l](t) − 1.0)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import grid2op
import numpy as np
from grid2op.Environment import BaseEnv
from lightsim2grid import ContingencyAnalysis, LightSimBackend

# Precomputed `select_lines_to_attack(n_lines=10, n_chronics=10)` results (2026-09-30), sampling
# every step, on the train split (case14: the unsplit sandbox). Keyed by base env name: line ids
# are identical across a grid's train/val/test splits.
DEFAULT_LINES_TO_ATTACK: dict[str, list[int]] = {
    "l2rpn_case14_sandbox": [9, 8, 17, 1, 3, 19, 11, 7, 15, 2],
    "l2rpn_wcci_2020": [4, 46, 1, 3, 45, 56, 47, 12, 58, 55],
    "l2rpn_wcci_2022": [114, 173, 175, 185, 15, 111, 113, 183, 100, 174],
}


def _base_env_name(env_name: str) -> str:
    """Strip any directory and `_train`/`_val`/`_test` suffix, e.g. "~/x/l2rpn_wcci_2020_val" -> "l2rpn_wcci_2020"."""
    return re.sub(r"_(train|val|test)$", "", Path(env_name).name)


def default_lines_to_attack(env_name: str) -> list[int]:
    """Look up the precomputed `lines_to_attack` for a grid.

    Args:
        env_name: Grid2Op environment name or path, with or without a split suffix
            (e.g. "l2rpn_wcci_2020_train").

    Returns:
        A copy of the grid's entry in `DEFAULT_LINES_TO_ATTACK`.

    Raises:
        KeyError: If no default exists for this grid — run `select_lines_to_attack` instead.
    """
    base_name = _base_env_name(env_name)
    if base_name not in DEFAULT_LINES_TO_ATTACK:
        raise KeyError(
            f"No default lines_to_attack for '{base_name}' (known: {list(DEFAULT_LINES_TO_ATTACK)}). "
            "Compute them with `build_action_space.py select-lines`."
        )
    return list(DEFAULT_LINES_TO_ATTACK[base_name])


@dataclass
class OutageImpactStats:
    """Per-line N-1 outage-impact statistics accumulated over sampled timesteps.

    Attributes:
        n_line: Number of lines in the grid.
        overload_threshold: `rho` above which a remaining line counts as overloaded.
        overload_score: Per-line `Σ_t max(0, max_rho_after_outage − overload_threshold)`.
        post_outage_rho_sum: Per-line `Σ_t max_rho_after_outage`, used only to break ties
            between lines whose outage never overloads anything (common on calm grids).
        unsolvable_count: Per-line count of samples where the outage could not be solved
            (it islands a load/generator or the powerflow diverges).
        n_samples: Number of timesteps folded in so far.
    """

    n_line: int
    overload_threshold: float = 1.0
    overload_score: np.ndarray = field(init=False)
    post_outage_rho_sum: np.ndarray = field(init=False)
    unsolvable_count: np.ndarray = field(init=False)
    n_samples: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.overload_score = np.zeros(self.n_line, dtype=np.float64)
        self.post_outage_rho_sum = np.zeros(self.n_line, dtype=np.float64)
        self.unsolvable_count = np.zeros(self.n_line, dtype=np.int64)

    def update(self, post_outage_max_rho: np.ndarray) -> None:
        """Fold one timestep's N-1 results into the running statistics.

        Args:
            post_outage_max_rho: Shape `(n_line,)`; entry `l` is the maximum `rho` over all
                remaining lines after disconnecting line `l`, or NaN if unsolvable.
        """
        unsolvable = np.isnan(post_outage_max_rho)
        solved_rho = np.where(unsolvable, 0.0, post_outage_max_rho)
        self.unsolvable_count += unsolvable
        self.overload_score += np.maximum(0.0, solved_rho - self.overload_threshold)
        self.post_outage_rho_sum += solved_rho
        self.n_samples += 1

    def unsolvable_lines(self) -> list[int]:
        """Return ids of lines whose outage was unsolvable in at least one sample."""
        return [int(line_id) for line_id in np.flatnonzero(self.unsolvable_count > 0)]

    def rank_lines(self, n_lines: int) -> list[int]:
        """Return the `n_lines` lines whose outage is most damaging.

        Ranks by `overload_score`, breaking ties by `post_outage_rho_sum`. Lines in
        `unsolvable_lines()` are excluded: the teacher's `calculate_attacked_max_rho` returns
        `inf` for *every* candidate action if any attacked line ends the episode, which would
        disable its N-1 search altogether.

        Args:
            n_lines: How many line ids to return.
            exclude: Line ids that must not be returned (e.g. bridges), regardless of rank.

        Returns:
            Line ids sorted from most to least damaging outage, length
            `min(n_lines, n_line - len(unsolvable_lines()))`.
        """
        order = np.lexsort((-self.post_outage_rho_sum, -self.overload_score))
        solvable_order = order[self.unsolvable_count[order] == 0]
        return [int(line_id) for line_id in solvable_order[:n_lines]]


def format_line_report(
        stats: OutageImpactStats,
        line_ids: list[int],
        line_or_to_subid: np.ndarray,
        line_ex_to_subid: np.ndarray,
) -> str:
    """Render a table of overload score and connected substations for the given lines.

    Lines whose outage was unsolvable (excluded from the ranking) are listed below the table.

    Args:
        stats: Accumulated outage-impact statistics.
        line_ids: Line ids to report, in display order (e.g. the output of `rank_lines`).
        line_or_to_subid: Shape `(n_line,)`, substation id at each line's origin end
            (`env.line_or_to_subid`).
        line_ex_to_subid: Shape `(n_line,)`, substation id at each line's extremity end
            (`env.line_ex_to_subid`).

    Returns:
        Multi-line string, one row per line: `line  sub_or -> sub_ex  overload_score`.
    """
    rows = [f"{'line':>5}  {'substations':>12}  {'overload_score':>14}"]
    for line_id in line_ids:
        substations = f"{line_or_to_subid[line_id]} -> {line_ex_to_subid[line_id]}"
        rows.append(f"{line_id:>5}  {substations:>12}  {stats.overload_score[line_id]:>14.2f}")

    unsolvable = [
        f"{line_id} ({line_or_to_subid[line_id]} -> {line_ex_to_subid[line_id]})"
        for line_id in stats.unsolvable_lines()
    ]
    if unsolvable:
        rows.append(f"Excluded (outage islands the grid): {', '.join(unsolvable)}")
    rows.append(f"Samples: {stats.n_samples}")
    return "\n".join(rows)


def compute_post_outage_max_rho(backend: LightSimBackend, thermal_limit: np.ndarray) -> np.ndarray:
    """Simulate every single-line outage on the backend's current state.

    Args:
        backend: LightSim backend holding the current grid state (it is copied, not modified).
        thermal_limit: Shape `(n_line,)`, per-line thermal limits in A (`env.get_thermal_limit()`).

    Returns:
        Shape `(n_line,)`; entry `l` is the maximum `rho` over all lines after disconnecting
        line `l` (the disconnected line itself carries 0), or NaN if that outage islands part
        of the grid or the powerflow diverges.
    """
    analysis = ContingencyAnalysis(backend)
    try:
        analysis.add_all_n1_contingencies()
        _, amps, _ = analysis.get_flows()
    finally:
        analysis.close()
    # Any NaN in a row (unsolved contingency) propagates to NaN through `max`, as intended.
    return np.max(amps / thermal_limit, axis=1)


def _rollout_do_nothing(env: BaseEnv, stats: OutageImpactStats, n_chronics: int) -> None:
    """Step a do-nothing agent through `n_chronics` chronics, updating `stats` at sampled steps."""
    for _ in range(n_chronics):
        obs = env.reset()
        thermal_limit = env.get_thermal_limit()  # only readable while the env is not game over
        done = False
        while not done:
            if bool(obs.line_status.all()):
                stats.update(compute_post_outage_max_rho(env.backend, thermal_limit))
            obs, _, done, _ = env.step(env.action_space({}))


def select_lines_to_attack(
        env_name: str,
        n_lines: int = 10,
        n_chronics: int = 20,
        overload_threshold: float = 1.0,
        verbose: bool = False,
) -> list[int]:
    """Select the `n_lines` lines whose outage overloads the rest of the grid the most.

    Args:
        env_name: Grid2Op environment name or path (e.g. "l2rpn_wcci_2020_train").
        n_lines: Number of line ids to return.
        n_chronics: Number of chronics to roll out with a do-nothing agent.
        overload_threshold: `rho` above which a remaining line counts as overloaded.
        verbose: If True, print overload score and connected substations of each selected line.

    Returns:
        Line ids sorted from most to least damaging outage, to pass as `lines_to_attack` to
        `NMinusOneTeacher`. Lines whose outage islands the grid are never returned.
    """
    env = grid2op.make(env_name, backend=LightSimBackend())
    try:
        stats = OutageImpactStats(n_line=env.n_line, overload_threshold=overload_threshold)
        _rollout_do_nothing(env, stats, n_chronics)
        line_or_to_subid, line_ex_to_subid = env.line_or_to_subid, env.line_ex_to_subid
    finally:
        env.close()

    lines_to_attack = stats.rank_lines(n_lines)
    if verbose:
        print(format_line_report(stats, lines_to_attack, line_or_to_subid, line_ex_to_subid))
    return lines_to_attack



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-lines", type=int, default=10)
    parser.add_argument("--n-chronics", type=int, default=20)
    args = parser.parse_args()

    for grid in ["l2rpn_case14_sandbox", "l2rpn_wcci_2020_train", "l2rpn_wcci_2022_train"]:
        lines_to_attack = select_lines_to_attack(
            env_name=grid,
            n_lines=args.n_lines,
            n_chronics=args.n_chronics,
            verbose=True
        )
        print(f"Selected {len(lines_to_attack)} lines to attack for {grid}: {lines_to_attack}")
