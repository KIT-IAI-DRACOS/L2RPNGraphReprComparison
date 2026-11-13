"""
Run the Fraunhofer `curriculumagent` N-1 Teacher on a Grid2Op grid.

This bypasses `curriculumagent.teacher.teacher.n_minus_1_teacher()`, which passes its `jobs`
argument positionally into `NMinusOneTeacher.collect_n_minus_1_experience`'s `number_of_years`
parameter (a bug in curriculumagent==1.1.2). `NMinusOneTeacher` is used directly instead, with
all arguments passed by keyword.

Must run in the `curriculum` conda environment (see `environment_curriculum.yaml`), never in
the main `L2RPN` training environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import grid2op
from curriculumagent.teacher.teachers.teacher_n_minus_1 import NMinusOneTeacher
from lightsim2grid import LightSimBackend
from tqdm import tqdm

from action_space_generation.line_selection import default_lines_to_attack


@dataclass
class TeacherConfig:
    """Per-grid configuration for the N-1 Teacher run.

    Attributes:
        env_name: Grid2Op environment name or path (e.g. "l2rpn_wcci_2020_train").
        save_path: Directory (or CSV file) to append teacher_experience rows to.
        lines_to_attack: Line ids used as N-1 contingencies, from `select_lines_to_attack`.
            If None, the grid's precomputed `line_selection.DEFAULT_LINES_TO_ATTACK` are used.
        seed: Random seed for the Grid2Op environment.
        jobs: Number of parallel worker processes. -1 uses all available CPUs.
        rho_n0_threshold: Threshold, when the do-nothing agent should stop and start searching for N-1.
        rho_max_threshold: Threshold, when the N-1 search should stop and only the best greedy.
    """

    env_name: str
    save_path: Path
    lines_to_attack: list[int] | None = None
    seed: int = 42
    jobs: int = -1
    rho_n0_threshold: float = 0.95  # matches this repo's HIGH_LEVEL_AGENT activation threshold (rho_threshold)
    rho_max_threshold: float = 1.0

    def __post_init__(self) -> None:
        if self.lines_to_attack is None:
            self.lines_to_attack = default_lines_to_attack(self.env_name)


def build_teacher(config: TeacherConfig) -> NMinusOneTeacher:
    """Construct an `NMinusOneTeacher` configured for one grid.

    Args:
        config: Per-grid teacher configuration.

    Returns:
        A configured, not-yet-run `NMinusOneTeacher`.
    """
    return NMinusOneTeacher(
        lines_to_attack=config.lines_to_attack,
        rho_n0_threshold=config.rho_n0_threshold,
        rho_max_threshold=config.rho_max_threshold,
        seed=config.seed,
    )


def list_chronic_ids(env_name: str) -> list[str]:
    """List chronic subpath names for an environment, for sharding across SLURM array tasks.

    Args:
        env_name: Grid2Op environment name or path.

    Returns:
        Chronic subpath names, in the order Grid2Op assigns them (stable given a fixed dataset).
    """
    env = grid2op.make(env_name, backend=LightSimBackend())
    try:
        return list(env.chronics_handler.subpaths)
    finally:
        env.close()


def run_teacher_chronics(config: TeacherConfig, chronics_ids: list[str]) -> None:
    """Run the N-1 Teacher sequentially over one or more chronics, appending all to one shard.

    `NMinusOneTeacher.collect_n_minus_1_experience` parallelizes across chronics with a local
    `multiprocessing.Pool`, which only scales to one node's core count. This calls the same
    per-chronic worker (`n_minus_one_agent`) directly instead, so a SLURM array can shard
    chronics across many nodes/tasks. Batching several chronics into one task (rather than one
    array task per chronic) keeps the array width under a cluster's `MaxArraySize` limit for
    datasets with thousands of chronics, and amortizes curriculumagent's heavy import cost
    (tensorflow, ray) across the batch instead of paying it once per chronic.

    All chronics in the batch append to the same `config.save_path`, run sequentially within
    this one process — safe, unlike `collect_n_minus_1_experience`'s own same-file writes from
    multiple concurrent processes.

    Args:
        config: Per-grid teacher configuration. `config.save_path` should be unique per SLURM
            task (e.g. suffixed with the array task id) so concurrent tasks don't share a file.
        chronics_ids: Chronic subpath names to process in order, from `list_chronic_ids`.

    Returns:
        None. Results are appended to `config.save_path`.
    """
    config.save_path.parent.mkdir(parents=True, exist_ok=True)
    teacher = build_teacher(config)

    env_path = Path(config.env_name).expanduser()
    chronics_path = str(env_path / "chronics") if env_path.is_dir() else None

    for chronics_id in tqdm(chronics_ids):
        teacher.n_minus_one_agent(
            env_path=config.env_name,
            chronics_path=chronics_path,
            chronics_id=chronics_id,
            save_path=config.save_path,
            save_greedy=True,
            active_search=True,
            disable_opponent=True,
        )

if __name__ == "__main__":
    #print(list_chronic_ids("l2rpn_wcci_2020_train"))
    config = TeacherConfig(
        env_name="l2rpn_wcci_2020_train",
        lines_to_attack=[4, 46, 1, 3, 45, 56, 47, 12, 58, 55],
        save_path=Path("/tmp/teacher_experience.csv"),
    )
    run_teacher_chronics(config, chronics_ids=["Scenario_september_238", "Scenario_september_239"])