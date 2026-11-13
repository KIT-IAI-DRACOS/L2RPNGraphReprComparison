"""
Shared per-episode table, so survival can be compared across agents (Teacher, Tutor, Junior,
Senior, RL policies) with one loader.

One row per (agent, chronic) episode, stored as CSV:
    agent           str    Label for plots, e.g. "teacher", "tutor", "junior", "senior".
    chronic         str    Chronic name (not path).
    seed            int    Env seed used for this chronic; -1 if unknown. Equal seeds on the same
                           chronic mean the same random maintenance schedule.
    steps_survived  int    Last step reached. For a truncated episode, a lower bound.
    max_step        int    Chronic length.
    survived        bool   steps_survived == max_step.
    truncated       bool   The run was cut off externally (e.g. SLURM walltime), not by a game over.
    n_decisions     int    States in which the agent was asked to act (rho above threshold); -1 if unknown.
    n_actions       int    Decisions in which the agent's own topology action (not do-nothing) was executed;
                           heuristic line reconnections alone do not count. -1 if unknown.
    n_illegal       int    Decisions in which the agent chose a topology action that the env rejected as
                           illegal (grid2op then plays do-nothing). -1 if unknown.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

EPISODE_COLUMNS = ["agent", "chronic", "seed", "steps_survived", "max_step", "survived", "truncated", "n_decisions",
                   "n_actions", "n_illegal"]
# Columns added later; older tables get them as -1 (unknown) when loaded.
_COUNT_COLUMNS = ("n_decisions", "n_actions", "n_illegal")
STEPS_PER_DAY = 288  # 5-minute steps

# Teacher (curriculumagent) log lines: "[<chronic>] Search stopped at <step>/<max_step>" at the end of
# a chronic, "[<chronic>] (<step>/<max_step>) ..." while it runs.
_TEACHER_STOP = re.compile(r"\[(?P<chronic>[^\]]+)\] Search stopped at (?P<step>\d+)/(?P<max>\d+)")
_TEACHER_STEP = re.compile(r"\[(?P<chronic>[^\]]+)\] \((?P<step>\d+)/(?P<max>\d+)\)")


def make_episodes(rows: list[dict]) -> pd.DataFrame:
    """Build an episode table from row dicts, filling `survived` and defaults.

    Args:
        rows: Dicts with at least agent, chronic, steps_survived, max_step.

    Returns:
        DataFrame with exactly `EPISODE_COLUMNS`.
    """
    df = pd.DataFrame(rows)
    for col, default in (("seed", -1), ("truncated", False), *((c, -1) for c in _COUNT_COLUMNS)):
        df[col] = df[col].fillna(default).astype(type(default)) if col in df else default
    df["survived"] = df["steps_survived"] >= df["max_step"]
    return df[EPISODE_COLUMNS]


def save_episodes(df: pd.DataFrame, path: Path) -> None:
    """Write an episode table as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df[EPISODE_COLUMNS].to_csv(path, index=False)


def load_episodes(*paths: Path) -> pd.DataFrame:
    """Read and concatenate episode tables, e.g. one per agent, for a joint plot.

    Tables written before a count column existed get it as -1 (unknown).
    """
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for col in _COUNT_COLUMNS:
        df[col] = df[col].fillna(-1).astype(int) if col in df else -1
    return df[EPISODE_COLUMNS]


@dataclass
class SurvivalMetrics:
    """Survival of one agent over a set of episodes (one row of a survival table).

    Attributes:
        chronics: Number of episodes.
        full_survival_pct: Share of chronics survived to the end, in %.
        full_survival_se_pct: Its standard error sqrt(p(1-p)/n), in %.
        mean_survival_pct: Mean over chronics of steps survived / chronic length, in %.
        decisions_per_day: Decisions (max rho above the threshold) per day survived.
        actions_per_day: Executed own topology actions per day survived.
        illegal_per_day: Illegal topology actions chosen per day survived.
    """

    chronics: int
    full_survival_pct: float
    full_survival_se_pct: float
    mean_survival_pct: float
    decisions_per_day: float
    actions_per_day: float
    illegal_per_day: float


def per_day(episodes: pd.DataFrame, column: str) -> float:
    """Sum of a count column per day survived (288 steps); NaN if a table predates the column (-1).

    The ratio of sums over all chronics, not the mean of per-chronic rates, which chronics that end
    after a few hours would dominate.
    """
    if (episodes[column] < 0).any():
        return np.nan
    return episodes[column].sum() / (episodes["steps_survived"].sum() / STEPS_PER_DAY)


def survival_metrics(episodes: pd.DataFrame) -> SurvivalMetrics:
    """Survival table row of an episode table (`EPISODE_COLUMNS`, one agent and mode)."""
    full = episodes["survived"].mean()
    return SurvivalMetrics(
        chronics=len(episodes),
        full_survival_pct=100 * full,
        full_survival_se_pct=100 * np.sqrt(full * (1 - full) / len(episodes)),
        mean_survival_pct=100 * (episodes["steps_survived"] / episodes["max_step"]).mean(),
        decisions_per_day=per_day(episodes, "n_decisions"),
        actions_per_day=per_day(episodes, "n_actions"),
        illegal_per_day=per_day(episodes, "n_illegal"),
    )


@dataclass
class SurvivalOverSeeds:
    """Survival of one agent summarized over training seeds (one row of a multi-seed survival table).

    Attributes:
        seeds: Number of seeds.
        steps_pct: Mean over seeds of the step survival (`SurvivalMetrics.mean_survival_pct`), in %.
        steps_sd: Its sample standard deviation over seeds (NaN for one seed), in %.
        episodes_pct: Mean over seeds of the episode survival (`SurvivalMetrics.full_survival_pct`), in %.
        episodes_sd: Its sample standard deviation over seeds (NaN for one seed), in %.
    """

    seeds: int
    steps_pct: float
    steps_sd: float
    episodes_pct: float
    episodes_sd: float


def survival_per_seed(folders: dict[int, Path], tag: str) -> pd.DataFrame:
    """`SurvivalMetrics` of each seed's rollout folder on one env split.

    Args:
        folders: Training seed -> folder with the episode tables `episodes_<tag>_*.csv` (one per array task).
        tag: Env split tag of the file names, e.g. "valenv" or "testenv".

    Returns:
        One row per seed with at least one file (index `seed`, columns: the `SurvivalMetrics` fields);
        empty with these columns if no seed has files.
    """
    rows = {seed: asdict(survival_metrics(load_episodes(*files)))
            for seed, folder in folders.items() if (files := sorted(folder.glob(f"episodes_{tag}_*.csv")))}
    columns = [f.name for f in fields(SurvivalMetrics)]
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns).rename_axis("seed")


def survival_over_seeds(per_seed: pd.DataFrame) -> SurvivalOverSeeds:
    """Mean and sample SD over seeds of step and episode survival.

    Args:
        per_seed: One row per seed with `mean_survival_pct` and `full_survival_pct`, e.g. from
            `survival_per_seed`.
    """
    return SurvivalOverSeeds(
        seeds=len(per_seed),
        steps_pct=per_seed["mean_survival_pct"].mean(),
        steps_sd=per_seed["mean_survival_pct"].std(ddof=1),
        episodes_pct=per_seed["full_survival_pct"].mean(),
        episodes_sd=per_seed["full_survival_pct"].std(ddof=1),
    )


def teacher_episodes_from_logs(log_dir: Path, agent: str = "teacher", seed: int = -1) -> pd.DataFrame:
    """Recover per-chronic survival of a Teacher array run from its `shard_*.err` logs.

    A chronic without a "Search stopped" line was cut off by the walltime; its last logged step
    is kept as a lower bound with `truncated=True`. The Teacher logs only states it acts in, so
    that bound can be far below the step the run actually reached.

    Args:
        log_dir: Teacher experiment directory (e.g. results/teacher/2026_09_17_teacher_n1_case36).
        agent: Label for the `agent` column.
        seed: Seed the Teacher used for every chronic, if known.

    Returns:
        Episode table; `n_decisions` is unknown (-1).
    """
    stopped: dict[str, tuple[int, int]] = {}
    last_seen: dict[str, tuple[int, int]] = {}
    for log in sorted(log_dir.glob("*.err")):
        for line in log.read_text(errors="replace").splitlines():
            if m := _TEACHER_STOP.search(line):
                stopped[m["chronic"]] = (int(m["step"]), int(m["max"]))
            elif m := _TEACHER_STEP.search(line):
                last_seen[m["chronic"]] = (int(m["step"]), int(m["max"]))
    rows = [{"agent": agent, "chronic": c, "seed": seed, "steps_survived": s, "max_step": m, "truncated": False}
            for c, (s, m) in stopped.items()]
    rows += [{"agent": agent, "chronic": c, "seed": seed, "steps_survived": s, "max_step": m, "truncated": True}
             for c, (s, m) in last_seen.items() if c not in stopped]
    return make_episodes(rows).sort_values("chronic", ignore_index=True)
