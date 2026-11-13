"""
Turn N-1 Teacher experience CSVs into a ranked, filtered action list.

Thin wrapper around `curriculumagent.teacher.collect_teacher_experience`'s existing
frequency-ranking logic (`read_experience`, `filter_good_experience`, `rank_actions_simple`),
plus the substation-distribution analysis called for in the experiment note
("Analyze how the found useful actions are distributed across substations").
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from curriculumagent.teacher.collect_teacher_experience import (
    filter_good_experience,
    rank_actions_simple,
    read_experience,
)
from curriculumagent.teacher.submodule.common import affected_substations
from curriculumagent.teacher.submodule.encoded_action import EncodedTopologyAction
from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv


def load_experience(experience_csv_paths: list[Path]) -> pd.DataFrame:
    """Load and filter teacher_experience CSVs (one or more SLURM-array shards).

    Args:
        experience_csv_paths: Paths to `teacher_experience.csv` shards, e.g. one per chronic.

    Returns:
        Concatenated experience, keeping only rows where the action improved `rho` by more
        than 2% (`filter_good_experience`'s threshold).
    """
    data = read_experience(experience_csv_paths)
    return filter_good_experience(data)


def action_frequency(data: pd.DataFrame) -> pd.Series:
    """Count how often each unique encoded action was selected.

    Args:
        data: Filtered experience, as returned by `load_experience`.

    Returns:
        Series mapping encoded action string -> selection count, sorted descending. Plot this
        (`.values`) to pick `k` from the frequency curve's knee, per the experiment note.
    """
    return data["best_action"].value_counts()


def select_top_k_actions(data: pd.DataFrame, env: BaseEnv, k: int) -> list[BaseAction]:
    """Select and decode the `k` most frequently chosen actions.

    Args:
        data: Filtered experience, as returned by `load_experience`.
        env: The environment the actions belong to (used to decode them).
        k: Number of actions to keep.

    Returns:
        The `k` most frequent actions as Grid2Op `BaseAction` objects, most frequent first.
    """
    return rank_actions_simple(data, env, best_n=k, plot_choice=False)


def substation_distribution(actions: list[BaseAction]) -> pd.Series:
    """Count how many selected actions affect each substation.

    Answers the experiment note's "how are useful actions distributed across substations".

    Args:
        actions: Decoded actions, e.g. from `select_top_k_actions`.

    Returns:
        Series mapping substation id -> number of actions affecting it, sorted descending.
        An action affecting multiple substations is counted once per substation.
    """
    substation_ids = [sub_id for action in actions for sub_id in affected_substations(action)]
    return pd.Series(substation_ids).value_counts()


def decode_actions(encoded: list[str], env: BaseEnv) -> list[BaseAction]:
    """Decode a list of base64-encoded action strings back to Grid2Op actions.

    Args:
        encoded: Encoded action strings, e.g. the index of `action_frequency`'s result.
        env: The environment the actions belong to.

    Returns:
        Decoded Grid2Op actions, in the same order.
    """
    return [EncodedTopologyAction.decode_action(act_string, env) for act_string in encoded]
