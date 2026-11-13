"""
Convert ranked Grid2Op actions to this repo's action-space JSON format.

This repo's `grid2op_env.action_converters.load_actions` expects a JSON list of dicts, each
passed straight to `env.action_space(action_dict)` (see e.g. `data/action_spaces/
l2rpn_case14_sandbox/medha.json`). `BaseAction.as_serializable_dict()` produces exactly that
dict shape, so the conversion is a direct serialization with no reformatting.
"""
from __future__ import annotations

import json
from pathlib import Path


from grid2op.Action import BaseAction
from grid2op.Environment import BaseEnv


def actions_to_json_dicts(actions: list[BaseAction]) -> list[dict]:
    """Serialize Grid2Op actions to this repo's action-space JSON dict format.

    Args:
        actions: Actions to serialize, e.g. from `aggregation.select_top_k_actions`.

    Returns:
        One `as_serializable_dict()` output per action, in the same order.
    """
    return [action.as_serializable_dict() for action in actions]


def export_action_space(actions: list[BaseAction], path: Path) -> None:
    """Write a list of actions to a JSON file loadable by `grid2op_env.action_converters.load_actions`.

    Args:
        actions: Actions to export.
        path: Destination JSON file, e.g. `data/action_spaces/l2rpn_wcci_2020/teacher_n1_k300.json`.

    Returns:
        None. Writes `path`, creating parent directories as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wt", encoding="utf-8") as action_space_file:
        json.dump(actions_to_json_dicts(actions), action_space_file, indent=2)


def verify_round_trip(actions: list[BaseAction], env: BaseEnv) -> bool:
    """Check that every action survives a serialize/deserialize round trip unchanged.

    Args:
        actions: Actions to check.
        env: The environment the actions belong to (used to rebuild actions from dicts).

    Returns:
        True if every rebuilt action equals the original, False otherwise.
    """
    for action, action_dict in zip(actions, actions_to_json_dicts(actions)):
        if env.action_space(action_dict) != action:
            return False
    return True
