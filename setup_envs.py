"""
Download the Grid2Op environments and split them into the train, val and test envs of the paper.

The val and test chronics are fixed (data/splits/<env>.json), so every user gets the paper's evaluation
episodes; all other chronics form the train env. Grid2Op's `train_val_split_random` would draw a new split
on every call.

Creates `<env>_train`, `<env>_val` and `<env>_test` in Grid2Op's data directory
(`grid2op.get_current_local_dir()`). Existing split envs are left untouched: delete them first to
recreate them.
"""
import json
import os
from pathlib import Path

import grid2op

SPLITS_DIR = Path(__file__).parent / "data" / "splits"
ENVS = {"l2rpn_case14_sandbox": "14 bus", "l2rpn_wcci_2020": "36 bus"}


def split_env(env_name: str) -> None:
    """Download `env_name` (if needed) and create its train/val/test envs from data/splits/<env_name>.json."""
    existing = [f"{env_name}_{s}" for s in ("train", "val", "test")
                if os.path.exists(os.path.join(grid2op.get_current_local_dir(), f"{env_name}_{s}"))]
    if existing:
        print(f"{env_name}: {', '.join(existing)} already exist(s), skipped (delete to recreate the split).")
        return
    split = json.loads((SPLITS_DIR / f"{env_name}.json").read_text())
    env = grid2op.make(env_name)
    env.train_val_split(val_scen_id=split["val"], test_scen_id=split["test"],
                        add_for_train="train", add_for_val="val", add_for_test="test")
    env.close()
    print(f"{env_name}: split into train / val ({len(split['val'])}) / test ({len(split['test'])}).")


if __name__ == "__main__":
    for name, label in ENVS.items():
        split_env(name)
        print(f"{label} system ready!")
