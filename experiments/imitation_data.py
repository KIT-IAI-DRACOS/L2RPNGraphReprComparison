"""
CLI to record imitation-learning data with the greedy Tutor in `src/imitation/tutor.py`.

Runs in the main `L2RPN` conda environment (no `curriculumagent` needed), so states, actions and
heuristics are exactly those of the RL env.

  list-chronics        List chronic ids, to size a SLURM array.
  run-tutor-chronics   Run the Tutor on a batch of chronics, one `.npz` per chronic — the unit of
                       work for one task of slurm/unicluster/imitation_tutor_array.sh.
  build-dataset        Convert the Tutor's raw per-chronic output into the shared dataset format of
                       `src/imitation/dataset.py`, building the policy targets (the paper: --target hard).
  split                Assign the grid's chronics to train/val/test (stratified by month), shared by
                       every dataset so all comparisons use the same held-out chronics.

Example:
    PYTHONPATH=$(pwd)/src python experiments/imitation_data.py run-tutor-chronics --grid case36 \\
        --action-space data/action_spaces/l2rpn_wcci_2020/teacher_n1_k208.json \\
        --chronic-ids Scenario_april_000 --save-dir results/tutor/smoketest/experience
    PYTHONPATH=$(pwd)/src python experiments/imitation_data.py build-dataset \\
        --experience-dir results/tutor/<experiment>/experience --target hard \\
        --out data/imitation/case36_k208_greedy_hard.npz
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imitation.dataset import HARD, SOFT, from_greedy_tutor, make_split, save_split
from imitation.tutor import (
    DO_NOTHING_MARGIN,
    RHO_THRESHOLD,
    TutorConfig,
    list_chronic_ids,
    run_tutor_chronics,
)

# Grid alias -> Grid2Op env base name, matching configs/rllib/env/case36.yaml.
GRID_ENV_NAMES = {
    "case36": "l2rpn_wcci_2020",
}


def _env_name(grid: str, split: str) -> str:
    """Resolve a grid alias + split (train/val/test) to a Grid2Op environment name."""
    return f"{GRID_ENV_NAMES[grid]}_{split}"


def cmd_list_chronics(args: argparse.Namespace) -> None:
    """Print chronic ids, one per line."""
    for chronic_id in list_chronic_ids(_env_name(args.grid, args.split)):
        print(chronic_id)


def cmd_run_tutor_chronics(args: argparse.Namespace) -> None:
    """Run the Tutor on a batch of chronics."""
    config = TutorConfig(
        env_name=_env_name(args.grid, args.split),
        action_space_path=Path(args.action_space),
        save_dir=Path(args.save_dir),
        seed=args.seed,
        rho_threshold=args.rho_threshold,
        do_nothing_margin=args.do_nothing_margin,
        line_reco=not args.no_line_reco,
        max_steps=args.max_steps,
    )
    run_tutor_chronics(config, chronic_ids=args.chronic_ids)


def cmd_build_dataset(args: argparse.Namespace) -> None:
    """Convert raw Tutor output into the shared dataset format."""
    data = from_greedy_tutor(Path(args.experience_dir), target=args.target, temperature=args.temperature)
    data.save(Path(args.out))
    n_dn = int((data.policy_target.argmax(axis=1) == 0).sum())
    print(f"Wrote {len(data)} samples from {len(set(data.chronic))} chronics "
          f"({n_dn} with do-nothing as top target) to {args.out}: {data.metadata}")


def cmd_split(args: argparse.Namespace) -> None:
    """Write a chronic-level train/val/test split."""
    split = make_split(list_chronic_ids(_env_name(args.grid, args.split)), tuple(args.fractions), args.seed)
    save_split(split, Path(args.out))
    print(f"Wrote split to {args.out}: " + ", ".join(f"{k} {len(v)}" for k, v in split.items()))


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_chronics = subparsers.add_parser("list-chronics", help="List chronic ids of a grid/split")
    list_chronics.add_argument("--grid", required=True, choices=GRID_ENV_NAMES)
    list_chronics.add_argument("--split", default="train", choices=["train", "val", "test"])
    list_chronics.set_defaults(func=cmd_list_chronics)

    run_tutor = subparsers.add_parser("run-tutor-chronics", help="Run the Tutor on a batch of chronics")
    run_tutor.add_argument("--grid", required=True, choices=GRID_ENV_NAMES)
    run_tutor.add_argument("--split", default="train", choices=["train", "val", "test"])
    run_tutor.add_argument("--action-space", required=True, help="Action-space JSON, as loaded by the RL env")
    run_tutor.add_argument("--chronic-ids", nargs="+", required=True)
    run_tutor.add_argument("--save-dir", required=True, help="Directory for the per-chronic .npz files")
    run_tutor.add_argument("--seed", type=int, default=42)
    run_tutor.add_argument("--rho-threshold", type=float, default=RHO_THRESHOLD)
    run_tutor.add_argument("--do-nothing-margin", type=float, default=DO_NOTHING_MARGIN)
    run_tutor.add_argument("--no-line-reco", action="store_true", help="Disable reconnection_rule (env line_reco: false)")
    run_tutor.add_argument("--max-steps", type=int, default=None, help="Cap steps per chronic (smoke tests)")
    run_tutor.set_defaults(func=cmd_run_tutor_chronics)

    build = subparsers.add_parser("build-dataset", help="Convert raw Tutor output into the shared dataset format")
    build.add_argument("--experience-dir", required=True, help="Directory of per-chronic Tutor files")
    build.add_argument("--target", default=HARD, choices=[HARD, SOFT])
    build.add_argument("--temperature", type=float, default=0.0, help="Soft-target temperature in rho units")
    build.add_argument("--out", required=True, help="Output .npz path")
    build.set_defaults(func=cmd_build_dataset)

    split = subparsers.add_parser("split", help="Chronic-level train/val/test split, stratified by month")
    split.add_argument("--grid", required=True, choices=GRID_ENV_NAMES)
    split.add_argument("--split", default="train", choices=["train", "val", "test"], help="Grid2Op env split to read chronics from")
    split.add_argument("--fractions", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--out", required=True, help="Output .json path")
    split.set_defaults(func=cmd_split)

    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cli_args = build_parser().parse_args()
    cli_args.func(cli_args)
