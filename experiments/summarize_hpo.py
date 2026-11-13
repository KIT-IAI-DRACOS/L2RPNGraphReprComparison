"""
Summarize a study of experiments/pretrain_hpo.py: writes `trials.csv` (one row per trial with
parameters, objective and the best epoch's val/test metrics) and the link `best` -> run directory
of the best complete trial, which experiments/imitation_rollout.py takes as `--run-dir`.

Example:
    PYTHONPATH=$(pwd)/src python experiments/summarize_hpo.py results/junior/junior_hard
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imitation.hpo import open_study, summarize_study


def main(args: argparse.Namespace) -> None:
    study_dir = Path(args.study_dir)
    study = open_study(study_dir, args.study_name or study_dir.name)
    best_dir = summarize_study(study, study_dir)
    states = [t.state.name for t in study.trials]
    counts = ", ".join(f"{s.lower()} {states.count(s)}" for s in sorted(set(states)))
    print(f"{study.study_name}: {len(states)} trials ({counts}); best {best_dir.name} (value {study.best_value:.4f})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study_dir", help="imitation.out_dir of the study, containing study.log")
    parser.add_argument("--study-name", default=None, help="Default: name of study_dir (= experiment.name)")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(build_parser().parse_args())
