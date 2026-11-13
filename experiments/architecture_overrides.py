"""
Print the Hydra overrides that give experiments/train.py the network and observation space of a Junior run
(`imitation.senior.architecture_overrides`), shell-quoted on one line, for the Senior SLURM chain
(slurm/unicluster/graph_comparison_36bus_senior.sh).

Example:
    PYTHONPATH=$(pwd)/src python experiments/architecture_overrides.py results/junior/graph36_substation/best/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imitation.senior import architecture_overrides, shell_overrides


def parse_args() -> argparse.Namespace:
    """Command-line arguments: Junior config path and dropout replacement."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="config.yaml of the Junior run")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout_prob of the RL model (default 0)")
    parser.add_argument("--keep-dropout", action="store_true", help="keep the Junior's dropout_prob")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Print the overrides for `args.config`."""
    dropout = None if args.keep_dropout else args.dropout
    print(shell_overrides(architecture_overrides(OmegaConf.load(args.config), dropout)))


if __name__ == "__main__":
    main(parse_args())
