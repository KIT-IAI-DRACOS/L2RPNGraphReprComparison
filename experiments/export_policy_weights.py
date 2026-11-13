"""
Export the RL_POLICY network of an RLlib checkpoint in the format of experiments/pretrain_policy.py
(`policy_weights.pt` + `config.yaml` in one run dir), so an RL agent can be evaluated by
experiments/imitation_rollout.py exactly like a pretrained ("junior") policy.

The weights are read from the policy state pickle (`policies/<RL_POLICY>/policy_state.pkl`, key
`weights`: the model's state_dict as numpy arrays), so neither Ray nor the policy class is needed.
The config is the Hydra config train.py saved for the run (`<storage_path>/<name>/.hydra/config.yaml`),
saved resolved like pretrain_policy.py's: `${hydra:...}` interpolations are filled from the run's own
`.hydra/hydra.yaml`, since they cannot be resolved outside a Hydra app.

With `--select best_val` the checkpoint with the highest val survival is exported instead of the
newest (`imitation.senior.select_best_val_checkpoint`); `checkpoint_selection.json` records the choice.

Example:
    PYTHONPATH=$(pwd)/src python experiments/export_policy_weights.py \\
        --experiment-dir results/senior/2026_10_14_IEEE36_imitation/senior_s0 \\
        --out results/senior/2026_10_14_IEEE36_imitation/eval/senior_s0
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.constants import RL_POLICY
from imitation.senior import VAL_SURVIVAL_KEY, select_best_val_checkpoint


CHECKPOINT_RESULTS_FILE = "checkpoint_results.json"  # written by core.train for every kept checkpoint
SELECTION_FILE = "checkpoint_selection.json"


def trial_checkpoints(experiment_dir: Path) -> list[Path]:
    """The kept `checkpoint_*` directories of the experiment's only trial, oldest first."""
    # Directories only: the trial dir also holds checkpoint_results.json.
    checkpoints = sorted(c for c in experiment_dir.glob("*/checkpoint_*") if c.is_dir())
    trials = {c.parent for c in checkpoints}
    if len(trials) != 1:
        raise ValueError(f"Expected one trial with checkpoints in {experiment_dir}, found {len(trials)}")
    return checkpoints


def latest_checkpoint(experiment_dir: Path) -> Path:
    """The newest checkpoint of the experiment's only trial (Tune keeps the final one)."""
    return trial_checkpoints(experiment_dir)[-1]


def best_val_checkpoint(experiment_dir: Path, evaluation_interval: int) -> tuple[Path, dict]:
    """The checkpoint with the highest val survival and the record of the choice.

    Args:
        experiment_dir: train.py output dir with one trial.
        evaluation_interval: Iterations between two evaluations of the run.

    Returns:
        `(checkpoint, selection)`; selection has the chosen checkpoint, its iteration and val survival,
        and the val survival of every candidate.
    """
    trial = trial_checkpoints(experiment_dir)[0].parent
    results = json.loads((trial / CHECKPOINT_RESULTS_FILE).read_text())
    kept = {c.name for c in trial_checkpoints(experiment_dir)}
    results = {name: metrics for name, metrics in results.items() if name in kept}
    name = select_best_val_checkpoint(results, evaluation_interval)
    selection = {
        "checkpoint": name,
        "training_iteration": results[name]["training_iteration"],
        "timesteps_total": results[name].get("timesteps_total"),
        "val_survival": results[name][VAL_SURVIVAL_KEY],
        "evaluation_interval": evaluation_interval,
        "candidates": {n: {"training_iteration": m.get("training_iteration"), "val_survival": m.get(VAL_SURVIVAL_KEY)}
                       for n, m in sorted(results.items())},
    }
    return trial / name, selection


def load_policy_weights(checkpoint: Path) -> dict[str, torch.Tensor]:
    """The RL_POLICY model's state_dict stored in an RLlib algorithm checkpoint."""
    policy_dir = checkpoint / "policies" / RL_POLICY
    info = json.loads((policy_dir / "rllib_checkpoint.json").read_text())
    with open(policy_dir / info["state_file"], "rb") as f:
        state = pickle.load(f)
    return {name: torch.as_tensor(value) for name, value in state["weights"].items()}


def resolved_run_config(hydra_dir: Path):
    """The run's Hydra config with all interpolations resolved, `${hydra:...}` from `hydra.yaml`."""
    hydra_cfg = OmegaConf.load(hydra_dir / "hydra.yaml").hydra
    OmegaConf.register_new_resolver("hydra", lambda key: OmegaConf.select(hydra_cfg, key), replace=True)
    cfg = OmegaConf.load(hydra_dir / "config.yaml")
    OmegaConf.resolve(cfg)
    return cfg


def main(args: argparse.Namespace) -> None:
    experiment_dir = Path(args.experiment_dir)
    cfg = resolved_run_config(experiment_dir / ".hydra")
    selection = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
    elif args.select == "best_val":
        checkpoint, selection = best_val_checkpoint(experiment_dir, int(cfg.evaluation.evaluation_interval))
    else:
        checkpoint = latest_checkpoint(experiment_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(load_policy_weights(checkpoint), out / "policy_weights.pt")
    # The training chronics_dir is often a node-local copy ($TMPDIR) that is gone after the job: rollouts read
    # the default Grid2Op data dir unless imitation_rollout.py gets --chronics-dir.
    cfg.env.chronics_dir = None
    OmegaConf.save(cfg, out / "config.yaml")
    (out / "source_checkpoint.txt").write_text(f"{checkpoint.resolve()}\n")
    if selection is not None:
        (out / SELECTION_FILE).write_text(json.dumps(selection, indent=1))
        print(f"Best val checkpoint: iteration {selection['training_iteration']}, val survival {selection['val_survival']}")
    print(f"Exported {checkpoint} to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", required=True,
                        help="train.py output dir: <experiment.storage_path>/<experiment.name>")
    parser.add_argument("--checkpoint", default=None, help="Export this checkpoint (overrides --select)")
    parser.add_argument("--select", choices=["latest", "best_val"], default="latest",
                        help="latest: the trial's newest checkpoint; best_val: the highest val survival among the "
                             "kept checkpoints of evaluation iterations")
    parser.add_argument("--out", required=True, help="Run dir for experiments/imitation_rollout.py")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
