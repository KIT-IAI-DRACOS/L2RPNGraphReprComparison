"""
Optuna search over imitation pretraining settings (configs/rllib/pretrain_hpo.yaml).

A trial samples overrides of the Hydra config from a declarative search space (dotted config paths
-> spec), trains one pretraining run (`imitation.pretrain_run.run_pretraining`) into
`<study_dir>/trial_<n>/` and returns the best validation value of `hpo.monitor` (e.g.
`val_regret_top20`, the regret of top-20 simulation). Epoch values are reported to Optuna so a
percentile pruner can stop weak trials. Every study and worker gets its own sampler seed
(`study_sampler_seed`), so no two of them start from the same configurations.

The study lives in a journal file (`<study_dir>/study.log`), which several workers on a shared
file system can use at once, unlike SQLite.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from imitation.pretrain import EpochCallback
from imitation.pretrain_run import ImitationSets, run_pretraining

logger = logging.getLogger(__name__)

STUDY_FILE = "study.log"
TRIALS_FILE = "trials.csv"
BEST_LINK = "best"

# Trial states that use up the study's budget. RUNNING is left out, so trials of a worker that hit
# the walltime are redone by the next submission (at the price of overshooting by < n_workers trials).
BUDGET_STATES = (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
# Trial errors that fail only the trial, not the worker (e.g. CUDA out of memory for large networks).
TRIAL_ERRORS = (RuntimeError, ValueError)


def trial_dir(study_dir: Path, number: int) -> Path:
    """Run directory of trial `number`."""
    return study_dir / f"trial_{number:03d}"


def suggest_value(trial: optuna.Trial, name: str, spec: dict) -> Any:
    """Sample one parameter.

    Args:
        trial: The Optuna trial.
        name: Parameter name (the dotted config path).
        spec: `{type: float, low, high, log?}`, `{type: int, low, high, log?}` or
            `{type: categorical, choices}`. Choices may be lists (e.g. layer sizes); Optuna stores
            them as JSON strings.

    Returns:
        The sampled value.
    """
    kind = spec["type"]
    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "categorical":
        choices = list(spec["choices"])
        if not any(isinstance(c, (list, tuple, dict)) for c in choices):
            return trial.suggest_categorical(name, choices)
        return json.loads(trial.suggest_categorical(name, [json.dumps(c) for c in choices]))
    raise ValueError(f"Unknown search space type '{kind}' for '{name}'. Use float, int or categorical.")


def sample_overrides(trial: optuna.Trial, search_space: dict[str, dict]) -> dict[str, Any]:
    """Sample every parameter of the search space.

    Returns:
        Dotted config path -> sampled value.
    """
    return {name: suggest_value(trial, name, dict(spec)) for name, spec in search_space.items()}


def apply_overrides(cfg: DictConfig, overrides: dict[str, Any]) -> DictConfig:
    """Copy of `cfg` with the overrides set (dotted paths must exist in `cfg`)."""
    trial_cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if OmegaConf.select(trial_cfg, key, default=KeyError) is KeyError:
            raise KeyError(f"Search space key '{key}' is not in the config")
        OmegaConf.update(trial_cfg, key, value, merge=False)
    return trial_cfg


def pruning_callback(trial: optuna.Trial, monitor: str) -> EpochCallback:
    """Epoch callback that reports `monitor` to Optuna and raises `optuna.TrialPruned` if told to prune."""

    def callback(epoch: int, row: dict[str, float]) -> None:
        trial.report(row[monitor], step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned after epoch {epoch} ({monitor}={row[monitor]:.4f})")

    return callback


def best_epoch(history: list[dict], monitor: str) -> dict[str, float]:
    """History row with the lowest `monitor` (the epoch whose weights the run keeps)."""
    return min(history, key=lambda row: row[monitor])


def make_objective(
    cfg: DictConfig, sets: ImitationSets, obs_space: gym.spaces.Space, study_dir: Path,
) -> Callable[[optuna.Trial], float]:
    """Objective of the study: one pretraining run per trial.

    Args:
        cfg: Full Hydra config with `hpo` section; trials override its search-space keys.
        sets: Train/val/test samples, shared by all trials.
        obs_space: Observation space of the policy network.
        study_dir: Trials are written to `trial_<n>/` below it.

    Returns:
        Function trial -> best validation `hpo.monitor`. Validation and test metrics of the best
        epoch are stored as trial user attributes (`val_*`, `test_*`, `best_epoch`, `n_epochs`).
    """
    monitor = cfg.hpo.monitor

    def objective(trial: optuna.Trial) -> float:
        overrides = sample_overrides(trial, OmegaConf.to_container(cfg.hpo.search_space, resolve=True))
        out_dir = trial_dir(study_dir, trial.number)
        trial_cfg = apply_overrides(cfg, {**overrides, "imitation.out_dir": str(out_dir)})
        logger.info(f"Trial {trial.number}: {overrides}")
        history, test_metrics = run_pretraining(trial_cfg, sets, obs_space, out_dir, pruning_callback(trial, monitor))
        best = best_epoch(history, monitor)
        trial.set_user_attr("best_epoch", int(best["epoch"]))
        trial.set_user_attr("n_epochs", len(history))
        for key, value in best.items():
            if key.startswith("val_"):
                trial.set_user_attr(key, float(value))
        for key, value in test_metrics.items():
            trial.set_user_attr(f"test_{key}", float(value))
        return float(best[monitor])

    return objective


def study_sampler_seed(study_name: str, worker_id: int, base_seed: int = 0) -> int:
    """Sampler seed of one worker of one study.

    A stable hash of all three inputs (Python's `hash` of a string changes between processes), so
    studies of different datasets and their workers do not start from the same configurations.

    Args:
        study_name: Name of the study (`experiment.name`).
        worker_id: Index of the parallel worker.
        base_seed: Extra seed, to repeat a study with different samples.

    Returns:
        Seed in [0, 2**32).
    """
    key = f"{base_seed}:{study_name}:{worker_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def make_pruner(pruner: dict | None = None) -> optuna.pruners.PercentilePruner:
    """Percentile pruner from config kwargs.

    For a minimised objective, a trial is pruned at an epoch if its best value so far is above the
    `percentile`-th percentile of the completed trials' values at that epoch. 75 prunes only the
    worst quarter; the default 50 equals Optuna's `MedianPruner`.

    Args:
        pruner: `PercentilePruner` kwargs (percentile, n_startup_trials, n_warmup_steps); None = defaults.
    """
    return optuna.pruners.PercentilePruner(**{"percentile": 50.0, **(pruner or {})})


def load_study(study_dir: Path, study_name: str, sampler_seed: int | None = None, pruner: dict | None = None) -> optuna.Study:
    """Create or join the study stored in `<study_dir>/study.log`.

    Args:
        study_dir: Directory of the journal file.
        study_name: Name of the study within the journal.
        sampler_seed: Seed of the TPE sampler; give every study and worker its own (`study_sampler_seed`).
        pruner: Kwargs of `make_pruner`.

    Returns:
        The study (direction: minimize).
    """
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(study_dir / STUDY_FILE)))
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        pruner=make_pruner(pruner),
        load_if_exists=True,
    )


def open_study(study_dir: Path, study_name: str) -> optuna.Study:
    """Open an existing study for analysis; raises if `<study_dir>/study.log` does not hold it."""
    if not (study_dir / STUDY_FILE).exists():
        raise FileNotFoundError(f"No study journal {study_dir / STUDY_FILE}")
    return optuna.load_study(study_name=study_name, storage=JournalStorage(JournalFileBackend(str(study_dir / STUDY_FILE))))


def run_study(study: optuna.Study, objective: Callable[[optuna.Trial], float], n_trials: int) -> None:
    """Run trials until the study holds `n_trials` finished trials (see `BUDGET_STATES`)."""
    if len(study.get_trials(deepcopy=False, states=BUDGET_STATES)) >= n_trials:
        logger.info(f"Study {study.study_name} already has {n_trials} finished trials")
        return
    study.optimize(objective, catch=TRIAL_ERRORS, callbacks=[MaxTrialsCallback(n_trials, states=BUDGET_STATES)])


def trials_table(study: optuna.Study) -> pd.DataFrame:
    """One row per trial: number, state, value, durations, `params_*` and `user_attrs_*` columns."""
    return study.trials_dataframe(attrs=("number", "state", "value", "duration", "params", "user_attrs"))


def best_trial_number(study: optuna.Study) -> int:
    """Number of the complete trial with the lowest objective value."""
    complete = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    if not complete:
        raise ValueError(f"Study {study.study_name} has no complete trial")
    return complete[int(np.argmin([t.value for t in complete]))].number


def summarize_study(study: optuna.Study, study_dir: Path) -> Path:
    """Write `trials.csv` and point the `best` link at the best trial's run directory.

    Returns:
        The best trial's run directory.
    """
    trials_table(study).to_csv(study_dir / TRIALS_FILE, index=False)
    best_dir = trial_dir(study_dir, best_trial_number(study))
    link = study_dir / BEST_LINK
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(best_dir.name, link)
    return best_dir
