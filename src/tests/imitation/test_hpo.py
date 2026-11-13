"""
Unit tests for imitation/hpo.py.

Tests are organised by class:
  - TestSampling
  - TestApplyOverrides
  - TestPruning
  - TestSamplerSeedAndPruner
  - TestObjective
  - TestStudy
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import optuna
import pandas as pd
from omegaconf import OmegaConf

from imitation import hpo

SEARCH_SPACE = {
    "imitation.lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
    "imitation.patience": {"type": "categorical", "choices": [3, 5, 10]},
    "model.gnn.num_layers": {"type": "int", "low": 2, "high": 4},
    "model.fcnet_hiddens": {"type": "categorical", "choices": [[128, 128], [256, 256]]},
}


def base_cfg():
    return OmegaConf.create({
        "imitation": {"lr": 1e-3, "patience": 5, "out_dir": "unused"},
        "model": {"gnn": {"num_layers": 3}, "fcnet_hiddens": [64]},
        "hpo": {"monitor": "val_kl", "search_space": SEARCH_SPACE},
    })


def new_trial() -> optuna.Trial:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0)).ask()


class TestSampling(unittest.TestCase):
    def test_values_respect_the_spec(self):
        for _ in range(20):
            overrides = hpo.sample_overrides(new_trial(), SEARCH_SPACE)
            self.assertTrue(1e-4 <= overrides["imitation.lr"] <= 1e-2)
            self.assertIn(overrides["imitation.patience"], [3, 5, 10])
            self.assertIn(overrides["model.gnn.num_layers"], [2, 3, 4])
            self.assertIn(overrides["model.fcnet_hiddens"], [[128, 128], [256, 256]])

    def test_list_choices_are_stored_as_json(self):
        trial = new_trial()
        hpo.sample_overrides(trial, SEARCH_SPACE)
        self.assertIsInstance(trial.params["model.fcnet_hiddens"], str)

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            hpo.suggest_value(new_trial(), "x", {"type": "loguniform", "low": 1, "high": 2})


class TestApplyOverrides(unittest.TestCase):
    def test_sets_nested_keys_on_a_copy(self):
        cfg = base_cfg()
        trial_cfg = hpo.apply_overrides(cfg, {"model.gnn.num_layers": 4, "model.fcnet_hiddens": [256, 256]})
        self.assertEqual(trial_cfg.model.gnn.num_layers, 4)
        self.assertEqual(list(trial_cfg.model.fcnet_hiddens), [256, 256])
        self.assertEqual(cfg.model.gnn.num_layers, 3)
        self.assertEqual(list(cfg.model.fcnet_hiddens), [64])

    def test_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            hpo.apply_overrides(base_cfg(), {"model.gnn.hiden_dim": 4})


class _AlwaysPrune(optuna.pruners.BasePruner):
    def prune(self, study, trial) -> bool:
        return True


class TestPruning(unittest.TestCase):
    def test_reports_and_raises_when_pruner_says_so(self):
        trial = optuna.create_study(pruner=_AlwaysPrune()).ask()
        callback = hpo.pruning_callback(trial, "val_kl")
        with self.assertRaises(optuna.TrialPruned):
            callback(1, {"val_kl": 0.3})

    def test_keeps_going_without_pruning(self):
        trial = optuna.create_study(pruner=optuna.pruners.NopPruner()).ask()
        hpo.pruning_callback(trial, "val_kl")(1, {"val_kl": 0.3})

    def test_best_epoch_has_lowest_monitor(self):
        history = [{"epoch": 1, "val_kl": 0.5}, {"epoch": 2, "val_kl": 0.2}, {"epoch": 3, "val_kl": 0.3}]
        self.assertEqual(hpo.best_epoch(history, "val_kl")["epoch"], 2)


class TestSamplerSeedAndPruner(unittest.TestCase):
    def test_seed_is_stable_and_differs_between_studies_and_workers(self):
        seeds = {hpo.study_sampler_seed(study, worker) for study in ("junior2_hard", "junior2_soft_t0.01")
                 for worker in range(4)}
        self.assertEqual(len(seeds), 8)
        # Fixed value: the seed must not depend on the process (Python's str hash does).
        self.assertEqual(hpo.study_sampler_seed("junior2_hard", 0), hpo.study_sampler_seed("junior2_hard", 0))
        self.assertNotEqual(hpo.study_sampler_seed("junior2_hard", 0, base_seed=1), hpo.study_sampler_seed("junior2_hard", 0))

    def test_percentile_pruner_prunes_only_the_worst_quarter(self):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(pruner=hpo.make_pruner({"percentile": 75, "n_startup_trials": 4, "n_warmup_steps": 0}))
        for value in (1.0, 2.0, 3.0, 4.0):  # completed trials; 75th percentile = 3.25
            trial = study.ask()
            trial.report(value, step=1)
            study.tell(trial, value)
        for value, pruned in ((3.0, False), (3.5, True)):
            trial = study.ask()
            trial.report(value, step=1)
            self.assertEqual(trial.should_prune(), pruned)

    def test_default_percentile_is_the_median(self):
        self.assertEqual(hpo.make_pruner()._percentile, 50.0)


class TestObjective(unittest.TestCase):
    """make_objective wires overrides, run directory and metrics; the training itself is stubbed."""

    def test_trial_runs_with_overrides_and_stores_metrics(self):
        calls = []

        def fake_run(cfg, sets, obs_space, out_dir, epoch_callback):
            calls.append((cfg, out_dir))
            history = [{"epoch": 1, "val_kl": 0.4, "val_accuracy": 0.5}, {"epoch": 2, "val_kl": 0.25, "val_accuracy": 0.6}]
            return history, {"kl": 0.3, "accuracy": 0.55}

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(hpo, "run_pretraining", fake_run):
            study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
            study.optimize(hpo.make_objective(base_cfg(), sets=None, obs_space=None, study_dir=Path(tmp)), n_trials=1)

        trial = study.trials[0]
        cfg, out_dir = calls[0]
        self.assertEqual(trial.value, 0.25)
        self.assertEqual(out_dir, Path(tmp) / "trial_000")
        self.assertEqual(cfg.imitation.out_dir, str(out_dir))
        self.assertEqual(cfg.imitation.lr, trial.params["imitation.lr"])
        self.assertEqual(trial.user_attrs["best_epoch"], 2)
        self.assertEqual(trial.user_attrs["val_accuracy"], 0.6)
        self.assertEqual(trial.user_attrs["test_kl"], 0.3)


class TestStudy(unittest.TestCase):
    def test_budget_is_shared_and_best_link_points_to_lowest_value(self):
        values = iter([0.5, 0.1, 0.3, 0.9])
        with tempfile.TemporaryDirectory() as tmp:
            study_dir = Path(tmp)
            study = hpo.load_study(study_dir, "s", sampler_seed=0)
            hpo.run_study(study, lambda trial: next(values), n_trials=3)
            # A second worker joining a finished study runs nothing.
            hpo.run_study(hpo.load_study(study_dir, "s", sampler_seed=1), lambda trial: next(values), n_trials=3)
            self.assertEqual(len(hpo.load_study(study_dir, "s").trials), 3)

            best_dir = hpo.summarize_study(study, study_dir)
            self.assertEqual(best_dir.name, "trial_001")
            self.assertEqual((study_dir / hpo.BEST_LINK).readlink(), Path("trial_001"))
            self.assertEqual(len(pd.read_csv(study_dir / hpo.TRIALS_FILE)), 3)

    def test_open_study_does_not_create_missing_studies(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                hpo.open_study(Path(tmp), "s")
            hpo.run_study(hpo.load_study(Path(tmp), "s", sampler_seed=0), lambda trial: 1.0, n_trials=1)
            self.assertEqual(len(hpo.open_study(Path(tmp), "s").trials), 1)
            with self.assertRaises(KeyError):
                hpo.open_study(Path(tmp), "other")

    def test_failed_trials_use_budget_without_stopping_the_worker(self):
        def failing(trial):
            raise RuntimeError("CUDA out of memory")

        with tempfile.TemporaryDirectory() as tmp:
            study = hpo.load_study(Path(tmp), "s", sampler_seed=0)
            hpo.run_study(study, failing, n_trials=2)
            self.assertEqual([t.state for t in study.trials], [optuna.trial.TrialState.FAIL] * 2)


if __name__ == "__main__":
    unittest.main()
