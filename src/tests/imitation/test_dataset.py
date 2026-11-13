"""
Unit tests for imitation/dataset.py.

Tests are organised by class:
  - TestSoftTargets
  - TestFromGreedyTutor
  - TestMakeSplit
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from imitation.dataset import (
    GREEDY_TUTOR,
    HARD,
    SOFT,
    ImitationData,
    from_greedy_tutor,
    load_split,
    make_split,
    save_split,
    soft_targets,
    split_indices,
)
from imitation.tutor import select_action


class TestSoftTargets(unittest.TestCase):
    """Tests for Boltzmann targets over do-nothing and the actions that beat it by the margin."""

    def test_rows_sum_to_one_and_argmax_is_best_action(self):
        max_rho = np.array([[1.0, 0.8, 0.9], [0.7, 0.95, 0.6]], dtype=np.float32)
        target = soft_targets(max_rho, np.ones_like(max_rho, dtype=bool), temperature=0.05)
        np.testing.assert_allclose(target.sum(axis=1), 1.0, rtol=1e-6)
        np.testing.assert_array_equal(target.argmax(axis=1), [1, 2])

    def test_invalid_actions_get_zero_probability(self):
        max_rho = np.array([[1.0, np.nan, 0.9]], dtype=np.float32)
        valid = np.array([[True, False, True]])
        self.assertEqual(soft_targets(max_rho, valid, temperature=0.1)[0, 1], 0.0)

    def test_actions_not_beating_do_nothing_by_margin_get_zero(self):
        """Worse actions, exact copies of do-nothing and gains within the margin are excluded."""
        max_rho = np.array([[1.0, 1.1, 1.0, 0.997, 0.9]], dtype=np.float32)
        target = soft_targets(max_rho, np.ones_like(max_rho, dtype=bool), temperature=0.1, margin=0.005)
        np.testing.assert_array_equal(target[0, 1:4], 0.0)
        self.assertGreater(target[0, 0], 0.0)
        self.assertGreater(target[0, 4], 0.0)

    def test_do_nothing_is_scored_with_the_margin(self):
        max_rho = np.array([[1.0, 0.99]], dtype=np.float32)
        target = soft_targets(max_rho, np.ones((1, 2), bool), temperature=0.01, margin=0.005)
        # Scores 0.995 vs 0.99: p(do-nothing) / p(action) = exp(-0.005 / 0.01).
        np.testing.assert_allclose(target[0, 0] / target[0, 1], np.exp(-0.5), rtol=1e-4)

    def test_do_nothing_mass_shrinks_with_the_best_gain(self):
        gains = np.array([0.004, 0.006, 0.01, 0.02, 0.1])
        max_rho = np.stack([np.ones_like(gains), 1.0 - gains], axis=1).astype(np.float32)
        p_dn = soft_targets(max_rho, np.ones_like(max_rho, dtype=bool), temperature=0.01, margin=0.005)[:, 0]
        self.assertEqual(p_dn[0], 1.0)
        self.assertTrue(np.all(np.diff(p_dn) < 0))

    def test_argmax_is_the_tutors_choice(self):
        rng = np.random.default_rng(0)
        max_rho = rng.uniform(0.8, 1.1, size=(500, 8)).astype(np.float32)
        max_rho[:250, 0] = max_rho[:250, 1:].min(axis=1) + rng.uniform(-0.01, 0.01, size=250)  # near the margin
        valid = rng.random((500, 8)) > 0.2
        max_rho[~valid] = np.nan
        target = soft_targets(max_rho, valid, temperature=0.01, margin=0.005)
        expected = [select_action(r, v, 0.005) for r, v in zip(max_rho, valid)]
        np.testing.assert_array_equal(target.argmax(axis=1), expected)

    def test_invalid_do_nothing_leaves_all_valid_actions_in_support(self):
        max_rho = np.array([[np.nan, 1.2, 1.3]], dtype=np.float32)
        target = soft_targets(max_rho, np.array([[False, True, True]]), temperature=0.1)
        self.assertEqual(target[0, 0], 0.0)
        self.assertTrue(np.all(target[0, 1:] > 0.0))

    def test_temperature_is_in_absolute_rho_units(self):
        """The same rho gap gives the same probabilities regardless of the rho level."""
        calm = soft_targets(np.array([[0.52, 0.50, 0.51]], dtype=np.float32), np.ones((1, 3), bool), 0.01)
        stressed = soft_targets(np.array([[1.22, 1.20, 1.21]], dtype=np.float32), np.ones((1, 3), bool), 0.01)
        np.testing.assert_allclose(calm, stressed, rtol=1e-4)

    def test_no_valid_action_falls_back_to_do_nothing(self):
        target = soft_targets(np.full((1, 3), np.nan, np.float32), np.zeros((1, 3), bool), 0.01)
        np.testing.assert_array_equal(target, [[1.0, 0.0, 0.0]])

    def test_rejects_non_positive_temperature(self):
        with self.assertRaises(ValueError):
            soft_targets(np.zeros((1, 2), np.float32), np.ones((1, 2), bool), 0.0)


class TestFromGreedyTutor(unittest.TestCase):
    """Tests for converting imitation.tutor files."""

    def _write_chronic(self, path: Path) -> None:
        np.savez(
            path,
            obs=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            step=np.array([10, 11], dtype=np.int32),
            max_rho=np.array([[1.0, 0.8, np.nan], [np.nan, np.nan, np.nan]], dtype=np.float32),
            valid=np.array([[True, True, False], [False, False, False]]),
            action=np.array([1, 0], dtype=np.int16),
            chronic=np.str_("Scenario_a"),
            action_space=np.str_("k2.json"),
            do_nothing_margin=np.float32(0.005),
        )

    def test_hard_and_soft_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._write_chronic(tmp / "Scenario_a.npz")

            hard = from_greedy_tutor(tmp, target=HARD)
            np.testing.assert_array_equal(hard.policy_target, [[0, 1, 0], [1, 0, 0]])
            self.assertEqual(hard.metadata["source"], GREEDY_TUTOR)

            soft = from_greedy_tutor(tmp, target=SOFT, temperature=0.1)
            self.assertGreater(soft.policy_target[0, 0], 0.0)
            self.assertEqual(soft.policy_target[0].argmax(), 1)
            self.assertAlmostEqual(soft.metadata["do_nothing_margin"], 0.005)

    def test_fallback_target_is_never_masked(self):
        """A state with no valid action targets do-nothing, which must stay selectable."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write_chronic(Path(tmp) / "Scenario_a.npz")
            data = from_greedy_tutor(Path(tmp), target=HARD)
            self.assertTrue(data.action_mask[1, 0])
            self.assertTrue(np.all(data.action_mask[data.policy_target > 0]))


class TestMakeSplit(unittest.TestCase):
    """Chronic-level split, stratified by month."""

    CHRONICS = [f"/data/chronics/Scenario_{m}_{i:03d}" for m in ("april", "may") for i in range(20)]

    def setUp(self):
        self.split = make_split(self.CHRONICS, (0.8, 0.1, 0.1), seed=0)

    def test_disjoint_and_complete(self):
        names = [c for members in self.split.values() for c in members]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), {Path(c).name for c in self.CHRONICS})

    def test_every_month_in_every_split(self):
        for members in self.split.values():
            self.assertEqual({c.split("_")[1] for c in members}, {"april", "may"})
        self.assertEqual(len(self.split["val"]), 4)
        self.assertEqual(len(self.split["test"]), 4)

    def test_deterministic_and_round_trips(self):
        self.assertEqual(self.split, make_split(list(reversed(self.CHRONICS)), (0.8, 0.1, 0.1), seed=0))
        with tempfile.TemporaryDirectory() as tmp:
            save_split(self.split, Path(tmp) / "split.json")
            self.assertEqual(load_split(Path(tmp) / "split.json"), self.split)

    def test_split_indices(self):
        val = self.split["val"][0]
        chronic = np.array([val, "other", val])
        np.testing.assert_array_equal(split_indices(chronic, self.split, "val"), [0, 2])

