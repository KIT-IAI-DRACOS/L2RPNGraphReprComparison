"""
Unit tests for imitation/tutor.py.

Tests are organised by class:
  - TestSelectAction
  - TestTutorEvaluate
  - TestChronicSeed
  - TestLoadPolicyActions
"""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import grid2op
import numpy as np
from lightsim2grid import LightSimBackend

from grid2op_env.utils import chronic_seed
from imitation.tutor import DO_NOTHING_INDEX, Tutor, load_policy_actions, select_action

ROOT = Path(__file__).resolve().parents[3]
ACTION_SPACE = ROOT / "data/action_spaces/l2rpn_wcci_2020/teacher_n1_k208.json"


class TestSelectAction(unittest.TestCase):
    """Tests for the Tutor's choice among simulated candidates."""

    def test_picks_best_topology_action_beyond_margin(self):
        max_rho = np.array([1.00, 0.90, 0.80, 0.95], dtype=np.float32)
        self.assertEqual(select_action(max_rho, np.ones(4, dtype=bool), margin=0.005), 2)

    def test_prefers_do_nothing_on_near_tie(self):
        max_rho = np.array([0.900, 0.898, 0.950], dtype=np.float32)
        self.assertEqual(select_action(max_rho, np.ones(3, dtype=bool), margin=0.005), DO_NOTHING_INDEX)

    def test_ignores_invalid_candidates(self):
        max_rho = np.array([1.0, np.nan, 0.9], dtype=np.float32)
        valid = np.array([True, False, True])
        self.assertEqual(select_action(max_rho, valid, margin=0.0), 2)

    def test_invalid_do_nothing_takes_any_valid_action(self):
        max_rho = np.array([np.nan, 1.5, 1.2], dtype=np.float32)
        valid = np.array([False, True, True])
        self.assertEqual(select_action(max_rho, valid, margin=0.005), 2)

    def test_nothing_valid_falls_back_to_do_nothing(self):
        max_rho = np.full(3, np.nan, dtype=np.float32)
        self.assertEqual(select_action(max_rho, np.zeros(3, dtype=bool), margin=0.005), DO_NOTHING_INDEX)


class _SimulatedObs:
    """Stands in for an observation whose actions are their indices: `simulate(i)` yields max rho[i]."""

    def __init__(self, max_rho: list[float]):
        self.max_rho = max_rho
        self.simulated: list[int] = []

    def simulate(self, action):
        self.simulated.append(action)
        return SimpleNamespace(rho=np.array([self.max_rho[action]])), 0.0, False, {}


class TestTutorEvaluate(unittest.TestCase):
    """Simulation of all candidates or of a subset (plus do-nothing)."""

    def setUp(self):
        self.tutor = Tutor(env=None, actions=[0, 1, 2, 3], line_reco=False, margin=0.005)

    def test_all_candidates_by_default(self):
        obs = _SimulatedObs([1.0, 0.9, 0.8, 0.95])
        max_rho, valid, chosen, action = self.tutor.evaluate(obs)
        self.assertEqual(sorted(obs.simulated), [0, 1, 2, 3])
        self.assertTrue(valid.all())
        self.assertEqual((chosen, action), (2, 2))

    def test_subset_always_includes_do_nothing(self):
        obs = _SimulatedObs([1.0, 0.9, 0.8, 0.95])
        max_rho, valid, chosen, _ = self.tutor.evaluate(obs, [3, 1])
        self.assertEqual(sorted(obs.simulated), [0, 1, 3])
        self.assertEqual(valid.tolist(), [True, True, False, True])
        self.assertTrue(np.isnan(max_rho[2]))
        self.assertEqual(chosen, 1)


class TestChronicSeed(unittest.TestCase):
    """Each chronic needs its own seed, or all chronics share one maintenance schedule."""

    def test_differs_between_chronics(self):
        self.assertNotEqual(chronic_seed(42, "Scenario_june_051"), chronic_seed(42, "Scenario_september_061"))

    def test_path_and_name_give_the_same_seed(self):
        self.assertEqual(chronic_seed(42, "/data/chronics/Scenario_june_051"), chronic_seed(42, "Scenario_june_051"))

    def test_depends_on_base_seed(self):
        self.assertNotEqual(chronic_seed(0, "Scenario_june_051"), chronic_seed(1, "Scenario_june_051"))

    def test_in_valid_range(self):
        self.assertTrue(0 <= chronic_seed(2**31 - 1, "Scenario_june_051") < 2**31)


class TestLoadPolicyActions(unittest.TestCase):
    """The Tutor's candidates must match the RL env's discrete action space."""

    @classmethod
    def setUpClass(cls):
        cls.env = grid2op.make("l2rpn_wcci_2020_train", backend=LightSimBackend())

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_do_nothing_first_then_json_order(self):
        actions = load_policy_actions(self.env, ACTION_SPACE)
        self.assertEqual(len(actions), 209)
        self.assertEqual(actions[DO_NOTHING_INDEX], self.env.action_space({}))
        self.assertNotEqual(actions[1], self.env.action_space({}))
