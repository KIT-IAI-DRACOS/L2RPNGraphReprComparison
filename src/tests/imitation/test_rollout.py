"""
Unit tests for the action choosers of imitation/rollout.py.

Tests are organised by class:
  - TestChoosers
  - TestTopKChooser
  - TestRunPolicyChronic
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from imitation.rollout import do_nothing_chooser, greedy_chooser, run_policy_chronic, top_k_chooser
from imitation.tutor import DO_NOTHING_INDEX, Tutor
from tests.imitation.test_tutor import _SimulatedObs


class _FixedLogits(torch.nn.Module):
    """Stands in for a policy model: logits are passed in as the observation."""

    def forward(self, input_dict, state, seq_lens):
        return input_dict["obs"]["logits"], state


class TestChoosers(unittest.TestCase):
    def test_greedy_chooser_takes_argmax_in_eval_mode(self):
        model = _FixedLogits()
        chooser = greedy_chooser(model, torch.device("cpu"))
        self.assertFalse(model.training)
        self.assertEqual(chooser(None, {"logits": [0.1, 2.0, -1.0, 0.5]}), 1)

    def test_do_nothing_chooser_ignores_observation(self):
        self.assertEqual(do_nothing_chooser(None, None), DO_NOTHING_INDEX)
        self.assertEqual(do_nothing_chooser(None, {"logits": [0.0, 9.0]}), DO_NOTHING_INDEX)


class TestTopKChooser(unittest.TestCase):
    """The policy ranks, simulation of its k best actions and do-nothing decides."""

    # Simulated max rho per action; the policy ranks 1 > 2 > 3, but 3 is the best action.
    MAX_RHO = [1.00, 0.99, 0.97, 0.90]
    LOGITS = [0.0, 3.0, 2.0, 1.0]

    def choose(self, k: int, max_rho: list[float] | None = None) -> int:
        actions = list(range(len(self.LOGITS)))
        tutor = Tutor(env=None, actions=actions, line_reco=False, margin=0.005)
        chooser = top_k_chooser(_FixedLogits(), torch.device("cpu"), tutor, k)
        return chooser(_SimulatedObs(max_rho or self.MAX_RHO), {"logits": self.LOGITS})

    def test_takes_best_simulated_among_top_k(self):
        self.assertEqual(self.choose(k=2), 2)

    def test_k_of_all_actions_is_the_tutor(self):
        self.assertEqual(self.choose(k=4), 3)

    def test_vetoes_top_action_not_beating_do_nothing(self):
        self.assertEqual(self.choose(k=1, max_rho=[0.95, 0.948, 0.80, 0.70]), DO_NOTHING_INDEX)

    def test_k_beyond_action_count_is_clipped(self):
        self.assertEqual(self.choose(k=100), 3)


class _ScriptedEnv:
    """Env stub: max rho per step is scripted; the action named "illegal" is rejected by the env."""

    def __init__(self, rho_per_step: list[float]):
        self.rho_per_step = rho_per_step
        self.step_count = 0

    def seed(self, seed: int) -> None:
        pass

    def set_id(self, chronic_id: str) -> None:
        pass

    def obs(self) -> SimpleNamespace:
        return SimpleNamespace(rho=np.array([self.rho_per_step[self.step_count]]), current_step=self.step_count,
                               max_step=len(self.rho_per_step) - 1)

    def reset(self) -> SimpleNamespace:
        self.step_count = 0
        return self.obs()

    def step(self, action: str) -> tuple:
        self.step_count += 1
        done = self.step_count == len(self.rho_per_step) - 1
        return self.obs(), 0.0, done, {"is_illegal": action == "illegal"}


class TestRunPolicyChronic(unittest.TestCase):
    def test_counts_decisions_executed_and_illegal_actions(self):
        actions = ["do_nothing", "topology", "illegal"]
        # Above the threshold in steps 0-3, below in step 4; the policy picks topology, illegal, do-nothing, topology.
        env = _ScriptedEnv([0.99, 0.99, 0.99, 0.99, 0.5, 0.5])
        picks = iter([1, 2, DO_NOTHING_INDEX, 1])
        row = run_policy_chronic(env, None, lambda obs, gym_obs: next(picks), actions, "chronic_a",
                                 base_seed=0, rho_threshold=0.95, line_reco=False)
        self.assertEqual((row["n_decisions"], row["n_actions"], row["n_illegal"]), (4, 2, 1))
        self.assertEqual(row["steps_survived"], row["max_step"])


if __name__ == "__main__":
    unittest.main()
