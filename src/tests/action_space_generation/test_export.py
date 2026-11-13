"""
Unit tests for export.py.

Tests are organised by class:
  - TestActionsToJsonDicts
  - TestExportAndReload
  - TestVerifyRoundTrip
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import grid2op

from action_space_generation.export import actions_to_json_dicts, export_action_space, verify_round_trip
from grid2op_env.action_converters import load_actions

ENV_NAME = "l2rpn_case14_sandbox"


class TestActionsToJsonDicts(unittest.TestCase):
    """Tests for serializing actions to this repo's action-space dict format."""

    def setUp(self):
        self.env = grid2op.make(ENV_NAME)
        self.addCleanup(self.env.close)

    def test_do_nothing_action_serializes_to_empty_dict(self):
        """The do-nothing action has no set_bus entries once serialized."""
        actions = [self.env.action_space({})]
        dicts = actions_to_json_dicts(actions)
        self.assertEqual(len(dicts), 1)
        self.assertNotIn("set_bus", dicts[0])

    def test_topology_action_serializes_with_set_bus(self):
        """A substation reconfiguration action serializes with a "set_bus" key."""
        substation_actions = self.env.action_space.get_all_unitary_topologies_set(self.env.action_space)
        self.assertGreater(len(substation_actions), 0)
        dicts = actions_to_json_dicts(substation_actions[:1])
        self.assertIn("set_bus", dicts[0])

    def test_preserves_order_and_count(self):
        """One dict is produced per action, in the same order."""
        substation_actions = self.env.action_space.get_all_unitary_topologies_set(self.env.action_space)[:3]
        dicts = actions_to_json_dicts(substation_actions)
        self.assertEqual(len(dicts), 3)


class TestExportAndReload(unittest.TestCase):
    """Round trip through a real file, using this repo's own loader."""

    def test_export_then_load_actions_reconstructs_equal_actions(self):
        """Actions written by export_action_space load back as equal Grid2Op actions."""
        env = grid2op.make(ENV_NAME)
        try:
            actions = env.action_space.get_all_unitary_topologies_set(env.action_space)[:5]
            with tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "test_action_space.json"
                export_action_space(actions, path)

                self.assertTrue(path.exists())
                with open(path) as f:
                    raw = json.load(f)
                self.assertEqual(len(raw), 5)

                reloaded = load_actions(str(path), env)
                self.assertEqual(len(reloaded), len(actions))
                for original, reloaded_action in zip(actions, reloaded):
                    self.assertEqual(original, reloaded_action)
        finally:
            env.close()


class TestVerifyRoundTrip(unittest.TestCase):
    """Tests for the round-trip sanity check used before writing an action space to disk."""

    def setUp(self):
        self.env = grid2op.make(ENV_NAME)
        self.addCleanup(self.env.close)

    def test_returns_true_for_well_formed_actions(self):
        """Actions built from the environment's own action space round-trip cleanly."""
        actions = self.env.action_space.get_all_unitary_topologies_set(self.env.action_space)[:5]
        self.assertTrue(verify_round_trip(actions, self.env))

    def test_returns_false_when_reconstruction_diverges(self):
        """A reconstruction mismatch (e.g. a future grid2op regression) is caught, not silently passed.

        `verify_round_trip` always serializes and rebuilds the same `actions` argument, so a
        real mismatch can only come from a bug in (de)serialization itself. That's exercised
        here with a fake `env` whose action_space always rebuilds the same fixed action,
        regardless of the dict it's given.
        """

        class _AlwaysWrongActionSpace:
            def __init__(self, wrong_action):
                self._wrong_action = wrong_action

            def __call__(self, _action_dict):
                return self._wrong_action

        class _FakeEnv:
            def __init__(self, action_space):
                self.action_space = action_space

        actions = self.env.action_space.get_all_unitary_topologies_set(self.env.action_space)[:2]
        wrong_action = self.env.action_space({})  # do-nothing, different from any topology action
        fake_env = _FakeEnv(action_space=_AlwaysWrongActionSpace(wrong_action))

        self.assertFalse(verify_round_trip(actions, fake_env))


if __name__ == "__main__":
    unittest.main()
