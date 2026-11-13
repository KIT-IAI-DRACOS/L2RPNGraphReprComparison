"""
Unit tests for aggregation.py.

Requires the `curriculumagent` package (see environment_curriculum.yaml) — skipped entirely
when run in the main `L2RPN` environment, where it isn't installed.

Tests are organised by class:
  - TestActionFrequency
  - TestSubstationDistribution
"""
from __future__ import annotations

import unittest

import pytest

pytest.importorskip("curriculumagent")

import grid2op
import pandas as pd
from curriculumagent.teacher.submodule.common import affected_substations

from action_space_generation.aggregation import action_frequency, substation_distribution

ENV_NAME = "l2rpn_case14_sandbox"


class TestActionFrequency(unittest.TestCase):
    """Tests for counting how often each encoded action was selected."""

    def test_counts_and_sorts_descending(self):
        """Ties aside, more frequent actions come first — this is what picking top-k relies on."""
        data = pd.DataFrame({"best_action": ["a", "b", "a", "a", "c", "b"]})
        freq = action_frequency(data)
        self.assertEqual(list(freq.index), ["a", "b", "c"])
        self.assertEqual(list(freq.values), [3, 2, 1])


class TestSubstationDistribution(unittest.TestCase):
    """Tests for counting how many selected actions touch each substation."""

    def test_total_count_matches_sum_of_per_action_substations(self):
        """Aggregating over actions preserves the "one count per affected substation" contract."""
        env = grid2op.make(ENV_NAME)
        try:
            actions = env.action_space.get_all_unitary_topologies_set(env.action_space)[:5]
            dist = substation_distribution(actions)
            expected_total = sum(len(affected_substations(action)) for action in actions)
            self.assertEqual(dist.sum(), expected_total)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
