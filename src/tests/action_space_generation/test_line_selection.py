"""
Unit tests for line_selection.py.

Tests are organised by class:
  - TestDefaultLinesToAttack
  - TestOutageImpactStats
  - TestFormatLineReport
  - TestComputePostOutageMaxRho
  - TestSelectLinesToAttack
"""
from __future__ import annotations

import unittest

import grid2op
import numpy as np
from grid2op.Observation import BaseObservation
from lightsim2grid import LightSimBackend

from action_space_generation.line_selection import (
    DEFAULT_LINES_TO_ATTACK,
    OutageImpactStats,
    compute_post_outage_max_rho,
    default_lines_to_attack,
    format_line_report,
    select_lines_to_attack,
)

ENV_NAME = "l2rpn_case14_sandbox"


class TestDefaultLinesToAttack(unittest.TestCase):
    """Tests for the precomputed per-grid lines_to_attack lookup."""

    def test_resolves_split_suffix_and_path(self):
        """Train/val/test splits and full paths all map to the grid's base entry."""
        expected = DEFAULT_LINES_TO_ATTACK["l2rpn_wcci_2020"]
        for env_name in ["l2rpn_wcci_2020", "l2rpn_wcci_2020_train", "l2rpn_wcci_2020_val", "~/data_grid2op/l2rpn_wcci_2020_test"]:
            self.assertEqual(default_lines_to_attack(env_name), expected)

    def test_returns_a_copy(self):
        """Mutating the returned list must not change the module-level defaults."""
        lines = default_lines_to_attack("l2rpn_case14_sandbox")
        lines.append(-1)
        self.assertNotIn(-1, DEFAULT_LINES_TO_ATTACK["l2rpn_case14_sandbox"])

    def test_unknown_grid_raises(self):
        """A grid without precomputed lines raises instead of silently using wrong ids."""
        with self.assertRaises(KeyError):
            default_lines_to_attack("l2rpn_neurips_2020_track1_small")

    def test_defaults_are_valid_solvable_lines_on_case14(self):
        """The case14 defaults are unique, in range, and never an islanding outage."""
        env = grid2op.make(ENV_NAME, backend=LightSimBackend())
        env.reset()
        post_outage = compute_post_outage_max_rho(env.backend, env.get_thermal_limit())
        n_line = env.n_line
        env.close()
        lines = default_lines_to_attack(ENV_NAME)
        self.assertEqual(len(set(lines)), len(lines))
        for line_id in lines:
            self.assertTrue(0 <= line_id < n_line)
            self.assertFalse(np.isnan(post_outage[line_id]))


class TestOutageImpactStats(unittest.TestCase):
    """Tests for the pure-numpy scoring logic (no Grid2Op env needed)."""

    def setUp(self):
        self.stats = OutageImpactStats(n_line=4, overload_threshold=1.0)

    def test_update_accumulates_only_overload_above_threshold(self):
        """overload_score sums the excess over the threshold, so 1.2 weighs far more than 1.01."""
        self.stats.update(np.array([1.2, 1.01, 0.9, 0.5]))
        self.stats.update(np.array([1.1, 1.01, 0.9, 0.5]))
        np.testing.assert_array_almost_equal(self.stats.overload_score, [0.3, 0.02, 0.0, 0.0])
        self.assertEqual(self.stats.n_samples, 2)

    def test_update_counts_nan_as_unsolvable_without_scoring_it(self):
        """A NaN (islanding/divergence) increments unsolvable_count and adds nothing to the scores."""
        self.stats.update(np.array([np.nan, 1.5, 0.9, 0.5]))
        np.testing.assert_array_equal(self.stats.unsolvable_count, [1, 0, 0, 0])
        self.assertEqual(self.stats.overload_score[0], 0.0)
        self.assertEqual(self.stats.post_outage_rho_sum[0], 0.0)
        self.assertEqual(self.stats.unsolvable_lines(), [0])

    def test_rank_lines_orders_by_overload_score(self):
        """A rare severe overload outranks a frequent marginal one when its summed excess is larger."""
        self.stats.update(np.array([1.01, 1.0, 0.5, 0.5]))
        self.stats.update(np.array([1.01, 1.3, 0.5, 0.5]))
        self.assertEqual(self.stats.rank_lines(n_lines=2), [1, 0])

    def test_rank_lines_breaks_ties_by_post_outage_rho_sum(self):
        """With no overloads at all (calm grid), rank by summed post-outage rho."""
        self.stats.update(np.array([0.3, 0.6, 0.1, 0.5]))
        self.assertEqual(self.stats.rank_lines(n_lines=4), [1, 3, 0, 2])

    def test_rank_lines_excludes_unsolvable_lines(self):
        """A line unsolvable in any sample is never returned, even with the highest score."""
        self.stats.update(np.array([2.0, 1.1, 0.5, 0.5]))
        self.stats.update(np.array([np.nan, 1.1, 0.5, 0.5]))
        ranked = self.stats.rank_lines(n_lines=4)
        self.assertNotIn(0, ranked)
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0], 1)

    def test_rank_lines_caps_at_n_line(self):
        """Requesting more lines than exist returns at most n_line ids."""
        self.stats.update(np.array([0.1, 0.1, 0.1, 0.1]))
        self.assertEqual(len(self.stats.rank_lines(n_lines=100)), 4)


class TestFormatLineReport(unittest.TestCase):
    """Tests for the human-readable per-line report."""

    def setUp(self):
        self.stats = OutageImpactStats(n_line=3)
        self.stats.update(np.array([1.25, np.nan, 0.5]))
        self.line_or_to_subid = np.array([0, 1, 4])
        self.line_ex_to_subid = np.array([2, 3, 5])

    def _report(self, line_ids: list[int]) -> list[str]:
        """Format the report for `line_ids` and split it into rows."""
        return format_line_report(self.stats, line_ids, self.line_or_to_subid, self.line_ex_to_subid).splitlines()

    def test_one_row_per_line_with_substations_and_score(self):
        """Each reported line shows its origin/extremity substations and its overload score."""
        rows = self._report([0, 2])
        self.assertEqual(rows[1].split(), ["0", "0", "->", "2", "0.25"])
        self.assertEqual(rows[2].split(), ["2", "4", "->", "5", "0.00"])

    def test_lists_excluded_unsolvable_lines_with_substations(self):
        """Unsolvable lines are listed with their substations below the table."""
        rows = self._report([0])
        self.assertIn("1 (1 -> 3)", rows[-2])
        self.assertEqual(rows[-1], "Samples: 1")


class TestComputePostOutageMaxRho(unittest.TestCase):
    """Checks the lightsim2grid contingency analysis against Grid2Op's own `obs.simulate`."""

    @classmethod
    def setUpClass(cls):
        cls.env = grid2op.make(ENV_NAME, backend=LightSimBackend())
        cls.obs = cls.env.reset()
        cls.result = compute_post_outage_max_rho(cls.env.backend, cls.env.get_thermal_limit())

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_returns_one_value_per_line(self):
        """One post-outage max rho per line."""
        self.assertEqual(self.result.shape, (self.env.n_line,))

    def _simulate_outage(self, line_id: int) -> tuple[BaseObservation, bool]:
        """Disconnect `line_id` in Grid2Op's `obs.simulate` on the current step."""
        action = self.env.action_space({"set_line_status": [(int(line_id), -1)]})
        sim_obs, _, done, _ = self.obs.simulate(action, time_step=0)
        return sim_obs, done

    def test_matches_simulated_line_disconnection(self):
        """Each solvable entry equals the max rho of `obs.simulate` with that line disconnected.

        Grid2Op instantly trips lines above HARD_OVERFLOW_THRESHOLD and re-runs the powerflow,
        so the comparison only holds when no such cascade happened; with a cascade, our
        (pre-cascade) value must be above the threshold that triggered it.
        """
        hard_threshold = self.env.parameters.HARD_OVERFLOW_THRESHOLD
        for line_id in np.flatnonzero(~np.isnan(self.result)):
            sim_obs, done = self._simulate_outage(line_id)
            self.assertFalse(done)
            if (~sim_obs.line_status).sum() == 1:
                self.assertAlmostEqual(self.result[line_id], sim_obs.rho.max(), places=4)
            else:
                self.assertGreater(self.result[line_id], hard_threshold)

    def test_nan_entries_end_the_episode_in_grid2op(self):
        """Every NaN entry (unsolvable outage) is a game over when simulated in Grid2Op."""
        nan_lines = np.flatnonzero(np.isnan(self.result))
        self.assertGreater(len(nan_lines), 0)  # case14 has a radial line
        for line_id in nan_lines:
            _, done = self._simulate_outage(line_id)
            self.assertTrue(done)

class TestSelectLinesToAttack(unittest.TestCase):
    """Smoke test for the end-to-end do-nothing rollout on a real (small) grid."""

    def test_returns_requested_number_of_valid_solvable_line_ids(self):
        """select_lines_to_attack returns n_lines valid, unique, solvable line ids."""
        env = grid2op.make(ENV_NAME, backend=LightSimBackend())
        env.reset()
        n_line = env.n_line
        unsolvable = np.flatnonzero(np.isnan(compute_post_outage_max_rho(env.backend, env.get_thermal_limit())))
        env.close()

        lines = select_lines_to_attack(env_name=ENV_NAME, n_lines=3, n_chronics=1)

        self.assertEqual(len(lines), 3)
        self.assertEqual(len(set(lines)), 3)
        for line_id in lines:
            self.assertTrue(0 <= line_id < n_line)
            self.assertNotIn(line_id, unsolvable)


if __name__ == "__main__":
    unittest.main()
