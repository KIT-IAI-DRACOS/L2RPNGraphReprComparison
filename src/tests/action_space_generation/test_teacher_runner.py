"""
Unit tests for teacher_runner.py.

Requires the `curriculumagent` package (see environment_curriculum.yaml) — skipped entirely
when run in the main `L2RPN` environment, where it isn't installed.

Tests are organised by class:
  - TestBuildTeacher
  - TestRunTeacherChronics
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("curriculumagent")

from action_space_generation.line_selection import DEFAULT_LINES_TO_ATTACK
from action_space_generation.teacher_runner import (
    TeacherConfig,
    build_teacher,
    run_teacher_chronics,
)


class TestBuildTeacher(unittest.TestCase):
    """Tests for constructing a configured NMinusOneTeacher from a TeacherConfig."""

    def test_uses_config_lines_and_seed(self):
        """lines_to_attack and seed are forwarded unchanged, not the package's track1 default."""
        config = TeacherConfig(env_name="dummy_env", lines_to_attack=[1, 2, 3], save_path=Path("/tmp/x.csv"), seed=7)
        teacher = build_teacher(config)
        self.assertEqual(teacher.lines_to_attack, [1, 2, 3])
        self.assertEqual(teacher.seed, 7)

    def test_defaults_lines_to_attack_per_grid_when_omitted(self):
        """Without explicit lines_to_attack, the grid's precomputed defaults are used."""
        config = TeacherConfig(env_name="l2rpn_wcci_2022_train", save_path=Path("/tmp/x.csv"))
        self.assertEqual(config.lines_to_attack, DEFAULT_LINES_TO_ATTACK["l2rpn_wcci_2022"])

    def test_uses_this_repos_rho_thresholds(self):
        """rho_n0 must match this repo's HIGH_LEVEL_AGENT activation threshold (0.95)."""
        config = TeacherConfig(env_name="dummy_env", lines_to_attack=[1], save_path=Path("/tmp/x.csv"))
        teacher = build_teacher(config)
        self.assertEqual(teacher.rho_n0, 0.95)
        self.assertEqual(teacher.rho_threshold, 1.0)

    def test_forwards_overridden_rho_thresholds(self):
        """Per-config thresholds reach the teacher (they are dataclass fields, not class constants)."""
        config = TeacherConfig(
            env_name="dummy_env", lines_to_attack=[1], save_path=Path("/tmp/x.csv"),
            rho_n0_threshold=0.9, rho_max_threshold=1.1,
        )
        teacher = build_teacher(config)
        self.assertEqual(teacher.rho_n0, 0.9)
        self.assertEqual(teacher.rho_threshold, 1.1)


class TestRunTeacherChronics(unittest.TestCase):
    """Tests for the batched SLURM-array-task entry point (regression: array-width limits).

    A one-array-task-per-chronic design hits a cluster's SLURM MaxArraySize on datasets with
    thousands of chronics (confirmed on BWUniCluster for l2rpn_wcci_2020's 2592 chronics —
    sbatch rejected the array outright). Batching chronics into one task per shard avoids this.
    """

    def test_calls_n_minus_one_agent_once_per_chronic_with_same_save_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = TeacherConfig(
                env_name="l2rpn_wcci_2020_train",
                lines_to_attack=[3, 12, 27],
                save_path=Path(tmp_dir) / "shard_0.csv",
                seed=42,
            )
            with patch("action_space_generation.teacher_runner.NMinusOneTeacher.n_minus_one_agent") as mock_agent:
                run_teacher_chronics(config, chronics_ids=["0000", "0001", "0002"])

            self.assertEqual(mock_agent.call_count, 3)
            for call, chronics_id in zip(mock_agent.call_args_list, ["0000", "0001", "0002"]):
                self.assertEqual(
                    call.kwargs,
                    dict(
                        env_path=config.env_name,
                        chronics_path=None,
                        chronics_id=chronics_id,
                        save_path=config.save_path,
                        save_greedy=True,
                        active_search=True,
                        disable_opponent=True,
                    ),
                )

    def test_empty_batch_does_not_call_n_minus_one_agent(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = TeacherConfig(
                env_name="l2rpn_wcci_2020_train",
                lines_to_attack=[3],
                save_path=Path(tmp_dir) / "shard_0.csv",
            )
            with patch("action_space_generation.teacher_runner.NMinusOneTeacher.n_minus_one_agent") as mock_agent:
                run_teacher_chronics(config, chronics_ids=[])
            mock_agent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
