"""
Unit tests for analysis/training_curves.py.

Tests are organised by class:
  - TestLoadScalars
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from analysis.training_curves import load_scalars


class TestLoadScalars(unittest.TestCase):
    def test_one_row_per_step_with_gaps_for_sparse_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = SummaryWriter(tmp)
            for i, step in enumerate((100, 250, 400)):
                writer.add_scalar("learner/vf_loss", 10.0 - i, step)
                if i == 1:  # evaluated in the second iteration only
                    writer.add_scalar("evaluation/survival", 3000.0, step)
            writer.close()
            table = load_scalars(Path(tmp), {"vf_loss": "learner/vf_loss", "eval": "evaluation/survival",
                                             "missing": "not/logged"})
        self.assertEqual(list(table.columns), ["iteration", "timesteps", "vf_loss", "eval"])
        self.assertEqual(list(table["iteration"]), [1, 2, 3])
        self.assertEqual(list(table["timesteps"]), [100, 250, 400])
        self.assertEqual(list(table["vf_loss"]), [10.0, 9.0, 8.0])
        self.assertTrue(np.isnan(table["eval"][0]) and table["eval"][1] == 3000.0)


if __name__ == "__main__":
    unittest.main()
