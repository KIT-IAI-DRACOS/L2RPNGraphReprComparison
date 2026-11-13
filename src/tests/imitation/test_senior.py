"""
Unit tests for imitation/senior.py (RL overrides from a Junior run's config).

Tests are organised by class:
  - TestFormatValue
  - TestArchitectureOverrides
  - TestSelectBestValCheckpoint
"""
from __future__ import annotations

import shlex
import unittest

from omegaconf import DictConfig, OmegaConf

from imitation.senior import (
    SKIPPED_KEYS,
    architecture_overrides,
    flatten,
    format_value,
    select_best_val_checkpoint,
    shell_overrides,
)
from tests.hydra_config import compose_config

# As in slurm/unicluster/graph_comparison_36bus_senior.sh, before the architecture overrides.
RL_OVERRIDES = ["training=ppo", "env=case36", "model=gcn_substation_readout", "obs_space=graph_comparison/substation",
                "imitation=default"]
# A Junior HPO trial that differs from the model defaults in every searched model setting.
JUNIOR_OVERRIDES = ["model.gnn.hidden_dim=512", "model.gnn.num_layers=4", "model.gnn.dropout_prob=0.27",
                    "model.fcnet_hiddens=[256,256]"]


def section_leaves(cfg: DictConfig, section: str) -> dict:
    """`{dotted.key: value}` of one config section, unresolved, without machine-specific keys."""
    flat = flatten(OmegaConf.to_container(cfg[section], resolve=False), section)
    return {k: v for k, v in flat.items() if k not in SKIPPED_KEYS}


class TestFormatValue(unittest.TestCase):
    def test_hydra_grammar(self):
        self.assertEqual(format_value([256, 256]), "[256,256]")
        self.assertEqual(format_value(["r", "t"]), "[r,t]")
        self.assertEqual(format_value(None), "null")
        self.assertEqual(format_value(False), "false")
        self.assertEqual(format_value(0.25), "0.25")


class TestArchitectureOverrides(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.junior = compose_config("pretrain_hpo_graph36", ["obs_space=graph_comparison/substation"] + JUNIOR_OVERRIDES)

    def test_rl_config_gets_junior_network_and_observations(self):
        rl = compose_config("config", RL_OVERRIDES + architecture_overrides(self.junior, dropout_prob=None))
        for section in ("model", "obs_space"):
            self.assertEqual(section_leaves(rl, section), section_leaves(self.junior, section), section)
        self.assertEqual(rl.obs_space.observation_space, "SubstationGraphObsSpace")
        self.assertEqual(list(rl.model.fcnet_hiddens), [256, 256])

    def test_dropout_is_replaced(self):
        rl = compose_config("config", RL_OVERRIDES + architecture_overrides(self.junior))
        self.assertEqual(rl.model.gnn.dropout_prob, 0.0)
        self.assertEqual(rl.model.gnn.hidden_dim, 512)

    def test_machine_specific_paths_are_skipped(self):
        self.assertFalse(any("edge_probs_path" in o for o in architecture_overrides(self.junior)))

    def test_shell_line_splits_back_into_the_overrides(self):
        overrides = architecture_overrides(self.junior)
        self.assertEqual(shlex.split(shell_overrides(overrides)), overrides)


if __name__ == "__main__":
    unittest.main()


def checkpoint(iteration: int | None, val_survival: float | None) -> dict:
    """Entry of checkpoint_results.json."""
    return {"training_iteration": iteration, "grid2op_end_mean": val_survival}


class TestSelectBestValCheckpoint(unittest.TestCase):
    def test_highest_val_survival_among_evaluation_iterations(self):
        results = {"checkpoint_000001": checkpoint(10, 3000), "checkpoint_000003": checkpoint(30, 5000),
                   "checkpoint_000004": checkpoint(40, 4000)}
        self.assertEqual(select_best_val_checkpoint(results, 10), "checkpoint_000003")

    def test_ignores_checkpoint_with_stale_evaluation(self):
        """The end-of-training checkpoint (iteration 153) repeats the evaluation of iteration 150."""
        results = {"checkpoint_000014": checkpoint(150, 4000), "checkpoint_000015": checkpoint(153, 6000)}
        self.assertEqual(select_best_val_checkpoint(results, 10), "checkpoint_000014")

    def test_later_checkpoint_wins_ties(self):
        results = {"checkpoint_000001": checkpoint(10, 5000), "checkpoint_000002": checkpoint(20, 5000)}
        self.assertEqual(select_best_val_checkpoint(results, 10), "checkpoint_000002")

    def test_without_evaluated_checkpoint_raises(self):
        with self.assertRaises(ValueError):
            select_best_val_checkpoint({"checkpoint_000000": checkpoint(3, 5000)}, 10)
        with self.assertRaises(ValueError):
            select_best_val_checkpoint({"checkpoint_000000": checkpoint(None, None)}, 10)
