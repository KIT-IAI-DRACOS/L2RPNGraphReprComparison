"""
Tests for the reward normalization by episode length (grid2op_env/rewards.py, used by grid2op_env/env.py).

Tests are organised by class:
  - TestNormalizeByEpisodeLength
  - TestEnvConfig
  - TestEnvRewardNormalization
"""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from core.constants import DO_NOTHING_AGENT, HIGH_LEVEL_AGENT, RL_AGENT
from core.rllib_config import build_env_config
from grid2op_env.env import CustomizedGrid2OpEnvironment
from grid2op_env.rewards import ConstantReward, normalize_by_episode_length
from tests.hydra_config import compose_config

ROOT = Path(__file__).resolve().parents[3]
ENV_NAME = "l2rpn_wcci_2020_train"


class TestNormalizeByEpisodeLength(unittest.TestCase):
    def test_divides_by_max_step(self):
        self.assertAlmostEqual(normalize_by_episode_length(1.0, SimpleNamespace(max_step=8064)), 1 / 8064)

    def test_rejects_chronics_without_length(self):
        with self.assertRaises(ValueError):
            normalize_by_episode_length(1.0, SimpleNamespace(max_step=-1))


class TestEnvConfig(unittest.TestCase):
    """The config switch reaches the env; configs from before the switch keep the plain reward."""

    def test_switch_is_passed_to_the_env(self):
        cfg = compose_config("config", ["obs_space=graph_comparison/complete", "reward=constant", "reward.normalize_by_episode_length=true"])
        self.assertTrue(build_env_config(cfg, "train")["normalize_reward_by_episode_length"])

    def test_old_configs_without_the_switch_are_not_normalized(self):
        cfg = compose_config("config", ["obs_space=graph_comparison/complete", "reward=constant"])
        OmegaConf.set_struct(cfg, False)
        del cfg.reward["normalize_by_episode_length"]
        self.assertFalse(build_env_config(cfg, "train")["normalize_reward_by_episode_length"])


class TestEnvRewardNormalization(unittest.TestCase):
    """One do-nothing step of the RL env pays 1, or 1 / max_step with the switch on."""

    @classmethod
    def setUpClass(cls):
        cls.env = CustomizedGrid2OpEnvironment({
            "env_name": ENV_NAME, "lib_dir": str(ROOT), "action_space": "teacher_n1_k208",
            "observation_space": "GraphObsSpace", "use_forecasts": False,
            "grid2op_kwargs": {"reward_class": ConstantReward},
            "normalize_reward_by_episode_length": True,
        })

    @classmethod
    def tearDownClass(cls):
        cls.env.env_g2op.close()

    def do_nothing_reward(self) -> float:
        """RL agent's reward for one do-nothing step after a reset."""
        self.env.reset()
        self.env.step({HIGH_LEVEL_AGENT: 1})  # 1: the do-nothing agent acts
        _, rewards, _, _, _ = self.env.step({DO_NOTHING_AGENT: 0})
        return rewards[RL_AGENT]

    def test_reward_is_divided_by_the_chronic_length(self):
        self.env.normalize_reward = True
        reward = self.do_nothing_reward()
        self.assertAlmostEqual(reward, 1 / self.env.cur_g2op_obs.max_step)

    def test_switch_off_keeps_the_reward(self):
        self.env.normalize_reward = False
        self.assertEqual(self.do_nothing_reward(), 1.0)


if __name__ == "__main__":
    unittest.main()
