"""
Tests for converters that read the backend's grid model (`needs_backend_state`: PTDF, LODF, Zbus).

A live env is driven through multi-element bus splits and a line outage. In every state the
converters' distance matrix (`_cached_D`, before normalisation) must
  - equal a fresh computation, also for a converter that has seen the previous states (the cache
    must notice bus splits, not only line switching), and
  - be reproduced offline from `obs.to_vect()` in another env after `sync_backend_to_observation`.

Tests are organised by class:
  - TestBackendStateConverters
  - TestEnvConverterBackend
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import grid2op
import numpy as np
from lightsim2grid import LightSimBackend

from grid2op_env.env import CustomizedGrid2OpEnvironment
from grid2op_env.observation_converter import (
    _DEFAULT_NODE_FEATURES,
    _FORECAST_NODE_FEATURES,
    ElementLODFGraphObservationConverter,
    LODFGraphObservationConverter,
    PTDFGraphObservationConverter,
    SubstationPTDFGraphObservationConverter,
    SubstationZbusGraphObservationConverter,
    ZbusGraphObservationConverter,
    sync_backend_to_observation,
)

ROOT = Path(__file__).resolve().parents[3]
ENV_NAME = "l2rpn_wcci_2020_train"
ACTION_SPACE = ROOT / "data/action_spaces/l2rpn_wcci_2020/teacher_n1_k208.json"
NO_FORECASTS = [f for f in _DEFAULT_NODE_FEATURES if f not in _FORECAST_NODE_FEATURES]

CONVERTERS = (
    PTDFGraphObservationConverter,
    ZbusGraphObservationConverter,
    LODFGraphObservationConverter,
    ElementLODFGraphObservationConverter,
    SubstationPTDFGraphObservationConverter,
    SubstationZbusGraphObservationConverter,
)


def make_converter(cls, env):
    if cls in (LODFGraphObservationConverter, ElementLODFGraphObservationConverter):
        return cls(env.backend, env.observation_space, use_forecasts=False)
    return cls(env.backend, env.observation_space, attr_to_observe=NO_FORECASTS)


def distance_matrix(converter, obs) -> np.ndarray:
    converter.to_gym(obs)
    return np.array(converter._cached_D, copy=True)


class TestBackendStateConverters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.live = grid2op.make(ENV_NAME, backend=LightSimBackend())
        cls.offline = grid2op.make(ENV_NAME, backend=LightSimBackend())
        cls.offline.reset()
        cls.live.seed(0)
        obs = cls.live.reset()
        # One converter per class sees every state in turn, as in an RL episode.
        cls.persistent = {c: make_converter(c, cls.live) for c in CONVERTERS}
        for c in CONVERTERS:
            cls.persistent[c].to_gym(obs)

        cls.states = []  # (description, obs, {converter class: live fresh matrix, persistent matrix})
        used_subs = set()
        for action_dict in json.loads(ACTION_SPACE.read_text()):
            action = cls.live.action_space(action_dict)
            sub = int(np.flatnonzero(action.get_topological_impact()[1])[0])
            sim_obs, _, done, info = obs.simulate(action)
            n_bus2 = int((sim_obs.topo_vect[cls.live.observation_space.sub_info[:sub].sum():][:cls.live.sub_info[sub]] == 2).sum())
            if done or info["is_illegal"] or sub in used_subs or n_bus2 < 2:
                continue
            obs, _, done, _ = cls.live.step(action)
            assert not done
            used_subs.add(sub)
            cls._record(f"bus split at substation {sub} ({n_bus2} elements on bus 2)", obs)
            if len(used_subs) == 2:
                break
        for line in range(cls.live.n_line):
            action = cls.live.action_space({"set_line_status": [(line, -1)]})
            _, _, done, info = obs.simulate(action)
            if not done and not info["is_illegal"]:
                obs, _, done, _ = cls.live.step(action)
                assert not done
                cls._record(f"line {line} disconnected", obs)
                break

    @classmethod
    def _record(cls, description, obs):
        matrices = {}
        for c in CONVERTERS:
            matrices[c] = (distance_matrix(make_converter(c, cls.live), obs), distance_matrix(cls.persistent[c], obs))
        cls.states.append((description, obs, matrices))

    @classmethod
    def tearDownClass(cls):
        cls.live.close()
        cls.offline.close()

    def test_scenario_covers_bus_splits_and_line_outage(self):
        descriptions = [d for d, _, _ in self.states]
        self.assertEqual(sum("bus split" in d for d in descriptions), 2, descriptions)
        self.assertTrue(any("disconnected" in d for d in descriptions), descriptions)

    def test_cache_follows_topology_changes(self):
        for description, _, matrices in self.states:
            for c, (fresh, persistent) in matrices.items():
                with self.subTest(converter=c.__name__, state=description):
                    np.testing.assert_allclose(persistent, fresh, atol=1e-6)

    def test_offline_conversion_matches_live_after_sync(self):
        for description, obs, matrices in self.states:
            stored = self.offline.observation_space.from_vect(obs.to_vect())
            self.assertTrue(sync_backend_to_observation(self.offline, stored))
            for c, (fresh, _) in matrices.items():
                with self.subTest(converter=c.__name__, state=description):
                    self.assertTrue(c.needs_backend_state)
                    np.testing.assert_allclose(distance_matrix(make_converter(c, self.offline), stored), fresh, atol=1e-6)


class TestEnvConverterBackend(unittest.TestCase):
    """The RL env's converter must read the backend that is stepped, not GymEnv's copy."""

    def test_rl_env_converter_reads_the_stepped_backend(self):
        env = CustomizedGrid2OpEnvironment({
            "env_name": ENV_NAME, "lib_dir": str(ROOT), "action_space": "teacher_n1_k208",
            "observation_space": "PTDFGraphObsSpace", "use_forecasts": False, "grid2op_kwargs": {},
        })
        try:
            self.assertIs(env.observation_converter._grid, env.env_g2op.backend._grid)
        finally:
            env.env_g2op.close()


if __name__ == "__main__":
    unittest.main()
