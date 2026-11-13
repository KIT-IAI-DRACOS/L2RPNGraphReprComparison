"""
Tests for the graph-comparison parts of observation_converter.py: the Complete node features on every
representation, bus merging, the LODF line features, the edgeless and flat controls, and NODE_SUBSTATIONS.

Tests are organised by class:
  - TestAggregateToBuses
  - TestResolveNodeFeatures
  - TestBusGraphsWithCompleteFeatures
  - TestLODFNodeFeatures
  - TestEdgelessAndFlat
  - TestNodeSubstations
"""
from __future__ import annotations

import unittest
import warnings

import grid2op
import numpy as np
from grid2op.gym_compat import GymEnv
from lightsim2grid import LightSimBackend

from grid2op_env.observation_converter import (
    EDGE_MASK, FLAT_FEATURES, NODE_MASK, NODE_SUBSTATIONS, NODES,
    EdgelessGraphObservationConverter,
    FlatGraphObservationConverter,
    GraphObservationConverter,
    LODFGraphObservationConverter,
    PTDFGraphObservationConverter,
    SubstationGraphObservationConverter,
    aggregate_to_buses,
    complete_node_features,
    lodf_node_features,
    make_observation_converter,
    resolve_node_features,
    scale_time_next_maintenance,
)

warnings.filterwarnings("ignore")
ENV_NAME = "l2rpn_case14_sandbox"
env = grid2op.make(ENV_NAME, backend=LightSimBackend())
gym_env = GymEnv(env)
COMPLETE = complete_node_features(use_forecasts=True, has_storage=env.n_storage > 0)


def split_substation(sub_id: int):
    """Observation after moving the first half of the elements of `sub_id` to bus 2 (episode 0)."""
    env.reset(seed=0, options={"time serie id": 0})
    topo = np.ones(env.sub_info[sub_id], dtype=int)
    topo[: env.sub_info[sub_id] // 2] = 2
    obs, _, done, info = env.step(env.action_space({"set_bus": {"substations_id": [(sub_id, topo)]}}))
    assert not done and not info["is_illegal"]
    return obs


def disconnect_line(line_id: int):
    """Observation after disconnecting `line_id` (episode 0)."""
    env.reset(seed=0, options={"time serie id": 0})
    obs, _, done, _ = env.step(env.action_space({"set_line_status": [(line_id, -1)]}))
    assert not done and not obs.line_status[line_id]
    return obs


class TestAggregateToBuses(unittest.TestCase):
    """Merge rules: powers summed, voltage averaged, rho maximum, time to maintenance minimum."""

    def setUp(self):
        self.features = {
            "active_power": np.array([1.0, 2.0, 5.0]),
            "voltage": np.array([10.0, 20.0, 30.0]),
            "rho": np.array([0.2, 0.9, 0.5]),
            "time_next_maintenance": np.array([0.5, 0.1, 1.0]),
        }
        self.slots = np.array([0, 0, 2])
        self.X, self.mask = aggregate_to_buses(self.features, list(self.features), self.slots, n_slots=3)

    def test_rules(self):
        np.testing.assert_allclose(self.X[0], [3.0, 15.0, 0.9, 0.1])
        np.testing.assert_allclose(self.X[2], [5.0, 30.0, 0.5, 1.0])

    def test_empty_slot_is_zero_and_masked(self):
        np.testing.assert_array_equal(self.X[1], 0.0)
        np.testing.assert_array_equal(self.mask, [True, False, True])


class TestResolveNodeFeatures(unittest.TestCase):

    def test_complete(self):
        features = resolve_node_features({"node_features": "complete"}, env.observation_space)
        self.assertEqual(features, COMPLETE)

    def test_complete_without_forecasts(self):
        features = resolve_node_features({"node_features": "complete", "use_forecasts": False}, env.observation_space)
        self.assertNotIn("active_power_forecast", features)

    def test_explicit_list_wins(self):
        config = {"node_features": "complete", "attr_to_observe": ["rho"]}
        self.assertEqual(resolve_node_features(config, env.observation_space), ["rho"])

    def test_default_is_converter_default(self):
        self.assertIsNone(resolve_node_features({}, env.observation_space))

    def test_unknown_value_raises(self):
        with self.assertRaises(ValueError):
            resolve_node_features({"node_features": "all"}, env.observation_space)


class TestBusGraphsWithCompleteFeatures(unittest.TestCase):
    """Substation and PTDF converters merge the Complete features per bus."""

    @classmethod
    def setUpClass(cls):
        cls.obs = split_substation(1)
        cls.converters = {
            "substation": SubstationGraphObservationConverter(env.observation_space, attr_to_observe=COMPLETE),
            "ptdf": PTDFGraphObservationConverter(env.backend, env.observation_space, attr_to_observe=COMPLETE),
        }

    def column(self, X: np.ndarray, name: str) -> np.ndarray:
        return X[:, COMPLETE.index(name)]

    def test_shape(self):
        for name, converter in self.converters.items():
            X, _ = converter._get_nodes_and_mask(self.obs)
            self.assertEqual(X.shape, (2 * env.n_sub, len(COMPLETE)), name)

    def test_bus_rho_is_maximum_of_its_lines(self):
        X, _ = self.converters["substation"]._get_nodes_and_mask(self.obs)
        sub, bus = int(env.line_or_to_subid[0]), int(self.obs.line_or_bus[0])
        lines_on_bus = [l for l in range(env.n_line)
                        if (env.line_or_to_subid[l] == sub and self.obs.line_or_bus[l] == bus)
                        or (env.line_ex_to_subid[l] == sub and self.obs.line_ex_bus[l] == bus)]
        self.assertAlmostEqual(self.column(X, "rho")[2 * sub + bus - 1], self.obs.rho[lines_on_bus].max(), places=5)

    def test_node_types_become_counts(self):
        """The type columns of all buses add up to the number of connected elements of each type."""
        X, _ = self.converters["substation"]._get_nodes_and_mask(self.obs)
        self.assertEqual(self.column(X, "node_type_gen").sum(), (self.obs.gen_bus > 0).sum())
        self.assertEqual(self.column(X, "node_type_line_or").sum(), (self.obs.line_or_bus > 0).sum())

    def test_split_activates_second_bus(self):
        for name, converter in self.converters.items():
            _, mask = converter._get_nodes_and_mask(self.obs)
            self.assertEqual(mask.sum(), env.n_sub + 1, name)

    def test_substation_cooldown_survives_merging(self):
        X, mask = self.converters["substation"]._get_nodes_and_mask(self.obs)
        cooldown = self.column(X, "time_before_cooldown_sub")
        slots = np.arange(2 * env.n_sub)
        np.testing.assert_allclose(cooldown[mask], self.obs.time_before_cooldown_sub[slots[mask] // 2])


class TestLODFNodeFeatures(unittest.TestCase):
    """LODF line nodes: line state of both ends plus the injections at both end buses."""

    def setUp(self):
        self.converter = LODFGraphObservationConverter(env.backend, env.observation_space)

    def column(self, X: np.ndarray, name: str) -> np.ndarray:
        return X[:, self.converter.feature_names.index(name)]

    def test_feature_names(self):
        self.assertEqual(self.converter.feature_names, lodf_node_features(use_forecasts=True))
        self.assertNotIn("injection_active_power_forecast_or", lodf_node_features(use_forecasts=False))

    def test_injection_at_line_end_is_net_injection_of_its_bus(self):
        obs = env.reset(seed=0, options={"time serie id": 0})
        X = self.converter._get_node_features(obs)
        line = 0
        sub = env.line_or_to_subid[line]
        expected = obs.gen_p[env.gen_to_subid == sub].sum() - obs.load_p[env.load_to_subid == sub].sum()
        self.assertAlmostEqual(self.column(X, "injection_active_power_or")[line], expected, places=3)

    def test_disconnected_line_stays_visible_without_edges(self):
        line = 3
        obs = disconnect_line(line)
        g = self.converter.to_gym(obs)
        self.assertTrue(g[NODE_MASK].all())
        X = self.converter._get_node_features(obs)
        self.assertEqual(self.column(X, "line_status")[line], 0.0)
        # Its state is kept, not zeroed: no maintenance planned scales to 1.
        self.assertEqual(self.column(X, "time_next_maintenance")[line],
                         scale_time_next_maintenance(obs.time_next_maintenance)[line])
        self.assertEqual(self.column(X, "time_next_maintenance")[line], 1.0)
        src, dst = g["edge_index"][:, g[EDGE_MASK]]
        self.assertFalse(((src == line) | (dst == line)).any())


class TestEdgelessAndFlat(unittest.TestCase):

    def setUp(self):
        self.obs = env.reset(seed=0, options={"time serie id": 0})

    def test_edgeless_has_no_edges_but_complete_nodes(self):
        converter = EdgelessGraphObservationConverter(env.observation_space, attr_to_observe=COMPLETE)
        g = converter.to_gym(self.obs)
        self.assertEqual(g[EDGE_MASK].sum(), 0)
        self.assertEqual(g[NODES].shape, (converter.num_nodes, len(COMPLETE)))

    def test_flat_is_normalized_graph_features_without_types_plus_bus(self):
        graph = GraphObservationConverter(env.observation_space, attr_to_observe=COMPLETE)
        reference = GraphObservationConverter(env.observation_space, attr_to_observe=COMPLETE)
        flat = FlatGraphObservationConverter(graph).to_gym(self.obs)[FLAT_FEATURES]
        kept = [i for i, name in enumerate(COMPLETE) if not name.startswith("node_type_")]
        np.testing.assert_allclose(flat[:, :-1], reference.to_gym(self.obs)[NODES][:, kept])
        np.testing.assert_array_equal(flat[:, -1], graph.node_buses(self.obs))

    def test_factory(self):
        converter = make_observation_converter(
            gym_env, {"observation_space": "FlatGraphObsSpace", "node_features": "complete"}, backend=env.backend)
        n_cols = len([n for n in COMPLETE if not n.startswith("node_type_")]) + 1
        self.assertEqual(converter.observation_space[FLAT_FEATURES].shape, (converter._graph.num_nodes, n_cols))
        self.assertTrue(converter.observation_space.contains(converter.to_gym(self.obs)))


class TestNodeSubstations(unittest.TestCase):
    """NODE_SUBSTATIONS for every comparison representation, added by `node_substations: true`."""

    MODES = ["GraphObsSpace", "HeterogeneousGraphObsSpace", "EdgelessGraphObsSpace", "SubstationGraphObsSpace",
             "PTDFGraphObsSpace", "ZbusGraphObsSpace", "SubstationPTDFGraphObsSpace", "SubstationZbusGraphObsSpace",
             "ElementGraphObsSpace", "LODFGraphObsSpace", "ElementLODFGraphObsSpace"]

    def setUp(self):
        self.obs = env.reset(seed=0, options={"time serie id": 0})

    def converter(self, mode: str):
        config = {"observation_space": mode, "node_features": "complete", "node_substations": True}
        return make_observation_converter(gym_env, config, backend=env.backend)

    def test_every_substation_has_a_node_and_ids_are_valid(self):
        for mode in self.MODES:
            with self.subTest(mode=mode):
                converter = self.converter(mode)
                g = converter.to_gym(self.obs)
                self.assertTrue(converter.observation_space.contains(g))
                subs = g[NODE_SUBSTATIONS]
                self.assertEqual(subs.shape, (g[NODES].shape[0], 2))
                self.assertTrue(((subs >= -1) & (subs < env.n_sub)).all())
                active = subs[g[NODE_MASK]]
                self.assertEqual(set(active[active >= 0].tolist()), set(range(env.n_sub)))

    def test_bus_slot_conventions(self):
        """Substation slots are 2*sub + bus - 1, PTDF slots (bus - 1)*n_sub + sub."""
        substation = self.converter("SubstationGraphObsSpace").to_gym(self.obs)[NODE_SUBSTATIONS][:, 0]
        ptdf = self.converter("PTDFGraphObsSpace").to_gym(self.obs)[NODE_SUBSTATIONS][:, 0]
        np.testing.assert_array_equal(substation, np.repeat(np.arange(env.n_sub), 2))
        np.testing.assert_array_equal(ptdf, np.tile(np.arange(env.n_sub), 2))

    def test_lodf_lines_have_both_ends(self):
        subs = self.converter("LODFGraphObsSpace").to_gym(self.obs)[NODE_SUBSTATIONS]
        np.testing.assert_array_equal(subs, np.stack([env.line_or_to_subid, env.line_ex_to_subid], axis=1))

    def test_off_by_default(self):
        converter = make_observation_converter(gym_env, {"observation_space": "GraphObsSpace"}, backend=env.backend)
        self.assertNotIn(NODE_SUBSTATIONS, converter.observation_space.spaces)


if __name__ == "__main__":
    unittest.main()
