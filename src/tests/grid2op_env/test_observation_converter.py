"""
Unit tests for observation_converter.py.

Tests are organised by class:
  - TestGridDimensions
  - TestGraphObservationConverterSpace
  - TestGraphObservationConverterNodeFeatures
  - TestGraphObservationConverterEdgeIndex
  - TestGraphObservationConverterNormalize
  - TestGraphObservationConverterIntegration
"""
from __future__ import annotations

import unittest

import grid2op
import numpy as np

from grid2op_env.observation_converter import (
    GraphObservationConverter,
    HeterogeneousGraphObservationConverter,
    SubstationGraphObservationConverter,
    ElementGraphObservationConverter,
    _GridDimensions,
    NODES, EDGE_INDEX, EDGE_MASK, NODE_MASK, EDGE_TYPE, EDGES, GLOBAL,
    _DEFAULT_NODE_FEATURES,
)

ENV_NAME = "l2rpn_wcci_2020"
env = grid2op.make(ENV_NAME)


class TestGridDimensions(unittest.TestCase):
    """Tests for the _GridDimensions dataclass."""

    def test_from_obs_space(self):
        """from_obs_space reads the correct attributes."""
        dims = _GridDimensions.from_obs_space(env.observation_space)
        self.assertEqual(dims.n_gen, env.n_gen)
        self.assertEqual(dims.n_load, env.n_load)
        self.assertEqual(dims.n_line, env.n_line)
        self.assertEqual(dims.n_storage, env.n_storage)


class TestGraphObservationConverterSpace(unittest.TestCase):
    """Tests for the gymnasium observation space definition."""

    def setUp(self):
        self.converter = GraphObservationConverter(env.observation_space)

    def test_node_feature_shape(self):
        """NODES box has shape (num_nodes, x_dim)."""
        x_dim = len(_DEFAULT_NODE_FEATURES)
        shape = self.converter.observation_space[NODES].shape
        self.assertEqual(shape, (self.converter.num_nodes, x_dim))

    def test_edge_index_shape(self):
        """EDGE_INDEX box has shape (2, max_num_edges)."""
        shape = self.converter.observation_space[EDGE_INDEX].shape
        self.assertEqual(shape[0], 2)
        self.assertGreater(shape[1], 0)

    def test_edge_mask_shape(self):
        """EDGE_MASK box length matches EDGE_INDEX width."""
        ei_shape = self.converter.observation_space[EDGE_INDEX].shape
        mask_shape = self.converter.observation_space[EDGE_MASK].shape
        self.assertEqual(mask_shape[0], ei_shape[1])

    def test_node_mask_shape(self):
        """NODE_MASK box has shape (num_nodes,)."""
        shape = self.converter.observation_space[NODE_MASK].shape
        self.assertEqual(shape, (self.converter.num_nodes,))

    def test_node_mask_matches_nodes_first_dim(self):
        """NODE_MASK length equals the first dim of NODES."""
        nodes_shape = self.converter.observation_space[NODES].shape
        mask_shape = self.converter.observation_space[NODE_MASK].shape
        self.assertEqual(mask_shape[0], nodes_shape[0])

    def test_max_nodes_equals_num_nodes(self):
        """max_nodes property matches num_nodes for the default converter."""
        self.assertEqual(self.converter.max_nodes, self.converter.num_nodes)

    def test_global_features_shape(self):
        """GLOBAL box has shape (6,)."""
        self.assertEqual(self.converter.observation_space[GLOBAL].shape, (6,))

    def test_custom_attr_to_observe(self):
        """x_dim matches the length of a custom attr_to_observe list."""
        attrs = ["active_power", "rho"]
        converter = GraphObservationConverter(env.observation_space, attr_to_observe=attrs)
        self.assertEqual(converter.x_dim, 2)


class TestGraphObservationConverterNodeFeatures(unittest.TestCase):
    """Tests for _compute_all_node_features and _get_node_features."""

    def setUp(self):
        self.converter = GraphObservationConverter(env.observation_space)
        self.dim = _GridDimensions.from_obs_space(env.observation_space)
        self.obs = env.reset()

    def test_node_feature_matrix_shape(self):
        """_get_node_features returns shape (num_nodes, x_dim)."""
        feats = self.converter._get_node_features(self.obs)
        num_nodes = self.converter.num_nodes
        x_dim = self.converter.x_dim
        self.assertEqual(feats.shape, (num_nodes, x_dim))

    def test_node_feature_dtype(self):
        """Node features are float32."""
        feats = self.converter._get_node_features(self.obs)
        self.assertEqual(feats.dtype, np.float32)

    def test_all_features_same_length(self):
        """Every feature vector in _compute_all_node_features has length num_nodes."""
        all_feats = self.converter._compute_all_node_features(self.obs)
        expected_len = self.converter.num_nodes
        for name, arr in all_feats.items():
            with self.subTest(feature=name):
                self.assertEqual(len(arr), expected_len, f"{name} has wrong length")

    def test_rho_zero_for_non_line_nodes(self):
        """rho is 0 for gen and load nodes."""
        all_feats = self.converter._compute_all_node_features(self.obs)
        rho = all_feats["rho"]
        load_idx = np.arange(start=0, stop=self.dim.n_load) + 2 * self.dim.n_line + self.dim.n_gen
        gen_idx = np.arange(start=0, stop=self.dim.n_gen) + 2 * self.dim.n_line
        np.testing.assert_array_equal(rho[gen_idx],  0.0)
        np.testing.assert_array_equal(rho[load_idx], 0.0)

    def test_rho_nonzero_for_line_nodes(self):
        """rho is copied to both line_or and line_ex nodes."""
        all_feats = self.converter._compute_all_node_features(self.obs)
        rho = all_feats["rho"]
        or_idx = np.arange(start=0, stop=self.dim.n_line)
        ex_idx = np.arange(start=0, stop=self.dim.n_line) + self.dim.n_line
        np.testing.assert_array_equal(rho[or_idx], self.obs.rho)
        np.testing.assert_array_equal(rho[ex_idx], self.obs.rho)

    def test_active_power_sign_convention(self):
        """Loads have negative active_power (consumption), gens positive."""
        all_feats = self.converter._compute_all_node_features(self.obs)
        p = all_feats["active_power"]
        load_idx = np.arange(start=0, stop=self.dim.n_load) + 2*self.dim.n_line + self.dim.n_gen
        gen_idx  = np.arange(start=0, stop=self.dim.n_gen) + 2*self.dim.n_line
        np.testing.assert_array_almost_equal(p[load_idx], -self.obs.load_p)
        np.testing.assert_array_almost_equal(p[gen_idx],   self.obs.gen_p)

    def test_only_requested_features_in_output(self):
        """_get_node_features only returns columns for attr_to_observe."""
        attrs = ["active_power", "rho"]
        obs_space = env.observation_space
        converter = GraphObservationConverter(obs_space, attr_to_observe=attrs)
        feats = converter._get_node_features(self.obs)
        self.assertEqual(feats.shape[1], 2)


class TestGraphObservationConverterEdgeIndex(unittest.TestCase):
    """Tests for _get_edge_index."""

    def setUp(self):
        self.converter = GraphObservationConverter(env.observation_space)
        self.obs = env.reset()

    def test_edge_index_shape(self):
        """Edge index has exactly 2 rows."""
        ei = self.converter._get_edge_index(self.obs)
        self.assertEqual(ei.shape[0], 2)

    def test_edge_index_dtype(self):
        """Edge index dtype is int64."""
        ei = self.converter._get_edge_index(self.obs)
        self.assertEqual(ei.dtype, np.int64)

    def test_edge_index_within_node_range(self):
        """All edge indices are valid node indices."""
        ei = self.converter._get_edge_index(self.obs)
        self.assertTrue((ei >= 0).all())
        self.assertTrue((ei < self.converter.num_nodes).all())

    def test_edges_are_bidirectional(self):
        """For every (i, j) edge there is a corresponding (j, i) edge."""
        ei = self.converter._get_edge_index(self.obs)
        edge_set = set(zip(ei[0].tolist(), ei[1].tolist()))
        for src, dst in list(edge_set):
            self.assertIn((dst, src), edge_set, f"Missing reverse edge ({dst}, {src})")

    def test_no_self_loops(self):
        """No node is connected to itself."""
        ei = self.converter._get_edge_index(self.obs)
        self.assertTrue((ei[0] != ei[1]).all())

    def test_num_edges_does_not_exceed_max(self):
        """Number of edges never exceeds max_num_edges."""
        ei = self.converter._get_edge_index(self.obs)
        self.assertLessEqual(ei.shape[1], self.converter.max_num_edges)


class TestGraphObservationConverterNormalize(unittest.TestCase):
    """Tests for the normalize method and running statistics."""

    def setUp(self):
        self.converter = GraphObservationConverter(env.observation_space)
        self.obs = env.reset()

    def _raw_obs(self) -> dict:
        """Build a raw (un-normalized) observation dict."""
        node_features = self.converter._get_node_features(self.obs)
        ei = self.converter._get_edge_index(self.obs)
        num_edges = ei.shape[1]
        ei_padded = np.zeros((2, self.converter.max_num_edges), dtype=np.int64)
        ei_padded[:, :num_edges] = ei
        mask = np.zeros(self.converter.max_num_edges, dtype=bool)
        mask[:num_edges] = True
        node_mask = np.ones(self.converter.num_nodes, dtype=np.bool_)
        return {
            NODES: node_features,
            EDGE_INDEX: ei_padded,
            EDGE_MASK: mask,
            NODE_MASK: node_mask,
            GLOBAL: self.converter._get_global_features(self.obs),
        }

    def test_normalize_returns_all_keys(self):
        """normalize returns a dict with NODES, EDGE_INDEX, EDGE_MASK, NODE_MASK, GLOBAL."""
        result = self.converter.normalize(self._raw_obs())
        self.assertIn(NODES, result)
        self.assertIn(EDGE_INDEX, result)
        self.assertIn(EDGE_MASK, result)
        self.assertIn(NODE_MASK, result)
        self.assertIn(GLOBAL, result)

    def test_normalize_does_not_alter_node_mask(self):
        """normalize passes NODE_MASK through unchanged."""
        raw = self._raw_obs()
        result = self.converter.normalize(raw)
        np.testing.assert_array_equal(result[NODE_MASK], raw[NODE_MASK])

    def test_normalized_node_features_dtype(self):
        """Normalized node features are float32."""
        result = self.converter.normalize(self._raw_obs())
        self.assertEqual(result[NODES].dtype, np.float32)

    def test_normalize_does_not_alter_edge_index(self):
        """normalize passes EDGE_INDEX through unchanged."""
        raw = self._raw_obs()
        result = self.converter.normalize(raw)
        np.testing.assert_array_equal(result[EDGE_INDEX], raw[EDGE_INDEX])

    def test_normalize_does_not_alter_edge_mask(self):
        """normalize passes EDGE_MASK through unchanged."""
        raw = self._raw_obs()
        result = self.converter.normalize(raw)
        np.testing.assert_array_equal(result[EDGE_MASK], raw[EDGE_MASK])

    def test_running_stats_updated_after_normalize(self):
        """Calling normalize increments the running count."""
        count_before = self.converter._normalizer.count
        self.converter.normalize(self._raw_obs())
        self.assertGreater(self.converter._normalizer.count, count_before)

    def test_normalized_shape_preserved(self):
        """normalize preserves the NODES shape."""
        raw = self._raw_obs()
        result = self.converter.normalize(raw)
        self.assertEqual(result[NODES].shape, raw[NODES].shape)


class TestGraphObservationConverterIntegration(unittest.TestCase):
    """End-to-end tests for to_gym."""

    def setUp(self):
        self.converter = GraphObservationConverter(env.observation_space)
        self.obs = env.reset()

    def test_to_gym_returns_all_keys(self):
        """to_gym returns a dict with all expected keys."""
        result = self.converter.to_gym(self.obs)
        for key in [NODES, EDGE_INDEX, EDGE_MASK, NODE_MASK, GLOBAL]:
            self.assertIn(key, result)

    def test_to_gym_node_mask_all_true(self):
        """NODE_MASK is all-True for the default converter (all nodes are real)."""
        result = self.converter.to_gym(self.obs)
        self.assertTrue(result[NODE_MASK].all())
        self.assertEqual(result[NODE_MASK].shape, (self.converter.num_nodes,))

    def test_to_gym_node_shape(self):
        """to_gym NODES has shape (num_nodes, x_dim)."""
        result = self.converter.to_gym(self.obs)
        self.assertEqual(result[NODES].shape, (self.converter.num_nodes, self.converter.x_dim))

    def test_to_gym_edge_index_padded(self):
        """to_gym EDGE_INDEX is padded to max_num_edges."""
        result = self.converter.to_gym(self.obs)
        self.assertEqual(result[EDGE_INDEX].shape[1], self.converter.max_num_edges)

    def test_to_gym_edge_mask_count_matches_real_edges(self):
        """The number of True values in EDGE_MASK equals the real edge count."""
        raw_ei = self.converter._get_edge_index(self.obs)
        result = self.converter.to_gym(self.obs)
        self.assertEqual(result[EDGE_MASK].sum(), raw_ei.shape[1])

    def test_to_gym_global_features_values(self):
        """Global features contain the correct time values."""
        result = self.converter.to_gym(self.obs)
        gf = result[GLOBAL]
        self.assertEqual(gf[0], self.obs.year)
        self.assertEqual(gf[1], self.obs.month)
        self.assertEqual(gf[2], self.obs.day)
        self.assertEqual(gf[3], self.obs.hour_of_day)
        self.assertEqual(gf[4], self.obs.day_of_week)
        self.assertEqual(gf[5], self.obs.minute_of_hour)

    def test_to_gym_deterministic(self):
        """Two calls with the same observation produce the same EDGE_INDEX."""
        r1 = self.converter.to_gym(self.obs)
        r2 = self.converter.to_gym(self.obs)
        np.testing.assert_array_equal(r1[EDGE_INDEX], r2[EDGE_INDEX])


class TestGraphObservationConverterWithGrid2op(unittest.TestCase):
    """Integration tests that run against a real grid2op environment."""

    @classmethod
    def setUpClass(cls):
        cls.converter = GraphObservationConverter(env.observation_space)

    def test_connectivity_matrix_shape_matches_num_nodes(self):
        """connectivity_matrix shape matches num_nodes."""
        obs = env.reset()
        conn = obs.connectivity_matrix()
        n = self.converter.num_nodes
        self.assertEqual(conn.shape, (n, n))

    def test_pos_topo_vect_cover_all_nodes(self):
        """pos_topo_vect arrays together cover every index in [0, num_nodes)."""
        all_positions = np.concatenate([
            env.load_pos_topo_vect,
            env.gen_pos_topo_vect,
            env.line_or_pos_topo_vect,
            env.line_ex_pos_topo_vect,
            env.storage_pos_topo_vect,
        ])
        expected = set(range(self.converter.num_nodes))
        self.assertEqual(set(all_positions.tolist()), expected)

    def test_line_endpoints_connected_in_connectivity_matrix(self):
        """Every line's origin and extremity are connected in connectivity_matrix."""
        obs = env.reset()
        conn = obs.connectivity_matrix().astype(bool)
        for line_id in range(env.n_line):
            or_idx = env.line_or_pos_topo_vect[line_id]
            ex_idx = env.line_ex_pos_topo_vect[line_id]
            self.assertTrue(
                conn[or_idx, ex_idx],
                f"line {line_id}: origin {or_idx} and extremity {ex_idx} not connected",
            )

    def test_to_gym_observation_space_compliant(self):
        """to_gym output lies within the declared observation space."""
        obs = env.reset()
        result = self.converter.to_gym(obs)
        space = self.converter.observation_space
        for key in [EDGE_INDEX, EDGE_MASK, NODE_MASK, GLOBAL]:
            self.assertEqual(result[key].shape, space[key].shape)


# ---------------------------------------------------------------------------
# HeterogeneousGraphObservationConverter
# ---------------------------------------------------------------------------

class TestHeterogeneousGraphObservationConverterSpace(unittest.TestCase):
    """Observation space definition for HeterogeneousGraphObservationConverter."""

    def setUp(self):
        self.converter = HeterogeneousGraphObservationConverter(env.observation_space)

    def test_edge_type_key_present(self):
        """Observation space must include EDGE_TYPE."""
        self.assertIn(EDGE_TYPE, self.converter.observation_space.spaces)

    def test_edge_type_shape_matches_max_num_edges(self):
        """EDGE_TYPE length equals max_num_edges."""
        et_shape = self.converter.observation_space[EDGE_TYPE].shape
        self.assertEqual(et_shape, (self.converter.max_num_edges,))

    def test_inherits_graph_obs_keys(self):
        """All keys from GraphObservationConverter are still present."""
        for key in [NODES, EDGE_INDEX, EDGE_MASK, NODE_MASK, GLOBAL]:
            self.assertIn(key, self.converter.observation_space.spaces)

    def test_node_shape_unchanged(self):
        """NODES shape is identical to the parent converter."""
        parent = GraphObservationConverter(env.observation_space)
        self.assertEqual(
            self.converter.observation_space[NODES].shape,
            parent.observation_space[NODES].shape,
        )


class TestHeterogeneousGraphObservationConverterEdgeTypes(unittest.TestCase):
    """Tests for _get_edge_index_with_types."""

    def setUp(self):
        self.converter = HeterogeneousGraphObservationConverter(env.observation_space)
        self.obs = env.reset()

    def test_returns_two_arrays(self):
        """_get_edge_index_with_types returns (edge_index, edge_types)."""
        result = self.converter._get_edge_index_with_types(self.obs)
        self.assertEqual(len(result), 2)

    def test_edge_index_and_types_same_length(self):
        """edge_index columns equals edge_types length."""
        ei, et = self.converter._get_edge_index_with_types(self.obs)
        self.assertEqual(ei.shape[1], et.shape[0])

    def test_edge_types_valid_values(self):
        """All edge types are 0, 1, or 2."""
        _, et = self.converter._get_edge_index_with_types(self.obs)
        self.assertTrue(np.all((et >= 0) & (et <= 2)))

    def test_powerline_edges_are_type_0(self):
        """Type-0 edges connect line_or and line_ex node slots."""
        ei, et = self.converter._get_edge_index_with_types(self.obs)
        n_line = env.n_line
        line_or_nodes = set(range(n_line))
        line_ex_nodes = set(range(n_line, 2 * n_line))
        type0_src = ei[0, et == 0].tolist()
        type0_dst = ei[1, et == 0].tolist()
        for s, d in zip(type0_src, type0_dst):
            self.assertTrue(
                (s in line_or_nodes and d in line_ex_nodes) or
                (s in line_ex_nodes and d in line_or_nodes),
                f"Type-0 edge ({s}, {d}) is not a line endpoint pair",
            )

    def test_types_0_and_1_present_in_default_obs(self):
        """Types 0 (powerlines) and 1 (same-bus) appear in the default observation."""
        _, et = self.converter._get_edge_index_with_types(self.obs)
        for t in [0, 1]:
            self.assertIn(t, et.tolist(), f"Edge type {t} not present")

    def test_type_2_present_after_split_substation(self):
        """Type-2 edges (diff-bus) appear after a topology action splits a substation."""
        # Move the first generator to bus 2 at its substation to create a split
        action = env.action_space({"set_bus": {"generators_id": [(0, 2)]}})
        obs, _, done, _ = env.step(action)
        if not done:
            _, et = self.converter._get_edge_index_with_types(obs)
            self.assertIn(2, et.tolist(), "Edge type 2 not present after topology split")

    def test_edges_bidirectional(self):
        """For every (i, j) edge there is a matching (j, i) edge of the same type."""
        ei, et = self.converter._get_edge_index_with_types(self.obs)
        forward = set(zip(ei[0].tolist(), ei[1].tolist(), et.tolist()))
        for s, d, t in list(forward):
            self.assertIn((d, s, t), forward, f"Missing reverse edge ({d}, {s}, type={t})")

    def test_no_self_loops(self):
        """No edge connects a node to itself."""
        ei, _ = self.converter._get_edge_index_with_types(self.obs)
        self.assertTrue((ei[0] != ei[1]).all())

    def test_does_not_exceed_max_num_edges(self):
        """Total edges do not exceed max_num_edges."""
        ei, _ = self.converter._get_edge_index_with_types(self.obs)
        self.assertLessEqual(ei.shape[1], self.converter.max_num_edges)


class TestHeterogeneousGraphObservationConverterIntegration(unittest.TestCase):
    """End-to-end tests for HeterogeneousGraphObservationConverter.to_gym."""

    def setUp(self):
        self.converter = HeterogeneousGraphObservationConverter(env.observation_space)
        self.obs = env.reset()

    def test_to_gym_returns_all_keys(self):
        """to_gym returns all expected keys including EDGE_TYPE."""
        result = self.converter.to_gym(self.obs)
        for key in [NODES, EDGE_INDEX, EDGE_MASK, NODE_MASK, EDGE_TYPE, GLOBAL]:
            self.assertIn(key, result)

    def test_edge_type_padded_length(self):
        """EDGE_TYPE is padded to max_num_edges."""
        result = self.converter.to_gym(self.obs)
        self.assertEqual(result[EDGE_TYPE].shape[0], self.converter.max_num_edges)

    def test_edge_type_only_valid_in_masked_region(self):
        """Edge types beyond the mask are 0 (padding default)."""
        result = self.converter.to_gym(self.obs)
        mask = result[EDGE_MASK]
        padding_types = result[EDGE_TYPE][~mask]
        self.assertTrue((padding_types == 0).all())

    def test_edge_type_consistent_with_edge_mask(self):
        """The count of non-zero mask entries matches _get_edge_index_with_types output."""
        ei, _ = self.converter._get_edge_index_with_types(self.obs)
        result = self.converter.to_gym(self.obs)
        self.assertEqual(result[EDGE_MASK].sum(), ei.shape[1])

    def test_normalize_passes_edge_type_through(self):
        """normalize does not alter EDGE_TYPE."""
        result = self.converter.to_gym(self.obs)
        # Call normalize again with the already-normalized obs to check passthrough
        et_before = result[EDGE_TYPE].copy()
        renormalized = self.converter.normalize(result)
        np.testing.assert_array_equal(renormalized[EDGE_TYPE], et_before)

    def test_node_mask_all_true(self):
        """NODE_MASK is all-True (all element-level nodes exist)."""
        result = self.converter.to_gym(self.obs)
        self.assertTrue(result[NODE_MASK].all())


# ---------------------------------------------------------------------------
# SubstationGraphObservationConverter
# ---------------------------------------------------------------------------

class TestSubstationGraphObservationConverterSpace(unittest.TestCase):
    """Observation space definition for SubstationGraphObservationConverter."""

    def setUp(self):
        self.converter = SubstationGraphObservationConverter(env.observation_space)

    def test_max_nodes_is_twice_n_sub(self):
        """max_nodes equals 2 * n_sub."""
        self.assertEqual(self.converter.max_nodes, 2 * env.n_sub)

    def test_node_shape(self):
        """NODES box has shape (2*n_sub, x_dim)."""
        shape = self.converter.observation_space[NODES].shape
        self.assertEqual(shape, (2 * env.n_sub, len(_DEFAULT_NODE_FEATURES)))

    def test_node_mask_shape(self):
        """NODE_MASK has shape (2*n_sub,)."""
        shape = self.converter.observation_space[NODE_MASK].shape
        self.assertEqual(shape, (2 * env.n_sub,))

    def test_max_num_edges_is_twice_n_line(self):
        """max_num_edges equals 2 * n_line (bidirectional line edges)."""
        self.assertEqual(self.converter.max_num_edges, 2 * env.n_line)

    def test_edge_index_shape(self):
        """EDGE_INDEX box has shape (2, 2*n_line)."""
        shape = self.converter.observation_space[EDGE_INDEX].shape
        self.assertEqual(shape, (2, 2 * env.n_line))

    def test_global_features_shape(self):
        """GLOBAL box has shape (6,)."""
        self.assertEqual(self.converter.observation_space[GLOBAL].shape, (6,))

    def test_smaller_than_element_graph(self):
        """Substation graph has fewer max nodes than the element-level graph."""
        element_converter = GraphObservationConverter(env.observation_space)
        self.assertLess(self.converter.max_nodes, element_converter.num_nodes)


class TestSubstationGraphObservationConverterNodesAndMask(unittest.TestCase):
    """Tests for _get_nodes_and_mask."""

    def setUp(self):
        self.converter = SubstationGraphObservationConverter(env.observation_space)
        self.obs = env.reset()

    def test_node_features_shape(self):
        """node_features has shape (max_nodes, x_dim)."""
        feats, _ = self.converter._get_nodes_and_mask(self.obs)
        self.assertEqual(feats.shape, (self.converter.max_nodes, self.converter.x_dim))

    def test_node_features_dtype(self):
        """node_features is float32."""
        feats, _ = self.converter._get_nodes_and_mask(self.obs)
        self.assertEqual(feats.dtype, np.float32)

    def test_node_mask_shape(self):
        """node_mask has shape (max_nodes,)."""
        _, mask = self.converter._get_nodes_and_mask(self.obs)
        self.assertEqual(mask.shape, (self.converter.max_nodes,))

    def test_node_mask_dtype(self):
        """node_mask is boolean."""
        _, mask = self.converter._get_nodes_and_mask(self.obs)
        self.assertEqual(mask.dtype, np.bool_)

    def test_at_least_n_sub_active_nodes(self):
        """In a connected grid there is at least one active node per substation."""
        _, mask = self.converter._get_nodes_and_mask(self.obs)
        self.assertGreaterEqual(mask.sum(), env.n_sub)

    def test_active_nodes_at_most_twice_n_sub(self):
        """Active nodes never exceed 2 * n_sub."""
        _, mask = self.converter._get_nodes_and_mask(self.obs)
        self.assertLessEqual(mask.sum(), 2 * env.n_sub)

    def test_inactive_node_features_are_zero(self):
        """Inactive bus slots have zero node features (no spurious aggregation)."""
        feats, mask = self.converter._get_nodes_and_mask(self.obs)
        inactive_feats = feats[~mask]
        np.testing.assert_array_equal(inactive_feats, 0.0)


class TestSubstationGraphObservationConverterEdgeIndex(unittest.TestCase):
    """Tests for SubstationGraphObservationConverter._get_edge_index."""

    def setUp(self):
        self.converter = SubstationGraphObservationConverter(env.observation_space)
        self.obs = env.reset()

    def test_edge_index_shape(self):
        """Edge index has exactly 2 rows."""
        ei = self.converter._get_edge_index(self.obs)
        self.assertEqual(ei.shape[0], 2)

    def test_edge_index_dtype(self):
        """Edge index is int64."""
        ei = self.converter._get_edge_index(self.obs)
        self.assertEqual(ei.dtype, np.int64)

    def test_edge_index_within_node_range(self):
        """All edge indices are valid node slots."""
        ei = self.converter._get_edge_index(self.obs)
        self.assertTrue((ei >= 0).all())
        self.assertTrue((ei < self.converter.max_nodes).all())

    def test_edges_bidirectional(self):
        """For every (i, j) edge there is a corresponding (j, i) edge."""
        ei = self.converter._get_edge_index(self.obs)
        edge_set = set(zip(ei[0].tolist(), ei[1].tolist()))
        for s, d in list(edge_set):
            self.assertIn((d, s), edge_set, f"Missing reverse edge ({d}, {s})")

    def test_num_edges_at_most_twice_n_line(self):
        """At most 2 * n_line edges (one bidirectional pair per connected line)."""
        ei = self.converter._get_edge_index(self.obs)
        self.assertLessEqual(ei.shape[1], 2 * env.n_line)

    def test_edge_endpoints_are_active_nodes(self):
        """Every edge endpoint has an active node (node_mask=True)."""
        ei = self.converter._get_edge_index(self.obs)
        _, mask = self.converter._get_nodes_and_mask(self.obs)
        for node in ei.flatten().tolist():
            self.assertTrue(mask[node], f"Edge endpoint {node} is not an active node")


class TestSubstationGraphObservationConverterIntegration(unittest.TestCase):
    """End-to-end tests for SubstationGraphObservationConverter.to_gym."""

    def setUp(self):
        self.converter = SubstationGraphObservationConverter(env.observation_space)
        self.obs = env.reset()

    def test_to_gym_returns_all_keys(self):
        """to_gym returns all expected keys."""
        result = self.converter.to_gym(self.obs)
        for key in [NODES, EDGE_INDEX, EDGE_MASK, NODE_MASK, GLOBAL]:
            self.assertIn(key, result)

    def test_to_gym_node_shape(self):
        """to_gym NODES has shape (max_nodes, x_dim)."""
        result = self.converter.to_gym(self.obs)
        self.assertEqual(result[NODES].shape, (self.converter.max_nodes, self.converter.x_dim))

    def test_to_gym_node_mask_partial(self):
        """NODE_MASK is not necessarily all-True (variable active nodes)."""
        result = self.converter.to_gym(self.obs)
        # At least one node is active, but possibly not all slots
        self.assertTrue(result[NODE_MASK].any())
        self.assertEqual(result[NODE_MASK].shape, (self.converter.max_nodes,))

    def test_to_gym_edge_index_padded(self):
        """EDGE_INDEX is padded to max_num_edges."""
        result = self.converter.to_gym(self.obs)
        self.assertEqual(result[EDGE_INDEX].shape, (2, self.converter.max_num_edges))

    def test_to_gym_edge_mask_count_matches_real_edges(self):
        """True count in EDGE_MASK matches the real edge count."""
        raw_ei = self.converter._get_edge_index(self.obs)
        result = self.converter.to_gym(self.obs)
        self.assertEqual(result[EDGE_MASK].sum(), raw_ei.shape[1])

    def test_to_gym_inactive_nodes_zeroed(self):
        """Inactive node slots have zero features after normalization."""
        result = self.converter.to_gym(self.obs)
        mask = result[NODE_MASK]
        np.testing.assert_array_equal(result[NODES][~mask], 0.0)

    def test_to_gym_deterministic(self):
        """Two calls with the same observation produce the same EDGE_INDEX."""
        r1 = self.converter.to_gym(self.obs)
        r2 = self.converter.to_gym(self.obs)
        np.testing.assert_array_equal(r1[EDGE_INDEX], r2[EDGE_INDEX])

    def test_to_gym_observation_space_compliant(self):
        """to_gym output shapes match the declared observation space."""
        result = self.converter.to_gym(self.obs)
        space = self.converter.observation_space
        for key in [EDGE_INDEX, EDGE_MASK, NODE_MASK, GLOBAL]:
            self.assertEqual(result[key].shape, space[key].shape)


# ---------------------------------------------------------------------------
# ElementGraphObservationConverter
# ---------------------------------------------------------------------------

class TestElementGraphObservationConverterSpace(unittest.TestCase):
    """Observation space definition for ElementGraphObservationConverter."""

    def setUp(self):
        self.converter = ElementGraphObservationConverter(env.observation_space)

    def test_num_nodes(self):
        """num_nodes = n_gen + n_load + n_line + n_storage + 3*n_sub."""
        expected = env.n_gen + env.n_load + env.n_line + env.n_storage + 3 * env.n_sub
        self.assertEqual(self.converter.num_nodes, expected)

    def test_x_dim(self):
        """x_dim equals ElementGraphObservationConverter._ELEM_X_DIM (28)."""
        self.assertEqual(self.converter.x_dim, ElementGraphObservationConverter._ELEM_X_DIM)

    def test_node_shape(self):
        """NODES box has shape (num_nodes, 28)."""
        self.assertEqual(
            self.converter.observation_space[NODES].shape,
            (self.converter.num_nodes, ElementGraphObservationConverter._ELEM_X_DIM),
        )

    def test_edge_index_shape(self):
        """EDGE_INDEX box has shape (2, num_edges)."""
        shape = self.converter.observation_space[EDGE_INDEX].shape
        self.assertEqual(shape[0], 2)
        self.assertEqual(shape[1], self.converter.num_edges)

    def test_edge_mask_shape(self):
        """EDGE_MASK has length num_edges."""
        self.assertEqual(
            self.converter.observation_space[EDGE_MASK].shape,
            (self.converter.num_edges,),
        )

    def test_edge_attr_shape(self):
        """EDGE_ATTR box has shape (num_edges, 1)."""
        self.assertEqual(
            self.converter.observation_space[EDGES].shape,
            (self.converter.num_edges, 1),
        )

    def test_node_mask_shape(self):
        """NODE_MASK has length num_nodes."""
        self.assertEqual(
            self.converter.observation_space[NODE_MASK].shape,
            (self.converter.num_nodes,),
        )

    def test_global_shape(self):
        """GLOBAL box has shape (6,)."""
        self.assertEqual(self.converter.observation_space[GLOBAL].shape, (6,))

    def test_all_keys_present(self):
        """All expected keys are present in the observation space."""
        for key in [NODES, EDGE_INDEX, EDGE_MASK, EDGES, NODE_MASK, GLOBAL]:
            self.assertIn(key, self.converter.observation_space.spaces)


class TestElementGraphObservationConverterNodeFeatures(unittest.TestCase):
    """Tests for _get_node_features."""

    def setUp(self):
        self.converter = ElementGraphObservationConverter(env.observation_space)
        self.obs = env.reset()
        self.X = self.converter._get_node_features(self.obs)

    def test_shape(self):
        """Node feature matrix has shape (num_nodes, 28)."""
        self.assertEqual(self.X.shape, (self.converter.num_nodes, ElementGraphObservationConverter._ELEM_X_DIM))

    def test_dtype(self):
        """Node features are float32."""
        self.assertEqual(self.X.dtype, np.float32)

    def test_gen_specific_slots_zero_for_non_gen(self):
        """Generator-specific slots are zero for non-generator nodes."""
        non_gen = np.ones(self.converter.num_nodes, dtype=bool)
        non_gen[self.converter._gen_offset:self.converter._gen_offset + env.n_gen] = False
        np.testing.assert_array_equal(self.X[non_gen, ElementGraphObservationConverter._ELEM_GEN_SLICE], 0.0)

    def test_line_specific_slots_zero_for_non_line(self):
        """Powerline-specific slots are zero for non-powerline nodes."""
        non_line = np.ones(self.converter.num_nodes, dtype=bool)
        non_line[self.converter._line_offset:self.converter._line_offset + env.n_line] = False
        np.testing.assert_array_equal(self.X[non_line, ElementGraphObservationConverter._ELEM_LINE_SLICE], 0.0)

    def test_bus_specific_slots_zero_for_non_bus(self):
        """Bus-specific slots are zero for non-bus nodes."""
        non_bus = np.ones(self.converter.num_nodes, dtype=bool)
        non_bus[self.converter._bus_offset:] = False
        np.testing.assert_array_equal(self.X[non_bus, ElementGraphObservationConverter._ELEM_BUS_SLICE], 0.0)

    def test_gen_rho_slot_is_zero(self):
        """Generators don't carry rho (line-specific slot 22)."""
        gen_slice = slice(self.converter._gen_offset, self.converter._gen_offset + env.n_gen)
        np.testing.assert_array_equal(self.X[gen_slice, 22], 0.0)

    def test_line_rho_values(self):
        """Line node slot 22 matches obs.rho."""
        line_slice = slice(self.converter._line_offset, self.converter._line_offset + env.n_line)
        np.testing.assert_array_almost_equal(self.X[line_slice, 22], self.obs.rho, decimal=5)

    def test_bus_one_hot_valid(self):
        """Each bus node has exactly one active b_type bit."""
        bus_nodes = self.X[self.converter._bus_offset:]
        one_hot = bus_nodes[:, 18:21]  # b_ground, b_bus1, b_bus2
        np.testing.assert_array_equal(one_hot.sum(axis=1), 1.0)

    def test_gen_type_one_hot_valid(self):
        """Each generator has at most one active g_type bit."""
        gen_slice = slice(self.converter._gen_offset, self.converter._gen_offset + env.n_gen)
        g_type = self.X[gen_slice, 13:18]  # g_type × 5
        self.assertTrue((g_type.sum(axis=1) <= 1).all())

    def test_base_features_nonzero_for_gen_and_load(self):
        """Base features (slots 0-4) are non-zero for generators and loads."""
        gen_slice = slice(self.converter._gen_offset, self.converter._gen_offset + env.n_gen)
        load_slice = slice(self.converter._load_offset, self.converter._load_offset + env.n_load)
        # |p| (slot 0) should be non-zero for active generators
        self.assertTrue((self.X[gen_slice, 0] >= 0).all())
        self.assertTrue((self.X[load_slice, 0] >= 0).all())

    def test_forecast_slots_zero_for_non_gen_load(self):
        """Forecast slots (26-27) are zero for lines, storage, and bus nodes."""
        non_gen_load = np.ones(self.converter.num_nodes, dtype=bool)
        non_gen_load[self.converter._gen_offset:self.converter._gen_offset + env.n_gen] = False
        non_gen_load[self.converter._load_offset:self.converter._load_offset + env.n_load] = False
        np.testing.assert_array_equal(self.X[non_gen_load, ElementGraphObservationConverter._ELEM_FORECAST_SLICE], 0.0)

    def test_forecast_slots_present_for_gen_and_load(self):
        """Forecast slots (26-27) are populated for generator and load nodes."""
        gen_slice = slice(self.converter._gen_offset, self.converter._gen_offset + env.n_gen)
        load_slice = slice(self.converter._load_offset, self.converter._load_offset + env.n_load)
        # Forecasts are finite floats (not NaN/inf) — content depends on the scenario
        self.assertTrue(np.isfinite(self.X[gen_slice, ElementGraphObservationConverter._ELEM_FORECAST_SLICE]).all())
        self.assertTrue(np.isfinite(self.X[load_slice, ElementGraphObservationConverter._ELEM_FORECAST_SLICE]).all())


class TestElementGraphObservationConverterEdges(unittest.TestCase):
    """Tests for static edge structure and dynamic edge attributes."""

    def setUp(self):
        self.converter = ElementGraphObservationConverter(env.observation_space)
        self.obs = env.reset()

    def test_edge_index_shape(self):
        """Static edge_index has 2 rows."""
        self.assertEqual(self.converter._edge_index.shape[0], 2)

    def test_edge_index_dtype(self):
        """Edge index is int64."""
        self.assertEqual(self.converter._edge_index.dtype, np.int64)

    def test_edge_index_within_range(self):
        """All edge indices are valid node indices."""
        ei = self.converter._edge_index
        self.assertTrue((ei >= 0).all())
        self.assertTrue((ei < self.converter.num_nodes).all())

    def test_edges_bidirectional(self):
        """For every (i, j) edge there is a corresponding (j, i) edge."""
        ei = self.converter._edge_index
        edge_set = set(zip(ei[0].tolist(), ei[1].tolist()))
        for s, d in list(edge_set):
            self.assertIn((d, s), edge_set, f"Missing reverse edge ({d}, {s})")

    def test_no_self_loops(self):
        """No edge connects a node to itself."""
        ei = self.converter._edge_index
        self.assertTrue((ei[0] != ei[1]).all())

    def test_gen_connected_to_three_bus_slots(self):
        """Each generator connects to exactly 3 bus nodes (ground, bus1, bus2)."""
        ei = self.converter._edge_index
        for i in range(env.n_gen):
            gen_node = self.converter._gen_offset + i
            neighbours = set(ei[1, ei[0] == gen_node].tolist())
            bus_neighbours = {n for n in neighbours if n >= self.converter._bus_offset}
            self.assertEqual(len(bus_neighbours), 3, f"Gen {i} has {len(bus_neighbours)} bus neighbours")

    def test_load_connected_to_two_bus_slots(self):
        """Each load connects to exactly 2 bus nodes (bus1, bus2 — no ground)."""
        ei = self.converter._edge_index
        for i in range(env.n_load):
            load_node = self.converter._load_offset + i
            neighbours = set(ei[1, ei[0] == load_node].tolist())
            bus_neighbours = {n for n in neighbours if n >= self.converter._bus_offset}
            self.assertEqual(len(bus_neighbours), 2, f"Load {i} has {len(bus_neighbours)} bus neighbours")

    def test_line_connected_to_six_bus_slots(self):
        """Each line connects to 6 bus nodes (3 at origin sub + 3 at extremity sub)."""
        ei = self.converter._edge_index
        for i in range(env.n_line):
            line_node = self.converter._line_offset + i
            neighbours = set(ei[1, ei[0] == line_node].tolist())
            bus_neighbours = {n for n in neighbours if n >= self.converter._bus_offset}
            self.assertEqual(len(bus_neighbours), 6, f"Line {i} has {len(bus_neighbours)} bus neighbours")

    def test_edge_attr_binary(self):
        """EDGE_ATTR values are 0 or 1."""
        ea = self.converter._get_edge_attr(self.obs)
        self.assertTrue(np.all((ea == 0.0) | (ea == 1.0)))

    def test_edge_attr_shape(self):
        """EDGE_ATTR has shape (num_edges, 1)."""
        ea = self.converter._get_edge_attr(self.obs)
        self.assertEqual(ea.shape, (self.converter.num_edges, 1))

    def test_edge_attr_symmetric(self):
        """Forward and reverse edge attributes are equal."""
        ea = self.converter._get_edge_attr(self.obs).flatten()
        # Edges are interleaved: (fwd, bwd, fwd, bwd, ...)
        np.testing.assert_array_equal(ea[0::2], ea[1::2])

    def test_each_element_active_on_correct_number_of_buses(self):
        """
        In the default connected state each element is active on the expected
        number of bus slots:
          - gens, loads, storage: 1 (single substation)
          - lines: 2 (one bus slot at each endpoint substation)
        """
        ea = self.converter._get_edge_attr(self.obs).flatten()
        ei = self.converter._edge_index
        conv = self.converter

        for i in range(conv._bus_offset):
            out_mask = ei[0] == i
            bus_out_mask = out_mask & (ei[1] >= conv._bus_offset)
            active_count = int(ea[bus_out_mask].sum())

            is_line = conv._line_offset <= i < conv._storage_offset
            expected = 2 if is_line else 1
            self.assertEqual(
                active_count, expected,
                f"Node {i} ({'line' if is_line else 'element'}) "
                f"active on {active_count} buses, expected {expected}",
            )


class TestElementGraphObservationConverterIntegration(unittest.TestCase):
    """End-to-end tests for ElementGraphObservationConverter.to_gym."""

    def setUp(self):
        self.converter = ElementGraphObservationConverter(env.observation_space)
        self.obs = env.reset()

    def test_to_gym_returns_all_keys(self):
        """to_gym returns all expected keys."""
        result = self.converter.to_gym(self.obs)
        for key in [NODES, EDGE_INDEX, EDGE_MASK, EDGES, NODE_MASK, GLOBAL]:
            self.assertIn(key, result)

    def test_to_gym_node_mask_all_true(self):
        """NODE_MASK is all-True (static graph — every node always exists)."""
        result = self.converter.to_gym(self.obs)
        self.assertTrue(result[NODE_MASK].all())

    def test_to_gym_edge_mask_all_true(self):
        """EDGE_MASK is all-True (static edges — no padding needed)."""
        result = self.converter.to_gym(self.obs)
        self.assertTrue(result[EDGE_MASK].all())

    def test_to_gym_edge_index_static(self):
        """EDGE_INDEX is identical across two calls (static graph)."""
        r1 = self.converter.to_gym(self.obs)
        r2 = self.converter.to_gym(self.obs)
        np.testing.assert_array_equal(r1[EDGE_INDEX], r2[EDGE_INDEX])

    def test_to_gym_observation_space_compliant(self):
        """to_gym output shapes match the declared observation space."""
        result = self.converter.to_gym(self.obs)
        space = self.converter.observation_space
        for key in [NODES, EDGE_INDEX, EDGE_MASK, EDGES, NODE_MASK, GLOBAL]:
            self.assertEqual(result[key].shape, space[key].shape, f"Shape mismatch for {key}")

    def test_normalize_passes_edge_attr_through(self):
        """normalize does not alter EDGE_ATTR."""
        result = self.converter.to_gym(self.obs)
        ea_before = result[EDGES].copy()
        renormalized = self.converter.normalize(result)
        np.testing.assert_array_equal(renormalized[EDGES], ea_before)

    def test_normalizer_updated_after_to_gym(self):
        """Running normalizer count increases after to_gym."""
        count_before = self.converter._normalizer.count
        self.converter.to_gym(self.obs)
        self.assertGreater(self.converter._normalizer.count, count_before)


if __name__ == "__main__":
    unittest.main()
