"""
Unit tests for get_node_styles in src/visualization/utils.py.

Covers:
  - SubstationGraphObservationConverter: length, position offset, colors, shapes
"""
from __future__ import annotations

import unittest

import grid2op
import numpy as np

from grid2op_env.observation_converter import (
    ElementGraphObservationConverter,
    GraphObservationConverter,
    HeterogeneousGraphObservationConverter,
    SubstationGraphObservationConverter,
)
from visualization.utils import get_node_styles

ENV_NAME = "l2rpn_wcci_2020"
env = grid2op.make(ENV_NAME)


class TestGetNodeStylesSubstationGraph(unittest.TestCase):
    """Tests for get_node_styles with SubstationGraphObservationConverter."""

    @classmethod
    def setUpClass(cls):
        cls.styles = get_node_styles(env, SubstationGraphObservationConverter)

    def test_length_is_twice_n_sub(self):
        """Returns exactly 2 * n_sub NodeStyle entries."""
        self.assertEqual(len(self.styles), 2 * env.n_sub)

    def test_bus1_and_bus2_positions_differ_per_sub(self):
        """Bus 1 and bus 2 nodes at the same substation have different positions."""
        for sub_id in range(env.n_sub):
            bus1_pos = self.styles[2 * sub_id].position
            bus2_pos = self.styles[2 * sub_id + 1].position
            self.assertFalse(
                np.allclose(bus1_pos, bus2_pos),
                f"Sub {sub_id}: bus 1 and bus 2 positions are identical",
            )

    def test_bus1_and_bus2_symmetric_around_substation(self):
        """Bus 1 and bus 2 are symmetric offsets from the substation center."""
        for sub_id in range(env.n_sub):
            bus1_pos = self.styles[2 * sub_id].position
            bus2_pos = self.styles[2 * sub_id + 1].position
            midpoint = (bus1_pos + bus2_pos) / 2
            # Midpoint should be the substation layout position
            from grid2op.PlotGrid import PlotMatplot
            layout = PlotMatplot(env.observation_space)._grid_layout
            expected = np.array(layout[f"sub_{sub_id}"], dtype=float)
            np.testing.assert_allclose(midpoint, expected, atol=1e-6)

    def test_bus1_color(self):
        """All bus-1 nodes (even slots) have color 'steelblue'."""
        for sub_id in range(env.n_sub):
            self.assertEqual(self.styles[2 * sub_id].color, "steelblue")

    def test_bus2_color(self):
        """All bus-2 nodes (odd slots) have color 'tomato'."""
        for sub_id in range(env.n_sub):
            self.assertEqual(self.styles[2 * sub_id + 1].color, "tomato")

    def test_shapes_are_circle(self):
        """All nodes use the circle marker 'o'."""
        for i, style in enumerate(self.styles):
            self.assertEqual(style.shape, "o", f"Node {i} has non-circle shape")

    def test_labels(self):
        """Bus 1 slots are labelled 'Busbar 1', bus 2 slots 'Busbar 2'."""
        for sub_id in range(env.n_sub):
            self.assertEqual(self.styles[2 * sub_id].label, "Busbar 1")
            self.assertEqual(self.styles[2 * sub_id + 1].label, "Busbar 2")

    def test_sizes_positive(self):
        """All node sizes are positive."""
        for style in self.styles:
            self.assertGreater(style.size, 0)

    def test_positions_are_2d(self):
        """Every position is a length-2 array."""
        for i, style in enumerate(self.styles):
            self.assertEqual(len(style.position), 2, f"Node {i} position is not 2D")


class TestGetNodeStylesHeterogeneousGraph(unittest.TestCase):
    """
    get_node_styles for HeterogeneousGraphObservationConverter must produce the
    same result as for GraphObservationConverter — the node structure is identical.
    """

    @classmethod
    def setUpClass(cls):
        cls.hetero_styles = get_node_styles(env, HeterogeneousGraphObservationConverter)
        cls.graph_styles = get_node_styles(env, GraphObservationConverter)
        cls.expected_n = 2 * env.n_line + env.n_gen + env.n_load + env.n_storage

    def test_length_matches_element_graph(self):
        """Returns the same number of styles as GraphObservationConverter."""
        self.assertEqual(len(self.hetero_styles), len(self.graph_styles))

    def test_length_is_element_count(self):
        """Length equals 2*n_line + n_gen + n_load + n_storage."""
        self.assertEqual(len(self.hetero_styles), self.expected_n)

    def test_positions_match_graph_converter(self):
        """Node positions are identical to those of GraphObservationConverter."""
        for i, (h, g) in enumerate(zip(self.hetero_styles, self.graph_styles)):
            np.testing.assert_allclose(
                h.position, g.position, atol=1e-6,
                err_msg=f"Node {i} position mismatch",
            )

    def test_colors_match_graph_converter(self):
        """Node colors are identical to those of GraphObservationConverter."""
        for i, (h, g) in enumerate(zip(self.hetero_styles, self.graph_styles)):
            self.assertEqual(h.color, g.color, f"Node {i} color mismatch")

    def test_shapes_match_graph_converter(self):
        """Node shapes are identical to those of GraphObservationConverter."""
        for i, (h, g) in enumerate(zip(self.hetero_styles, self.graph_styles)):
            self.assertEqual(h.shape, g.shape, f"Node {i} shape mismatch")

    def test_labels_match_graph_converter(self):
        """Node labels are identical to those of GraphObservationConverter."""
        for i, (h, g) in enumerate(zip(self.hetero_styles, self.graph_styles)):
            self.assertEqual(h.label, g.label, f"Node {i} label mismatch")


class TestGetNodeStylesElementGraph(unittest.TestCase):
    """Tests for get_node_styles with ElementGraphObservationConverter."""

    @classmethod
    def setUpClass(cls):
        cls.styles = get_node_styles(env, ElementGraphObservationConverter)
        cls.expected_n = (
            env.n_gen + env.n_load + env.n_line + env.n_storage + 3 * env.n_sub
        )
        # Slice boundaries matching the converter's node ordering
        cls.gen_end = env.n_gen
        cls.load_end = env.n_gen + env.n_load
        cls.line_end = env.n_gen + env.n_load + env.n_line
        cls.storage_end = env.n_gen + env.n_load + env.n_line + env.n_storage
        cls.bus_start = cls.storage_end

    def test_length(self):
        """Returns one style per node: n_gen + n_load + n_line + n_storage + 3*n_sub."""
        self.assertEqual(len(self.styles), self.expected_n)

    def test_generator_colors(self):
        """Generator nodes are green."""
        for i in range(self.gen_end):
            self.assertEqual(self.styles[i].color, "green", f"Gen node {i} wrong color")

    def test_generator_shapes(self):
        """Generator nodes use pentagon marker 'p'."""
        for i in range(self.gen_end):
            self.assertEqual(self.styles[i].shape, "p", f"Gen node {i} wrong shape")

    def test_generator_labels(self):
        """Generator nodes are labelled 'Generator'."""
        for i in range(self.gen_end):
            self.assertEqual(self.styles[i].label, "Generator")

    def test_load_colors(self):
        """Load nodes are orange."""
        for i in range(self.gen_end, self.load_end):
            self.assertEqual(self.styles[i].color, "orange", f"Load node {i} wrong color")

    def test_load_shapes(self):
        """Load nodes use triangle marker '^'."""
        for i in range(self.gen_end, self.load_end):
            self.assertEqual(self.styles[i].shape, "^", f"Load node {i} wrong shape")

    def test_line_colors(self):
        """Line nodes are gray."""
        for i in range(self.load_end, self.line_end):
            self.assertEqual(self.styles[i].color, "gray", f"Line node {i} wrong color")

    def test_line_labels(self):
        """Line nodes are labelled 'Powerline'."""
        for i in range(self.load_end, self.line_end):
            self.assertEqual(self.styles[i].label, "Powerline")

    def test_line_positions_between_substations(self):
        """Each line node is positioned at the midpoint of its or/ex substations."""
        from grid2op.PlotGrid import PlotMatplot
        layout = PlotMatplot(env.observation_space)._grid_layout
        for lid in range(env.n_line):
            or_pos = np.array(layout[f"sub_{int(env.line_or_to_subid[lid])}"], dtype=float)
            ex_pos = np.array(layout[f"sub_{int(env.line_ex_to_subid[lid])}"], dtype=float)
            expected_mid = (or_pos + ex_pos) / 2.0
            actual_pos = self.styles[self.load_end + lid].position
            np.testing.assert_allclose(actual_pos, expected_mid, atol=1e-6,
                                       err_msg=f"Line {lid} position mismatch")

    def test_bus_block_structure(self):
        """Each substation contributes 3 consecutive bus nodes: ground, bus1, bus2."""
        for sub_id in range(env.n_sub):
            base = self.bus_start + 3 * sub_id
            ground, bus1, bus2 = self.styles[base], self.styles[base + 1], self.styles[base + 2]
            self.assertEqual(ground.label, "Ground",  f"Sub {sub_id} slot 0 not Ground")
            self.assertEqual(bus1.label,  "Busbar 1", f"Sub {sub_id} slot 1 not Busbar 1")
            self.assertEqual(bus2.label,  "Busbar 2", f"Sub {sub_id} slot 2 not Busbar 2")
            self.assertEqual(ground.color, "black",    f"Sub {sub_id} Ground wrong color")
            self.assertEqual(bus1.color,  "steelblue", f"Sub {sub_id} Bus1 wrong color")
            self.assertEqual(bus2.color,  "tomato",    f"Sub {sub_id} Bus2 wrong color")

    def test_bus_nodes_are_separated(self):
        """Ground, bus1, and bus2 are all at distinct, well-separated positions."""
        min_dist = 5.0
        for sub_id in range(env.n_sub):
            base = self.bus_start + 3 * sub_id
            ground_pos = self.styles[base].position
            bus1_pos   = self.styles[base + 1].position
            bus2_pos   = self.styles[base + 2].position
            for name, p1, p2 in [
                ("ground-bus1", ground_pos, bus1_pos),
                ("ground-bus2", ground_pos, bus2_pos),
                ("bus1-bus2",   bus1_pos,   bus2_pos),
            ]:
                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                self.assertGreater(dist, min_dist,
                                   f"Sub {sub_id}: {name} nodes too close (dist={dist:.2f})")

    def test_positions_are_2d(self):
        """Every position is a length-2 array."""
        for i, style in enumerate(self.styles):
            self.assertEqual(len(style.position), 2, f"Node {i} position is not 2D")

    def test_sizes_positive(self):
        """All node sizes are positive."""
        for style in self.styles:
            self.assertGreater(style.size, 0)


class TestGetNodeStylesNotImplemented(unittest.TestCase):
    """get_node_styles raises NotImplementedError for unsupported converters."""

    def test_raises_for_unknown_converter(self):
        """Passing an unsupported class raises NotImplementedError."""
        from grid2op_env.observation_converter import ObservationConverter

        class _Dummy(ObservationConverter):
            pass

        with self.assertRaises(NotImplementedError):
            get_node_styles(env, _Dummy)


if __name__ == "__main__":
    unittest.main()
