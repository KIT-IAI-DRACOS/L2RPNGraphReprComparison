"""
Unit tests for the shared/per-axis legend helpers in src/visualization/utils.py.

Covers:
  - build_node_legend_handles: dedup by label, phantom nodes skipped
  - dedupe_legend_handles: merge handles collected across several subplots
  - PlottingArgs.show_node_legend / show_edge_legend: end-to-end through visualize_graph
"""
from __future__ import annotations

import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.lines import Line2D

from visualization.utils import (
    NodeStyle,
    PlottingArgs,
    build_node_legend_handles,
    dedupe_legend_handles,
    visualize_graph,
)


def _node_style(label: str, alpha: float = 1.0, position=(0.0, 0.0)) -> NodeStyle:
    return NodeStyle(position=np.array(position), color="steelblue", shape="o", size=100, label=label, alpha=alpha)


class TestBuildNodeLegendHandles(unittest.TestCase):
    """Tests for build_node_legend_handles."""

    def test_one_handle_per_unique_label(self):
        """Duplicate labels across node styles collapse to a single handle."""
        styles = [_node_style("Load"), _node_style("Load"), _node_style("Generator")]
        handles = build_node_legend_handles(styles)
        self.assertEqual([h.get_label() for h in handles], ["Load", "Generator"])

    def test_phantom_nodes_are_skipped(self):
        """Node styles with alpha == 0 (phantom, expand-bounds-only) get no legend entry."""
        styles = [_node_style("Load"), _node_style("Storage", alpha=0.0)]
        handles = build_node_legend_handles(styles)
        self.assertEqual([h.get_label() for h in handles], ["Load"])

    def test_empty_input_returns_empty_list(self):
        """No node styles produces no handles."""
        self.assertEqual(build_node_legend_handles([]), [])

    def test_first_seen_order_is_preserved(self):
        """Handles are returned in the order labels first appear."""
        styles = [_node_style("Busbar 2"), _node_style("Busbar 1"), _node_style("Busbar 2")]
        handles = build_node_legend_handles(styles)
        self.assertEqual([h.get_label() for h in handles], ["Busbar 2", "Busbar 1"])


class TestDedupeLegendHandles(unittest.TestCase):
    """Tests for dedupe_legend_handles."""

    def test_duplicate_labels_collapse_to_first_occurrence(self):
        """When two handles share a label, only the first is kept."""
        first = Line2D([0], [0], color="green", label="Generator")
        second = Line2D([0], [0], color="red", label="Generator")
        merged = dedupe_legend_handles([first, second])
        self.assertEqual(len(merged), 1)
        self.assertIs(merged[0], first)

    def test_distinct_labels_are_all_preserved(self):
        """Handles with distinct labels are all kept, in original order."""
        handles = [Line2D([0], [0], label="Load"), Line2D([0], [0], label="Generator")]
        merged = dedupe_legend_handles(handles)
        self.assertEqual([h.get_label() for h in merged], ["Load", "Generator"])

    def test_empty_input_returns_empty_list(self):
        """No handles produces no merged handles."""
        self.assertEqual(dedupe_legend_handles([]), [])


class TestPlottingArgsLegendFlags(unittest.TestCase):
    """End-to-end: show_node_legend / show_edge_legend control what visualize_graph draws."""

    def _args(self, **overrides) -> PlottingArgs:
        node_styles = [_node_style("Load", position=(0.0, 0.0)), _node_style("Generator", position=(1.0, 0.0))]
        defaults = dict(
            num_nodes=2,
            node_styles=node_styles,
            powerline_edge_index=np.array([[0], [1]]),
            show_legend=True,
        )
        defaults.update(overrides)
        return PlottingArgs(**defaults)

    def test_default_legend_has_node_and_edge_entries(self):
        """With both flags at their default (True), the legend contains node and edge labels."""
        fig = visualize_graph(self._args())
        labels = {h.get_label() for h in fig.axes[0].get_legend().legend_handles}
        self.assertIn("Load", labels)
        self.assertIn("Generator", labels)
        self.assertIn("Physical Connection", labels)

    def test_show_node_legend_false_hides_node_entries(self):
        """With show_node_legend=False, only edge labels remain in the legend."""
        fig = visualize_graph(self._args(show_node_legend=False))
        labels = {h.get_label() for h in fig.axes[0].get_legend().legend_handles}
        self.assertNotIn("Load", labels)
        self.assertNotIn("Generator", labels)
        self.assertIn("Physical Connection", labels)

    def test_show_edge_legend_false_hides_edge_entries(self):
        """With show_edge_legend=False, only node labels remain in the legend."""
        fig = visualize_graph(self._args(show_edge_legend=False))
        labels = {h.get_label() for h in fig.axes[0].get_legend().legend_handles}
        self.assertIn("Load", labels)
        self.assertNotIn("Physical Connection", labels)

    def test_both_legend_parts_disabled_draws_no_legend(self):
        """With neither part enabled, visualize_graph draws no legend at all."""
        fig = visualize_graph(self._args(show_node_legend=False, show_edge_legend=False))
        self.assertIsNone(fig.axes[0].get_legend())


if __name__ == "__main__":
    unittest.main()
