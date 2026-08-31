import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from grid2op.Environment import Environment
from grid2op.PlotGrid import PlotMatplot
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from grid2op_env.observation_converter import (
    ElementGraphObservationConverter,
    GraphObservationConverter,
    HeterogeneousGraphObservationConverter,
    ElementLODFGraphObservationConverter,
    LODFGraphObservationConverter,
    PTDFGraphObservationConverter,
    SubstationGraphObservationConverter,
    SubstationPTDFGraphObservationConverter,
    SubstationZbusGraphObservationConverter,
    ZbusGraphObservationConverter,
    ObservationConverter,
    EDGES,
    EDGE_MASK,
    EDGE_TYPE, EDGE_INDEX,
)
from core.graph import fully_connected_edge_index

logger = logging.getLogger(__name__)


@dataclass
class NodeStyle:
    position: npt.NDArray
    color: str
    shape: str
    size: int
    label: str
    label_offset: tuple[float, float] = (0.0, 0.0)
    alpha: float = 1.0  # set to 0.0 for phantom nodes (expand axes bounds without being visible)


@dataclass
class EdgeStyle:
    """Visual style for a single edge in a graph visualization."""
    color: str
    width: float = 1.0
    label: str = "Edge"
    alpha: float = 1.0
    linestyle: str = "-"


_HETERO_EDGE_TYPE_STYLES: List[EdgeStyle] = [
    EdgeStyle(color="darkorange",  width=2.0, label="Powerline", alpha=1.0, linestyle="--"),
    EdgeStyle(color="forestgreen", width=1.5, label="Same-bus",  alpha=0.9, linestyle="-"),
    EdgeStyle(color="crimson",     width=1.5, label="Disconnected",  alpha=0.9, linestyle=":"),
]

_ELEM_EDGE_STYLES: List[EdgeStyle] = [
    EdgeStyle(color="lightgray",   width=0.6, label="Inactive connection", alpha=0.5, linestyle="-"),
    EdgeStyle(color="forestgreen", width=1.8, label="Active connection",   alpha=1.0, linestyle="-"),
]


@dataclass
class PlottingArgs:
    num_nodes: int
    node_styles: Optional[List[NodeStyle]] = None
    powerline_edge_index: Optional[npt.NDArray] = None
    latent_edge_probs: Optional[npt.NDArray] = None
    latent_edge_weight: float = 5.0
    do_weight_sweep: bool = False
    skip_last_edge_type: bool = True
    visualize_edge_prob_threshold: float = 0.5
    node_labels: Optional[dict[int, str]] = None  # Dict mapping node_id to label text
    enumerate_nodes: bool = False  # Draw node index as a label inside each node
    enumerate_nodes_exclude_labels: Optional[set[str]] = None  # NodeStyle.label values to skip when enumerating
    node_sizes_override: Optional[dict[int, float]] = None  # Dict mapping node_id to size multiplier
    powerline_edge_colors: Optional[List[str]] = None  # Custom colors for powerline edges
    powerline_edge_widths: Optional[List[float]] = None  # Custom widths for powerline edges
    edge_styles: Optional[List[EdgeStyle]] = None  # Per-edge style (parallel to powerline_edge_index columns)
    show_legend: bool = True  # master switch: no legend at all when False
    show_node_legend: bool = True  # include node-type entries (Generator/Load/...) when show_legend is True
    show_edge_legend: bool = True  # include edge-type entries when show_legend is True
    edge_labels: Optional[dict[tuple[int, int], str]] = None  # (min_u, max_u) -> label drawn at edge midpoint
    edge_label_font_size: int = 15
    substation_node_groups: Optional[dict[int, list[int]]] = None  # sub_id -> [node_idx, ...] for enclosing circles
    substation_circle_padding: float = 10.0  # extra radius beyond the outermost node (data units)
    curved_edges: Optional[dict[tuple[int, int], float]] = None  # (min_u, max_u) -> arc rad (positive = left-hand curve)


@dataclass
class GridPlottingArgs:
    """Arguments for visualize_grid — a substation-level power grid view."""
    env: Environment
    # Per-line style overrides (length must match env.n_line or be None for defaults)
    line_colors: Optional[List[str]] = None    # e.g. ["red", "gray", ...]
    line_widths: Optional[List[float]] = None  # e.g. [2.0, 1.0, ...]
    line_styles: Optional[List[str]] = None    # e.g. ["-", "--", ":", "-."]
    node_size: int = 600           # matplotlib scatter size for substation circles
    node_color: str = "white"      # fill color of substation circles (used when node_colors is None)
    node_colors: Optional[List] = None  # per-substation fill colors (overrides node_color when set)
    font_size: int = 15            # font size for substation index labels
    font_color: str = "black"      # label colour inside circles
    show_legend: bool = False      # whether to draw a legend
    legend_font_size: int = 15     # font size for legend text


@dataclass
class AgentMetrics:
    label: str
    returns: List[float]
    survival_duration: List[int]


def visualize_agent_survival(datasets: List[AgentMetrics], save_to: Optional[Path] = None, show: bool = True, title=None):
    records = []
    for data in datasets:
        records.extend([{"Agent": data.label, "Survival Duration": d}
                        for d in data.survival_duration])
    df = pd.DataFrame(records)

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(2 * len(datasets), 4))
    # --- Boxplot ---
    sns.boxplot(
        data=df,
        x="Agent",
        y="Survival Duration",
        hue="Agent",
        palette="muted",#["#4878D0", "#63BE5D", "#82C6E2", "#956CB4"],
        legend=False
    )

    plt.title(title if title is not None else "Survival Duration Boxplot per Agent", fontsize=14, fontweight='bold')
    plt.xlabel("Agent")
    plt.ylabel("Time Steps")

    plt.tight_layout()

    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_to)

    if show:
        plt.show()

    plt.close()


def compare_experiment_runs(experiment_path: Path):
    """
    This method assumes a folder structure like this:
    - experiment_name/
        - variant_1/
            - agent/
            - rl_algorithm/
        - variant_2/
            - agent/
            - rl_algorithm/

    Generates and stores root level comparison plots for the different runs.

    @param experiment_path: the path to the experiment folder
    @return: a list of figure comparing the variants
    """
    if not experiment_path.exists() or not experiment_path.is_dir():
        raise FileNotFoundError(f"Experiment folder {experiment_path} does not exist")

    if len(os.listdir(experiment_path)) == 0:
        raise FileNotFoundError(f"Experiment folder {experiment_path} does not contain any files")

    metrics = {}

    for variant in os.listdir(experiment_path):
        for sub_variant in ["agent", "rl_algorithm"]:
            variant_path = Path(experiment_path, variant, sub_variant)
            if not variant_path.exists() or not variant_path.is_dir():
                logger.warning(f"Folder {variant_path} does not exist")
                continue

            if len(os.listdir(variant_path)) == 0:
                logger.warning(f"Folder {variant_path} does not contain any files")
                continue

            for dataset in os.listdir(sub_variant):
                metrics[dataset][sub_variant][variant] = get_evaluation_metrics(
                    Path(experiment_path, variant, sub_variant, dataset),
                    variant
                )

    for dataset in metrics.keys():
        for sub_variant in metrics[dataset].keys():
            visualize_agent_survival(
                datasets=metrics[dataset][sub_variant],
                save_to=Path(experiment_path, f"compare_{sub_variant}_on_{dataset}.png"),
                show=False
            )


def visualize_performance_vs_prior(datasets: List[AgentMetrics]):
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    # Define figure with a GridSpec: widths [1, 2, 1]
    plt.figure(figsize=(20, 5))

    # --- Strip Plot ---
    records = []
    for data in datasets:
        records.extend([{"Agent": data.label, "Survival Duration": d} for d in data.survival_duration])
    df = pd.DataFrame(records)

    sns.boxplot(
        data=df,
        y="Survival Duration",
        x="Agent",
    )

    plt.title("Survival Duration (Strip Plot)")
    plt.ylabel("Time Steps")
    plt.xlabel("Agent")
    plt.show()
    plt.close()


def visualize_agent_survival_return_relationship(datasets: List[AgentMetrics]):
    plt.figure(figsize=(10, 5))
    for data in datasets:
        sns.scatterplot(
            x=data.survival_duration,
            y=data.returns,
            label=data.label
        )
    plt.title("Returns vs Survival Duration")
    plt.xlabel("Survival Duration (Time Steps)")
    plt.ylabel("Return")
    plt.legend()
    plt.show()
    plt.close()


def get_evaluation_metrics(path: Path, agent_name: str) -> AgentMetrics:
    survival_duration = []
    returns = []
    for folder in path.iterdir():
        if folder.is_dir():
            meta_file = Path.joinpath(folder, "episode_meta.json")
            if not meta_file.exists():
                continue
            with meta_file.open() as f:
                episode_metadata = json.load(f)
                survival_duration.append(episode_metadata["nb_timestep_played"])
                returns.append(episode_metadata["cumulative_reward"])

    return AgentMetrics(agent_name, returns, survival_duration)


def get_training_progress(path: Path, agent_name: str) -> AgentMetrics:
    returns = []
    with path.open() as f:
        df = pd.read_csv(f)
        survival_duration = df["Value"].tolist()

    return AgentMetrics(agent_name, returns, survival_duration)


def display_training_progress(metrics: List[AgentMetrics], show: bool = True) -> Figure:
    fig = plt.figure(figsize=(10, 5))
    shortest_training_duration = min([len(metrics.survival_duration) for metrics in metrics])
    xs = np.arange(shortest_training_duration)
    cmap = plt.get_cmap("tab10")
    for i, metric in enumerate(metrics):
        survival_duration = metric.survival_duration[:shortest_training_duration]
        smooth_survival = smooth_curve_conv(survival_duration)
        plt.plot(xs, smooth_survival, label=metric.label, color=cmap(i))

    plt.legend()

    plt.title("Survival Duration vs Steps")
    plt.xlabel("Training Steps x 10^3")
    plt.ylabel("Survival Duration")

    if show:
        plt.show()

    return fig


def smooth_curve_conv(curve: List[float] | npt.NDArray, window: int = 5) -> npt.NDArray | List[float]:
    curve = np.asarray(curve, dtype=float)

    if window < 1:
        raise ValueError("window must be >= 1")

    kernel = np.ones(window) / window
    pad = window // 2

    # Repeat edge values outside the signal
    padded = np.pad(curve, pad_width=pad, mode="edge")

    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed

def smooth_curve(curve: List[float] | npt.NDArray, alpha=1.0) -> npt.NDArray | List[float]:
    smoothed = np.zeros_like(curve, dtype=float)
    smoothed[0] = curve[0]
    for i in range(1, len(curve)):
        smoothed[i] = alpha * curve[i] + (1 - alpha) * smoothed[i-1]
    return smoothed


def visualize_graph(args: PlottingArgs, ax=None) -> Figure:
    """
    Visualize the power grid graph.
    :param args: args for plotting
    :param ax: optional matplotlib axis to draw on. If None, creates new figure.
    :return a figure (or None if ax is provided)
    """
    assert args.latent_edge_probs is None or args.num_nodes * (args.num_nodes - 1) == args.latent_edge_probs.shape[0]

    scale = 0.66

    # Create new figure if no axis provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(18 * scale, 10 * scale), dpi=100)
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False

    G = nx.MultiDiGraph()
    G.add_nodes_from(range(args.num_nodes))

    # Equal aspect ensures node positions use the same scale in x and y,
    # matching the physical grid layout for all converter types.
    ax.set_aspect('equal', adjustable='datalim')

    # draw substation enclosing circles (lowest z-order — behind all edges and nodes)
    if args.substation_node_groups is not None and args.node_styles is not None:
        from matplotlib.patches import Circle

        # Compute per-group centroids and the global max radius for uniform sizing
        group_centroids: dict[int, npt.NDArray] = {}
        max_content_radius = 0.0
        for sub_id, node_indices in args.substation_node_groups.items():
            valid = [i for i in node_indices if i < len(args.node_styles)]
            if not valid:
                continue
            positions = np.array([args.node_styles[i].position for i in valid])
            cx, cy = positions.mean(axis=0)
            group_centroids[sub_id] = np.array([cx, cy])
            r = np.linalg.norm(positions - np.array([cx, cy]), axis=1).max()
            max_content_radius = max(max_content_radius, r)

        uniform_radius = 65 + args.substation_circle_padding

        for sub_id, center in group_centroids.items():
            cx, cy = center
            circle = Circle(
                (cx, cy), uniform_radius,
                facecolor='slategray',
                edgecolor='black',
                alpha=0.10,
                zorder=0,
            )
            ax.add_patch(circle)
            ax.text(
                cx, cy + uniform_radius - 15,
                str(sub_id),
                fontsize=15,
                ha='center', va='center',
                color='slategray',
                fontweight='bold',
                zorder=1,
            )

    # base edges - convert to undirected by filtering out duplicate directed edges
    if args.powerline_edge_index is not None:
        seen_edges = set()
        for i, (src, dst) in enumerate(args.powerline_edge_index.T):
            # Create unordered edge tuple (always smaller node first)
            edge_tuple = tuple(sorted([int(src), int(dst)]))
            if edge_tuple not in seen_edges:
                seen_edges.add(edge_tuple)
                if args.edge_styles is not None:
                    es = args.edge_styles[i]
                    edge_color, edge_width = es.color, es.width
                    edge_alpha, edge_linestyle, edge_label = es.alpha, es.linestyle, es.label
                else:
                    edge_color = args.powerline_edge_colors[i] if args.powerline_edge_colors is not None else "gray"
                    edge_width = args.powerline_edge_widths[i] if args.powerline_edge_widths is not None else 1
                    edge_alpha, edge_linestyle, edge_label = 1.0, "--", "Physical Connection"
                G.add_edge(int(src), int(dst), color=edge_color, weight=edge_width,
                           alpha=edge_alpha, style=edge_linestyle, label=edge_label,
                           type="Connection")

    # latent edges
    if args.latent_edge_probs is not None:
        cmap = plt.get_cmap("Pastel1")
        edge_index_full = fully_connected_edge_index(num_nodes=args.num_nodes)
        probs_array = args.latent_edge_probs  # [E, num_edge_types]
        max_type = probs_array.shape[1] - 1
        num_types = max_type if args.skip_last_edge_type else max_type + 1

        # Prefilter with numpy before entering Python loops: for each edge type,
        # find only the edges whose probability exceeds the threshold.  For large
        # grids this avoids iterating over O(N²) edges in pure Python.
        for t in range(num_types):
            above = np.where(probs_array[:, t] > args.visualize_edge_prob_threshold)[0]
            for e_idx in above:
                p = probs_array[e_idx, t]
                # 1 for p = 0.5, args.latent_edge_weight for p = 1
                w = (2 * args.latent_edge_weight - 2) * p - (args.latent_edge_weight - 2)
                if w >= 1:
                    src, dst = edge_index_full[:, e_idx]
                    G.add_edge(int(src), int(dst), color=cmap(1 + t), weight=w, type="Dependency")

    # get positions
    if args.node_styles is not None:
        pos = {i: ns.position for i, ns in enumerate(args.node_styles)}
    else:
        pos = nx.circular_layout(range(args.num_nodes))

    # separate edges
    conn_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["type"] == "Connection"]
    dep_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d["type"] == "Dependency"]

    # draw latent edges FIRST (bottom layer) with transparency
    if dep_edges:
        lc = nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v) for u, v, _ in dep_edges],
            edge_color=[d["color"] for _, _, d in dep_edges],
            width=[d["weight"] for _, _, d in dep_edges],
            arrows=False,
            alpha=0.5,  # Make latent edges semi-transparent
            ax=ax
        )
        lc.set_zorder(1)

    # draw base edges ON TOP, grouped by (linestyle, alpha) so each group gets one draw call
    if conn_edges:
        # Split out edges that should be drawn curved
        curved_set = set(args.curved_edges.keys()) if args.curved_edges else set()
        straight_edges = []
        curved_edge_data = []
        for u, v, d in conn_edges:
            key = (min(u, v), max(u, v))
            if key in curved_set:
                curved_edge_data.append((u, v, d, args.curved_edges[key]))
            else:
                straight_edges.append((u, v, d))

        # Straight edges — batched by style
        style_groups: dict = {}
        for u, v, d in straight_edges:
            key = (d.get("style", "--"), d.get("alpha", 1.0))
            style_groups.setdefault(key, []).append((u, v, d))

        for (linestyle, alpha), edges in style_groups.items():
            lc = nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v) for u, v, _ in edges],
                edge_color=[d["color"] for _, _, d in edges],
                width=[d["weight"] for _, _, d in edges],
                arrows=False,
                style=linestyle,
                alpha=alpha,
                ax=ax,
            )
            if lc is not None:
                lc.set_zorder(3)

        # Curved edges — drawn individually with arc3 connectionstyle
        for u, v, d, rad in curved_edge_data:
            ax.annotate(
                '', xy=pos[v], xytext=pos[u],
                arrowprops=dict(
                    arrowstyle='-',
                    connectionstyle=f'arc3,rad={rad}',
                    color=d['color'],
                    lw=d['weight'],
                    linestyle=d.get('style', '-'),
                    alpha=d.get('alpha', 1.0),
                ),
                zorder=3,
            )

    # draw edge labels at midpoints (above edges, below nodes)
    if args.edge_labels:
        for (u, v), text in args.edge_labels.items():
            if u in pos and v in pos:
                mx = (pos[u][0] + pos[v][0]) / 2
                my = (pos[u][1] + pos[v][1]) / 2
                ax.text(
                    mx, my, text,
                    fontsize=args.edge_label_font_size,
                    ha='center', va='center',
                    color='black', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=1.0),
                    zorder=12,
                )

    # draw nodes (with highest z-order to be on top of all edges)
    if args.node_styles is not None:
        # Group by (shape, alpha) so phantom nodes (alpha=0) get a separate draw
        # call — they still register positions for matplotlib autoscaling.
        node_style_groups: dict[tuple[str, float], list[int]] = {}
        for i, ns in enumerate(args.node_styles):
            node_style_groups.setdefault((ns.shape, ns.alpha), []).append(i)

        for (shape, alpha), idx in node_style_groups.items():
            node_sizes = []
            for i in idx:
                base_size = args.node_styles[i].size * scale
                if args.node_sizes_override is not None and i in args.node_sizes_override:
                    base_size *= args.node_sizes_override[i]
                node_sizes.append(base_size)

            node_collection = nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=idx,
                node_color=[args.node_styles[i].color for i in idx],
                node_shape=shape,
                node_size=node_sizes,
                alpha=alpha,
                ax=ax,
            )
            node_collection.set_zorder(10)  # Highest z-order to be on top

        # Draw node labels if provided or if enumerate_nodes is set
        node_labels = args.node_labels
        if node_labels is None and args.enumerate_nodes:
            exclude_labels = args.enumerate_nodes_exclude_labels or set()
            node_labels = {i: str(i) for i in range(args.num_nodes) if args.node_styles[i].label not in exclude_labels}

        if node_labels is not None:
            # Create label dict filtered to existing labels
            labels_to_draw = {node_id: label for node_id, label in node_labels.items()
                            if node_id < len(args.node_styles)}

            # Build per-node label positions, applying label_offset from NodeStyle
            label_pos = {
                node_id: args.node_styles[node_id].position + np.array(args.node_styles[node_id].label_offset)
                for node_id in labels_to_draw
            }

            # Draw labels inside nodes — no bbox so text sits directly on the node fill
            label_artists = nx.draw_networkx_labels(
                G,
                label_pos,
                labels=labels_to_draw,
                font_size=15,
                font_color='white',
                font_weight='bold',
                ax=ax,
            )
            for text in label_artists.values():
                text.set_zorder(15)

        # Create legend (pass ax if provided)
        if args.show_legend:
            _create_legend(args, G, ax)

        # Explicitly register all node positions (including alpha=0 phantom nodes)
        # so autoscaling uses the full bounding box, not just visible nodes.
        all_positions = np.array([ns.position for ns in args.node_styles])
        ax.update_datalim(all_positions)
        ax.autoscale_view()
    else:
        node_collection = nx.draw_networkx_nodes(G, pos, node_color="grey", ax=ax)
        node_collection.set_zorder(10)

    ax.axis("off")

    if return_fig:
        fig.tight_layout()
        return fig
    else:
        return None


def build_node_legend_handles(node_styles: List[NodeStyle]) -> List[Line2D]:
    """
    Build one legend handle per unique visible node label.

    Phantom node styles (``alpha == 0``, used only to expand axes bounds)
    are skipped since they carry no visible marker to explain.

    :param node_styles: node styles to draw legend entries for
    :return: one :class:`~matplotlib.lines.Line2D` handle per unique label,
             in first-seen order
    """
    unique_labels: dict[str, tuple[str, str, int]] = {}
    for ns in node_styles:
        if ns.alpha > 0 and ns.label not in unique_labels:
            unique_labels[ns.label] = (ns.color, ns.shape, ns.size)

    return [
        Line2D(
            [0], [0],
            marker=shape,
            color='w',
            markerfacecolor=color,
            markeredgecolor=color,  # required for edge-only markers like "x" and "+"
            markersize=20,
            linestyle='None',
            label=label,
        )
        for label, (color, shape, size) in unique_labels.items()
    ]


def dedupe_legend_handles(handles: List[Line2D]) -> List[Line2D]:
    """
    Remove handles with a duplicate label, keeping the first occurrence.

    Used to merge legend handles collected from several converters/subplots
    into a single shared legend.

    :param handles: legend handles, possibly containing duplicate labels
    :return: handles with duplicate labels removed, original order preserved
    """
    seen: dict[str, Line2D] = {}
    for handle in handles:
        seen.setdefault(handle.get_label(), handle)
    return list(seen.values())


def _build_edge_legend_handles(G: nx.Graph) -> List[Line2D]:
    """Build one legend handle per unique edge type/label present in ``G``."""
    # --- Dependency (latent) edges ---
    dependency_edge_colors = [d["color"] for (_, _, d) in G.edges(data=True) if d["type"] == "Dependency"]
    dependency_edge_colors_unique = set(dependency_edge_colors)
    edge_legend = [
        Line2D([0], [0],
               color=c,
               lw=2,
               label=f"Node pairs with high\nposterior mean/variance")
        for i, c in enumerate(dependency_edge_colors_unique)
    ]
    # Deduplicate connection edges by label for the legend
    seen_conn_labels: dict = {}
    for _, _, d in G.edges(data=True):
        if d["type"] == "Connection":
            lbl = d.get("label", "Physical Connection")
            if lbl not in seen_conn_labels:
                seen_conn_labels[lbl] = d
    for lbl, d in seen_conn_labels.items():
        edge_legend.insert(0, Line2D(
            [0], [0],
            color=d["color"],
            lw=2,
            linestyle=d.get("style", "--"),
            label=lbl,
        ))
    return edge_legend


def _create_legend(args: PlottingArgs, G: nx.Graph, ax=None) -> None:
    node_legend = build_node_legend_handles(args.node_styles) if args.show_node_legend else []
    edge_legend = _build_edge_legend_handles(G) if args.show_edge_legend else []
    handles = node_legend + edge_legend
    if not handles:
        return

    if node_legend:
        # Full (or node-only) legend: place it just outside the right edge of the
        # axes so it never overlaps content. tight_layout / constrained_layout
        # will automatically expand the margin to fit it.
        legend_kwargs = dict(loc="upper left", bbox_to_anchor=(0.83, 0.9), frameon=False, fontsize=20)
    else:
        # Edge-only legend (node types are merged into a shared legend elsewhere):
        # placed near the axes edge rather than reserving a dedicated side
        # margin; callers with multi-row grids control clearance via their own
        # subplots_adjust(hspace=...). loc="center" anchors the box's true
        # center (not its bottom edge) to bbox_to_anchor, so boxes with a
        # different number of entries (hence different heights) still line up
        # across subplots.
        legend_kwargs = dict(loc="center", bbox_to_anchor=(0.5, 0.0), frameon=True, framealpha=0.85, fontsize=20)

    if ax is not None:
        ax.legend(handles=handles, bbox_transform=ax.transAxes, **legend_kwargs)
    else:
        plt.legend(handles=handles, **legend_kwargs)


def visualize_grid(args: GridPlottingArgs, ax=None) -> Optional[Figure]:
    """
    Visualize the power grid at the substation level.

    Each substation is drawn as a labeled circle (one node per substation).
    Transmission lines are drawn as edges between the substations they connect.
    No intra-substation structure is shown — this is purely the physical grid topology.

    Per-line style can be configured via ``args.line_colors``, ``args.line_widths``
    and ``args.line_styles``.

    :param args: plotting configuration (see :class:`GridPlottingArgs`)
    :param ax: optional matplotlib :class:`~matplotlib.axes.Axes` to draw on.
               If *None* a new figure is created and returned.  If an axis is
               supplied the function draws into it and returns *None*.
    :return: the created :class:`~matplotlib.figure.Figure`, or *None* when
             *ax* was provided.
    """
    env = args.env
    n_sub = env.n_sub
    n_line = env.n_line

    # ------------------------------------------------------------------ #
    # Resolve substation positions from the grid2op plot helper            #
    # ------------------------------------------------------------------ #
    plot_helper = PlotMatplot(env.observation_space)
    layout = plot_helper._grid_layout  # dict: "sub_0" -> [x, y], etc.

    pos = {}
    for sub_id in range(n_sub):
        xy = layout[f"sub_{sub_id}"]
        pos[sub_id] = np.array(xy, dtype=float)

    # ------------------------------------------------------------------ #
    # Build graph                                                          #
    # ------------------------------------------------------------------ #
    G = nx.MultiGraph()
    G.add_nodes_from(range(n_sub))

    # Default line styles
    default_color = "gray"
    default_width = 1.5
    default_style = "-"

    for line_idx in range(n_line):
        src = int(env.line_or_to_subid[line_idx])
        dst = int(env.line_ex_to_subid[line_idx])
        color = args.line_colors[line_idx] if args.line_colors is not None else default_color
        width = args.line_widths[line_idx] if args.line_widths is not None else default_width
        style = args.line_styles[line_idx] if args.line_styles is not None else default_style
        G.add_edge(src, dst, color=color, weight=width, style=style, line_idx=line_idx)

    # ------------------------------------------------------------------ #
    # Figure / axis                                                        #
    # ------------------------------------------------------------------ #
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False

    # ------------------------------------------------------------------ #
    # Draw edges — group by (color, width, style) for efficiency          #
    # ------------------------------------------------------------------ #
    # Collect edges grouped by their visual style so we can batch draw

    all_edges = list(G.edges(data=True))

    # Group by style triple so each group gets one draw call
    style_groups: dict = {}
    for u, v, d in all_edges:
        key = (d["color"], d["weight"], d["style"])
        style_groups.setdefault(key, []).append((u, v))

    for (color, width, style), edgelist in style_groups.items():
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist,
            edge_color=color,
            width=width,
            style=style,
            arrows=False,
            ax=ax,
        )

    # ------------------------------------------------------------------ #
    # Draw nodes as filled circles                                         #
    # ------------------------------------------------------------------ #
    node_collection = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(range(n_sub)),
        node_color=args.node_colors if args.node_colors is not None else args.node_color,
        linewidths=1,
        edgecolors="black",
        node_shape="o",
        node_size=args.node_size,
        ax=ax,
    )
    node_collection.set_zorder(5)

    # ------------------------------------------------------------------ #
    # Draw substation index labels inside the circles                      #
    # ------------------------------------------------------------------ #
    label_artists = nx.draw_networkx_labels(
        G,
        pos,
        labels={i: str(i) for i in range(n_sub)},
        font_size=args.font_size,
        font_color=args.font_color,
        font_weight="bold",
        ax=ax,
    )
    for text in label_artists.values():
        text.set_zorder(10)

    # ------------------------------------------------------------------ #
    # Optional legend                                                      #
    # ------------------------------------------------------------------ #
    if args.show_legend:
        legend_handles = []
        # Substation node entry — derive marker size from the scatter node_size
        # node_size is a scatter area (points²); convert to a Line2D markersize (points)
        legend_marker_size = np.sqrt(args.node_size) * 0.5
        legend_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=args.node_color,
                markersize=legend_marker_size,
                linestyle="None",
                label="Substation",
            )
        )
        # Unique line styles
        seen_styles: set = set()
        for line_idx in range(n_line):
            color = args.line_colors[line_idx] if args.line_colors is not None else default_color
            width = args.line_widths[line_idx] if args.line_widths is not None else default_width
            style = args.line_styles[line_idx] if args.line_styles is not None else default_style
            key = (color, style)
            if key not in seen_styles:
                seen_styles.add(key)
                legend_handles.append(
                    Line2D([0], [0], color=color, lw=width, linestyle=style, label="Transmission Line")
                )
        ax.legend(
            handles=legend_handles,
            loc="best",
            frameon=False,
            prop={"size": args.legend_font_size},
        )

    ax.axis("off")

    if return_fig:
        fig.tight_layout()
        return fig
    return None


def latent_edge_hist(accumulated_edge_probabilities: npt.NDArray, skip_last_edge_type: bool = True):
    """
    Visualize a histogram showcasing the probabilities for different edges for any edge type except the first.

    :param accumulated_edge_probabilities: Edge probabilities of shape [E, num_edge_types] with probabilities for each edge - edge_type combination
    :param skip_last_edge_type: Whether to skip last edge type (default: True)
    :return: Reference to the Seaborn-styled matplotlib figure
    """
    if skip_last_edge_type:
        df = pd.DataFrame({'Edge probability': accumulated_edge_probabilities[:, :-1].sum(axis=-1).tolist()})
    else:
        df = pd.DataFrame({'Edge probability': accumulated_edge_probabilities.sum(axis=-1).tolist()})

    # Plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.histplot(df, x='Edge probability', bins=50, kde=True, color='skyblue', edgecolor='black', ax=ax)

    ax.set_title("Histogram of Latent Edge Probabilities", fontsize=18)
    ax.set_xlabel("Edge Probability", fontsize=14)
    ax.set_ylabel("Number of Edges", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    return fig


def _make_phantom_element_nodes(env, layout: dict) -> List[NodeStyle]:
    """
    Return invisible NodeStyle objects that match the Element+LODF bounding box.

    These phantom nodes have alpha=0 so they are not rendered, but their
    positions force matplotlib's autoscaling to use the same axes extent as
    ElementLODFGraphObservationConverter, giving the LODF graph the same scale.

    :param env: grid2op Environment
    :param layout: grid layout dict from PlotMatplot._grid_layout
    :return: list of phantom NodeStyle objects with alpha=0
    """
    r = 80.0
    bus_h = 32.0
    bus_v = 27.0

    def _pos_elem(sub_id: int, src_pos: npt.NDArray) -> npt.NDArray:
        target = np.array(layout[f"sub_{sub_id}"], dtype=float)
        vec = target - src_pos
        norm = np.linalg.norm(vec)
        return target if norm == 0 else target - (vec / norm) * r

    phantom: List[NodeStyle] = []

    for gid, sid in enumerate(env.gen_to_subid):
        src = np.array(layout.get(f"gen_{sid}_{gid}", layout[f"sub_{sid}"]), dtype=float)
        phantom.append(NodeStyle(position=_pos_elem(int(sid), src), color="green",   shape="p", size=780, label="Generator", alpha=0.0))

    for lid, sid in enumerate(env.load_to_subid):
        src = np.array(layout.get(f"load_{sid}_{lid}", layout[f"sub_{sid}"]), dtype=float)
        phantom.append(NodeStyle(position=_pos_elem(int(sid), src), color="orange",  shape="^", size=800, label="Load",      alpha=0.0))

    for stor_id, sid in enumerate(env.storage_to_subid):
        src = np.array(layout.get(f"storage_{sid}_{stor_id}", layout[f"sub_{sid}"]), dtype=float)
        phantom.append(NodeStyle(position=_pos_elem(int(sid), src), color="purple",  shape="D", size=780, label="Storage",   alpha=0.0))

    for sub_id in range(env.n_sub):
        base = np.array(layout[f"sub_{sub_id}"], dtype=float)
        phantom.append(NodeStyle(position=base + np.array([0.0,    -bus_v]), color="black",     shape="x", size=520, label="Ground",   alpha=0.0))
        phantom.append(NodeStyle(position=base + np.array([-bus_h, +bus_v]), color="steelblue", shape="s", size=800, label="Busbar 1", alpha=0.0))
        phantom.append(NodeStyle(position=base + np.array([+bus_h, +bus_v]), color="tomato",    shape="s", size=800, label="Busbar 2", alpha=0.0))

    return phantom


def get_node_styles(env: Environment, observation_space: type[ObservationConverter]) -> List[NodeStyle]:
    """
    For a given environment and observation space class, return a list of node style objects. Each node style object
    contains position, color and shape.
    :param env: the environment
    :param observation_space: the class of the observation space that dictates which entities are nodes
    :return: a list of node positions similar to the ones used by the grid2op plots
    """
    if observation_space in (GraphObservationConverter, HeterogeneousGraphObservationConverter):
        plot_helper = PlotMatplot(env.observation_space)

        r = 40.0
        layout = plot_helper._grid_layout

        def _natural_angle(sub_id: int, src_position: npt.NDArray) -> float:
            """Angle (radians) from substation center toward src_position."""
            center = np.array(layout[f"sub_{sub_id}"], dtype=float)
            vec = src_position - center
            norm = np.linalg.norm(vec)
            return float(np.arctan2(vec[1], vec[0])) if norm > 0 else 0.0

        # assemble substation IDs
        sub_ids = np.concatenate([
            env.line_or_to_subid,
            env.line_ex_to_subid,
            env.gen_to_subid,
            env.load_to_subid,
            env.storage_to_subid
        ])

        # assemble corresponding source locations (each row shape [2])
        pointing_towards_locs = [
            [layout[f"sub_{sid}"] for sid in env.line_ex_to_subid],
            [layout[f"sub_{sid}"] for sid in env.line_or_to_subid],
            [layout.get(f"gen_{sid}_{gid}", layout[f"sub_{sid}"]) for gid, sid in enumerate(env.gen_to_subid)],
            [layout.get(f"load_{sid}_{lid}", layout[f"sub_{sid}"]) for lid, sid in enumerate(env.load_to_subid)],
            [layout.get(f"storage_{sid}_{stor_id}", layout[f"sub_{sid}"]) for stor_id, sid in enumerate(env.storage_to_subid)],
        ]
        # filter out empty lists
        pointing_towards_locs = np.vstack([sub for sub in pointing_towards_locs if len(sub) > 0])

        # Seed positions: each element node starts on the circle of radius r at its natural angle
        n_elem = len(sub_ids)
        n_sub = env.n_sub
        init_pos: dict[int, npt.NDArray] = {}
        for node_idx, sid in enumerate(sub_ids):
            angle = _natural_angle(int(sid), np.array(pointing_towards_locs[node_idx]))
            center = np.array(layout[f"sub_{int(sid)}"], dtype=float)
            init_pos[node_idx] = center + r * np.array([np.cos(angle), np.sin(angle)])

        # Anchor nodes (one per substation) are fixed at substation centers.
        # Element nodes are attracted to their anchor (keeps them near their substation)
        # and to their powerline partner (line_or ↔ line_ex pairs), while repelling
        # every other element node — producing an organic, overlap-free layout.
        for sub_id in range(n_sub):
            init_pos[n_elem + sub_id] = np.array(layout[f"sub_{sub_id}"], dtype=float)

        G_layout = nx.Graph()
        G_layout.add_nodes_from(range(n_elem + n_sub))

        # Element → anchor edges: high weight keeps nodes close to substation center.
        # Powerline edges: low weight gives a gentle cross-substation pull.
        # Tune anchor_weight up (stiffer) to stay closer to initial positions.
        anchor_weight = 5.0
        powerline_weight = 0.1
        for node_idx, sub_id in enumerate(sub_ids):
            G_layout.add_edge(node_idx, n_elem + int(sub_id), weight=anchor_weight)

        for line_id in range(env.n_line):
            G_layout.add_edge(line_id, env.n_line + line_id, weight=powerline_weight)

        spring_pos = nx.spring_layout(
            G_layout,
            pos=init_pos,
            fixed=list(range(n_elem, n_elem + n_sub)),
            k=r,
            iterations=50,
            seed=0,
            weight="weight",
        )

        positions = [spring_pos[i] for i in range(n_elem)]
        colors = ["gray"] * 2 * env.n_line + ["green"] * env.n_gen + ["orange"] * env.n_load + [
            "purple"] * env.n_storage
        shapes = ["o"] * 2 * env.n_line + ["p"] * env.n_gen + ["^"] * env.n_load + ["D"] * env.n_storage
        labels = (["Powerline endpoint"] * 2 * env.n_line + ["Generator"] * env.n_gen +
                  ["Load"] * env.n_load + ["Storage"] * env.n_storage)
        sizes = [520] * 2 * env.n_line + [780] * (env.n_load + env.n_storage + env.n_gen)
        offsets = [(0.0, 0.0)] * 2 * env.n_line + [(0.0, 0.0)] * env.n_gen + [(0.0, -6.0)] * env.n_load + [(0.0, 0.0)] * env.n_storage

        node_styles = [
            NodeStyle(position=positions[i], color=colors[i], shape=shapes[i], label=labels[i], size=sizes[i], label_offset=offsets[i])
            for i in range(len(positions))
        ]

        return node_styles
    elif observation_space in (SubstationGraphObservationConverter,
                               SubstationPTDFGraphObservationConverter,
                               SubstationZbusGraphObservationConverter):
        plot_helper = PlotMatplot(env.observation_space)
        layout = plot_helper._grid_layout

        # Small horizontal offset to separate bus 1 and bus 2 nodes at each substation.
        offset = 32.0

        # Slot ordering: 2 * sub_id + (bus - 1), so bus 1 at even slots, bus 2 at odd slots.
        node_styles = []
        for sub_id in range(env.n_sub):
            base = np.array(layout[f"sub_{sub_id}"], dtype=float)
            node_styles.append(NodeStyle(
                position=base + np.array([-offset, 0.0]),
                color="steelblue",
                shape="s",
                size=800,
                label="Busbar 1",
            ))
            node_styles.append(NodeStyle(
                position=base + np.array([+offset, 0.0]),
                color="tomato",
                shape="s",
                size=800,
                label="Busbar 2",
            ))

        return node_styles
    elif observation_space in (ElementGraphObservationConverter, ElementLODFGraphObservationConverter):
        plot_helper = PlotMatplot(env.observation_space)
        layout = plot_helper._grid_layout

        r = 80.0           # distance from substation center for element nodes
        bus_h = 32.0       # horizontal half-spread for bus1/bus2
        bus_v = 27.0       # vertical half-spread: buses above, ground below

        def _pos_elem(sub_id: int, src_pos: npt.NDArray) -> npt.NDArray:
            """Offset element node toward its substation center by distance r."""
            target = np.array(layout[f"sub_{sub_id}"], dtype=float)
            vec = target - src_pos
            norm = np.linalg.norm(vec)
            if norm == 0:
                return target
            return target - (vec / norm) * r

        node_styles = []

        # Generators
        for gid, sid in enumerate(env.gen_to_subid):
            src = np.array(layout.get(f"gen_{sid}_{gid}", layout[f"sub_{sid}"]), dtype=float)
            node_styles.append(NodeStyle(
                position=_pos_elem(int(sid), src),
                color="green",
                shape="p",
                size=780,
                label="Generator",
            ))

        # Loads — label_offset shifts text down toward the triangle base
        for lid, sid in enumerate(env.load_to_subid):
            src = np.array(layout.get(f"load_{sid}_{lid}", layout[f"sub_{sid}"]), dtype=float)
            node_styles.append(NodeStyle(
                position=_pos_elem(int(sid), src),
                color="orange",
                shape="^",
                size=800,
                label="Load",
                label_offset=(0.0, -6.0),
            ))

        # Lines: single node at midpoint between origin and extremity substation
        for lid in range(env.n_line):
            or_pos = np.array(layout[f"sub_{int(env.line_or_to_subid[lid])}"], dtype=float)
            ex_pos = np.array(layout[f"sub_{int(env.line_ex_to_subid[lid])}"], dtype=float)
            node_styles.append(NodeStyle(
                position=(or_pos + ex_pos) / 2.0,
                color="dimgray",
                shape="h",
                size=520,
                label="Powerline",
            ))

        # Storage
        for stor_id, sid in enumerate(env.storage_to_subid):
            src = np.array(layout.get(f"storage_{sid}_{stor_id}", layout[f"sub_{sid}"]), dtype=float)
            node_styles.append(NodeStyle(
                position=_pos_elem(int(sid), src),
                color="purple",
                shape="D",
                size=780,
                label="Storage",
            ))

        # Bus/Ground nodes: 3 per substation in order (ground, bus1, bus2).
        # Arranged in a triangle: ground below center, bus1 upper-left, bus2 upper-right.
        for sub_id in range(env.n_sub):
            base = np.array(layout[f"sub_{sub_id}"], dtype=float)
            node_styles.append(NodeStyle(
                position=base + np.array([0.0, -bus_v]),
                color="black",
                shape="x",
                size=520,
                label="Ground",
            ))
            node_styles.append(NodeStyle(
                position=base + np.array([-bus_h, +bus_v]),
                color="steelblue",
                shape="s",
                size=800,
                label="Busbar 1",
            ))
            node_styles.append(NodeStyle(
                position=base + np.array([+bus_h, +bus_v]),
                color="tomato",
                shape="s",
                size=800,
                label="Busbar 2",
            ))

        return node_styles
    elif observation_space in (PTDFGraphObservationConverter, ZbusGraphObservationConverter):
        plot_helper = PlotMatplot(env.observation_space)
        layout = plot_helper._grid_layout
        offset = 32.0
        n_sub = env.n_sub

        # Bus slot ordering: slot = (busbar - 1) * n_sub + sub_id
        # Slots 0..n_sub-1 → busbar 1; slots n_sub..2*n_sub-1 → busbar 2.
        node_styles = [None] * (2 * n_sub)
        for sub_id in range(n_sub):
            base = np.array(layout[f"sub_{sub_id}"], dtype=float)
            node_styles[sub_id] = NodeStyle(
                position=base + np.array([-offset, 0.0]),
                color="steelblue",
                shape="s",
                size=800,
                label="Busbar 1",
            )
            node_styles[n_sub + sub_id] = NodeStyle(
                position=base + np.array([+offset, 0.0]),
                color="tomato",
                shape="s",
                size=800,
                label="Busbar 2",
            )
        return node_styles
    elif observation_space == LODFGraphObservationConverter:
        plot_helper = PlotMatplot(env.observation_space)
        layout = plot_helper._grid_layout

        # One visible node per powerline at the midpoint between its substations.
        node_styles = []
        for lid in range(env.n_line):
            or_pos = np.array(layout[f"sub_{int(env.line_or_to_subid[lid])}"], dtype=float)
            ex_pos = np.array(layout[f"sub_{int(env.line_ex_to_subid[lid])}"], dtype=float)
            node_styles.append(NodeStyle(
                position=(or_pos + ex_pos) / 2.0,
                color="dimgray",
                shape="h",
                size=520,
                label="Powerline",
            ))
        # Phantom nodes (alpha=0) matching the Element+LODF bounding box so that
        # matplotlib autoscales to the same extent as the Element+LODF graph.
        node_styles += _make_phantom_element_nodes(env, layout)
        return node_styles
    else:
        raise NotImplementedError()


def get_edge_styles(
    gym_obs: dict,
    observation_space: type[ObservationConverter],
    edge_mask: Optional[npt.NDArray] = None,
) -> Optional[List[EdgeStyle]]:
    """
    Return per-edge style objects for the active edges in a gym observation.

    Returns one :class:`EdgeStyle` per column in the masked ``EDGE_INDEX``,
    ordered the same way as ``gym_obs[EDGE_INDEX][:, mask]``.
    Returns *None* for converters that do not carry typed or attributed edges
    (e.g. :class:`GraphObservationConverter`, :class:`SubstationGraphObservationConverter`).

    :param gym_obs: gym observation dict produced by ``converter.to_gym(obs)``
    :param observation_space: the converter class used to produce ``gym_obs``
    :param edge_mask: optional boolean mask override (e.g. a top-K filter); if
                      *None* ``gym_obs[EDGE_MASK]`` is used.
    :return: list of :class:`EdgeStyle` objects (one per active edge), or *None*
    """
    mask = gym_obs[EDGE_MASK].astype(bool) if edge_mask is None else edge_mask.astype(bool)

    if observation_space == HeterogeneousGraphObservationConverter:
        edge_types = gym_obs[EDGE_TYPE][mask]
        return [_HETERO_EDGE_TYPE_STYLES[int(t)] for t in edge_types]

    if observation_space == ElementLODFGraphObservationConverter:
        edge_types = gym_obs[EDGE_TYPE][mask]   # [total_E]
        edge_attrs = gym_obs[EDGES][:, 0][mask]  # [total_E]

        # Normalise LODF weights (type-1 only) for colour mapping.
        lodf_weights = edge_attrs[edge_types == 1]
        if lodf_weights.size > 0:
            w_min, w_max = lodf_weights.min(), lodf_weights.max()
            norm_lodf = np.clip((lodf_weights - w_min) / (w_max - w_min + 1e-9), 0.0, 1.0)
        else:
            norm_lodf = np.empty(0)
        cmap = plt.get_cmap("YlOrRd")

        lodf_idx = 0
        styles: List[EdgeStyle] = []
        for t, attr in zip(edge_types, edge_attrs):
            if int(t) == 0:
                styles.append(_ELEM_EDGE_STYLES[int(np.clip(attr, 0, 1))])
            else:
                w = float(norm_lodf[lodf_idx])
                lodf_idx += 1
                styles.append(EdgeStyle(
                    color=mcolors.to_hex(cmap(w)),
                    width=0.1 + 2.0 * w,
                    alpha=0.1 + 0.7 * w,
                    label="LODF",
                ))
        return styles

    if observation_space == ElementGraphObservationConverter:
        edge_attrs = gym_obs[EDGES][:, 0][mask]
        return [_ELEM_EDGE_STYLES[int(a)] for a in edge_attrs]

    if observation_space in (PTDFGraphObservationConverter,
                             LODFGraphObservationConverter,
                             ZbusGraphObservationConverter):
        weights = np.nan_to_num(gym_obs[EDGES][:, 0][mask], nan=0.0, posinf=0.0, neginf=0.0)
        w_min, w_max = weights.min(), weights.max()
        norm_w = np.clip((weights - w_min) / (w_max - w_min + 1e-9), 0.0, 1.0)
        cmap = plt.get_cmap("YlOrRd")
        if observation_space is PTDFGraphObservationConverter:
            label = "PTDF coupling"
        elif observation_space is LODFGraphObservationConverter:
            label = "LODF"
        elif observation_space is ZbusGraphObservationConverter:
            label = "Zbus admittance"
        return [
            EdgeStyle(
                color=mcolors.to_hex(cmap(float(w))),
                width=0.1 + 3 * float(w),
                alpha=0.1 + 0.9 * float(w),
                label=label,
            )
            for w in norm_w
        ]

    if observation_space in (SubstationPTDFGraphObservationConverter,
                             SubstationZbusGraphObservationConverter):
        edge_types = gym_obs[EDGE_TYPE][mask]   # [total_E]
        edge_attrs = gym_obs[EDGES][:, 0][mask]  # [total_E]

        # Normalize physics-edge weights (type 1) for color mapping.
        type1_sel = edge_types == 1
        phys_weights = np.nan_to_num(edge_attrs[type1_sel], nan=0.0, posinf=0.0, neginf=0.0)
        if phys_weights.size > 0:
            w_min, w_max = phys_weights.min(), phys_weights.max()
            norm_phys = np.clip((phys_weights - w_min) / (w_max - w_min + 1e-9), 0.0, 1.0)
        else:
            norm_phys = np.empty(0)
        cmap = plt.get_cmap("YlOrRd")

        phys_idx = 0
        styles: List[EdgeStyle] = []
        for t in edge_types:
            if int(t) == 0:
                styles.append(EdgeStyle(
                    color="steelblue",
                    width=1.0,
                    alpha=0.7,
                    label="Topology",
                    linestyle="dashed",
                ))
            else:
                w = float(norm_phys[phys_idx])
                phys_idx += 1
                label = ("PTDF coupling"
                         if observation_space == SubstationPTDFGraphObservationConverter
                         else "Zbus admittance")
                styles.append(EdgeStyle(
                    color=mcolors.to_hex(cmap(w)),
                    width=0.1 + 2.5 * w,
                    alpha=0.1 + 0.8 * w,
                    label=label,
                ))
        return styles

    return None


def get_node_labels(
    env: Environment,
    observation_space: type[ObservationConverter],
) -> dict[int, str]:
    """
    Return a node_labels dict suitable for PlottingArgs.node_labels.

    Labels follow per-type enumeration aligned with grid2op IDs:
    - Loads, generators, storages, powerlines: grid2op element index.
    - Busbars: "{sub_id}:{bus_id}" (bus_id ∈ {1, 2}).
    - Powerline-bus-connection nodes (Graph/HeterogeneousGraph): no label
      (the edge carries the powerline ID instead).
    - Ground nodes: no label.

    :param env: grid2op Environment
    :param observation_space: converter class
    :return: dict mapping node index → label string
    """
    if observation_space in (GraphObservationConverter, HeterogeneousGraphObservationConverter):
        # Node order: [line_or×n | line_ex×n | gen | load | storage]
        # Powerline-bus-connection nodes get no label (edge carries the ID).
        n_line = env.n_line
        labels: dict[int, str] = {}
        gen_offset = 2 * n_line
        for gid in range(env.n_gen):
            labels[gen_offset + gid] = str(gid)
        load_offset = gen_offset + env.n_gen
        for lid in range(env.n_load):
            labels[load_offset + lid] = str(lid)
        storage_offset = load_offset + env.n_load
        for sid in range(env.n_storage):
            labels[storage_offset + sid] = str(sid)
        return labels

    if observation_space in (SubstationGraphObservationConverter,
                             SubstationPTDFGraphObservationConverter,
                             SubstationZbusGraphObservationConverter):
        return {}

    if observation_space in (ElementGraphObservationConverter, ElementLODFGraphObservationConverter):
        # Node order: [gen | load | line | storage | (ground, bus1, bus2)×n_sub]
        labels = {}
        gen_offset = 0
        load_offset = env.n_gen
        line_offset = env.n_gen + env.n_load
        storage_offset = line_offset + env.n_line

        for gid in range(env.n_gen):
            labels[gen_offset + gid] = str(gid)
        for lid in range(env.n_load):
            labels[load_offset + lid] = str(lid)
        for lid in range(env.n_line):
            labels[line_offset + lid] = str(lid)
        for sid in range(env.n_storage):
            labels[storage_offset + sid] = str(sid)

        return labels

    if observation_space in (PTDFGraphObservationConverter, ZbusGraphObservationConverter):
        return {}

    if observation_space == LODFGraphObservationConverter:
        # One node per powerline
        return {lid: str(lid) for lid in range(env.n_line)}

    raise NotImplementedError(f"get_node_labels not implemented for {observation_space}")


def get_curved_powerline_edges(
    env: Environment,
    g2op_obs,
    observation_space: type[ObservationConverter],
    line_ids: list[int],
    rad: float = 0.3,
) -> Optional[dict[tuple[int, int], float]]:
    """
    For substation-level converters return a curved_edges dict suitable for PlottingArgs,
    mapping the canonical node-pair of each requested powerline to the given arc radius.

    Only applies to SubstationGraphObservationConverter and its PTDF/Zbus variants.
    Returns None for all other converters.

    :param env: grid2op Environment
    :param g2op_obs: raw grid2op BaseObservation (provides bus assignments)
    :param observation_space: converter class
    :param line_ids: grid2op powerline indices to curve
    :param rad: arc3 radius — positive curves upward for left-to-right edges
    :return: dict (min_u, max_u) → rad, or None
    """
    if observation_space not in (SubstationGraphObservationConverter,
                                  SubstationPTDFGraphObservationConverter,
                                  SubstationZbusGraphObservationConverter):
        return None

    result: dict[tuple[int, int], float] = {}
    line_or_bus = g2op_obs.line_or_bus
    line_ex_bus = g2op_obs.line_ex_bus
    for lid in line_ids:
        if line_or_bus[lid] > 0 and line_ex_bus[lid] > 0:
            or_slot = int(2 * env.line_or_to_subid[lid] + (line_or_bus[lid] - 1))
            ex_slot = int(2 * env.line_ex_to_subid[lid] + (line_ex_bus[lid] - 1))
            result[(min(or_slot, ex_slot), max(or_slot, ex_slot))] = rad
    return result or None


def get_substation_node_groups(
    env: Environment,
    observation_space: type[ObservationConverter],
) -> Optional[dict[int, list[int]]]:
    """
    Return a mapping from substation ID to the list of node indices belonging to it.

    Used by visualize_graph to draw an enclosing circle around each substation cluster.
    Returns None for LODFGraphObservationConverter, where powerline nodes span two substations.

    :param env: grid2op Environment
    :param observation_space: converter class
    :return: dict sub_id → [node_idx, ...], or None
    """
    groups: dict[int, list[int]] = {s: [] for s in range(env.n_sub)}

    if observation_space in (GraphObservationConverter, HeterogeneousGraphObservationConverter):
        # Node order: [line_or×n | line_ex×n | gen | load | storage]
        n_line = env.n_line
        for lid in range(n_line):
            groups[int(env.line_or_to_subid[lid])].append(lid)
            groups[int(env.line_ex_to_subid[lid])].append(n_line + lid)
        gen_offset = 2 * n_line
        for gid in range(env.n_gen):
            groups[int(env.gen_to_subid[gid])].append(gen_offset + gid)
        load_offset = gen_offset + env.n_gen
        for lid in range(env.n_load):
            groups[int(env.load_to_subid[lid])].append(load_offset + lid)
        storage_offset = load_offset + env.n_load
        for sid in range(env.n_storage):
            groups[int(env.storage_to_subid[sid])].append(storage_offset + sid)
        return {s: idxs for s, idxs in groups.items() if idxs}

    if observation_space in (SubstationGraphObservationConverter,
                             SubstationPTDFGraphObservationConverter,
                             SubstationZbusGraphObservationConverter):
        # Two bus nodes per substation: slot = 2*sub_id + (bus-1)
        for sub_id in range(env.n_sub):
            groups[sub_id] = [2 * sub_id, 2 * sub_id + 1]
        return groups

    if observation_space in (ElementGraphObservationConverter, ElementLODFGraphObservationConverter):
        # Node order: [gen | load | line | storage | (ground, bus1, bus2)×n_sub]
        # Line nodes sit at the midpoint between two substations — excluded from grouping.
        gen_offset = 0
        load_offset = env.n_gen
        line_offset = env.n_gen + env.n_load
        storage_offset = line_offset + env.n_line
        bus_offset = storage_offset + env.n_storage
        for gid in range(env.n_gen):
            groups[int(env.gen_to_subid[gid])].append(gen_offset + gid)
        for lid in range(env.n_load):
            groups[int(env.load_to_subid[lid])].append(load_offset + lid)
        for sid in range(env.n_storage):
            groups[int(env.storage_to_subid[sid])].append(storage_offset + sid)
        for sub_id in range(env.n_sub):
            groups[sub_id].extend([
                bus_offset + 3 * sub_id,      # ground
                bus_offset + 3 * sub_id + 1,  # bus 1
                bus_offset + 3 * sub_id + 2,  # bus 2
            ])
        return {s: idxs for s, idxs in groups.items() if idxs}

    if observation_space in (PTDFGraphObservationConverter, ZbusGraphObservationConverter):
        # Slot = (bus-1)*n_sub + sub_id  →  bus1: sub_id, bus2: n_sub+sub_id
        n_sub = env.n_sub
        for sub_id in range(n_sub):
            groups[sub_id] = [sub_id, n_sub + sub_id]
        return groups

    # LODFGraphObservationConverter: powerlines span two substations — no grouping
    return None


def get_edge_labels(
    env: Environment,
    g2op_obs,
    gym_obs: dict,
    observation_space: type[ObservationConverter],
    display_mask: npt.NDArray,
) -> Optional[dict[tuple[int, int], str]]:
    """
    Return edge labels mapping canonical (min_u, max_u) → powerline ID string,
    for active edges in the masked edge index that represent powerlines.

    Physics-only and element-level converters return None (no edge labels).

    :param env: grid2op Environment
    :param g2op_obs: raw grid2op BaseObservation (provides bus assignments)
    :param gym_obs: gym observation dict from converter.to_gym(g2op_obs)
    :param observation_space: converter class
    :param display_mask: boolean mask selecting which edges to consider
    :return: dict (min_u, max_u) → str label, or None
    """
    # Converters where edges are not individual powerlines — no labels
    if observation_space in (ElementGraphObservationConverter,
                             ElementLODFGraphObservationConverter,
                             PTDFGraphObservationConverter,
                             LODFGraphObservationConverter,
                             ZbusGraphObservationConverter):
        return None

    edge_index = gym_obs[EDGE_INDEX][:, display_mask]  # [2, E_active]
    labels: dict[tuple[int, int], str] = {}

    if observation_space == GraphObservationConverter:
        # Powerline edges connect node lid (or) ↔ n_line + lid (ex)
        n_line = env.n_line
        line_edge_set = {(lid, n_line + lid) for lid in range(n_line)}
        for lid in range(n_line):
            key = (lid, n_line + lid)
            if key in {(int(min(u, v)), int(max(u, v))) for u, v in edge_index.T}:
                labels[key] = str(lid)
        return labels or None

    if observation_space == HeterogeneousGraphObservationConverter:
        # Type-0 edges are powerlines; same node mapping as GraphObservationConverter
        n_line = env.n_line
        edge_types = gym_obs[EDGE_TYPE][display_mask]
        for i, (u, v) in enumerate(edge_index.T):
            if int(edge_types[i]) == 0:
                key = (int(min(u, v)), int(max(u, v)))
                lid = int(min(u, v))  # line_or node index == line id
                labels[key] = str(lid)
        return labels or None

    if observation_space == SubstationGraphObservationConverter:
        # Reconstruct (or_slot, ex_slot) → line_id from bus assignments
        line_or_bus = g2op_obs.line_or_bus
        line_ex_bus = g2op_obs.line_ex_bus
        slot_to_lid: dict[tuple[int, int], int] = {}
        for lid in range(env.n_line):
            if line_or_bus[lid] > 0 and line_ex_bus[lid] > 0:
                or_slot = int(2 * env.line_or_to_subid[lid] + (line_or_bus[lid] - 1))
                ex_slot = int(2 * env.line_ex_to_subid[lid] + (line_ex_bus[lid] - 1))
                slot_to_lid[(min(or_slot, ex_slot), max(or_slot, ex_slot))] = lid
        for u, v in edge_index.T:
            key = (int(min(u, v)), int(max(u, v)))
            if key in slot_to_lid:
                labels[key] = str(slot_to_lid[key])
        return labels or None

    if observation_space in (SubstationPTDFGraphObservationConverter,
                             SubstationZbusGraphObservationConverter):
        # Only type-0 topology edges get labels; same slot→lid logic as SubstationGraph
        line_or_bus = g2op_obs.line_or_bus
        line_ex_bus = g2op_obs.line_ex_bus
        slot_to_lid = {}
        for lid in range(env.n_line):
            if line_or_bus[lid] > 0 and line_ex_bus[lid] > 0:
                or_slot = int(2 * env.line_or_to_subid[lid] + (line_or_bus[lid] - 1))
                ex_slot = int(2 * env.line_ex_to_subid[lid] + (line_ex_bus[lid] - 1))
                slot_to_lid[(min(or_slot, ex_slot), max(or_slot, ex_slot))] = lid
        edge_types = gym_obs[EDGE_TYPE][display_mask]
        for i, (u, v) in enumerate(edge_index.T):
            if int(edge_types[i]) == 0:
                key = (int(min(u, v)), int(max(u, v)))
                if key in slot_to_lid:
                    labels[key] = str(slot_to_lid[key])
        return labels or None

    return None


def visualize_posterior(latent_edge_posterior: npt.NDArray, latent_edge_prior: npt.NDArray,
                        skip_last: bool = True) -> Figure:
    assert latent_edge_posterior.shape == latent_edge_prior.shape
    assert latent_edge_posterior.ndim == 2

    if skip_last:
        latent_edge_prior = latent_edge_prior[:, :-1].sum(axis=-1, keepdims=True)
        latent_edge_posterior = latent_edge_posterior[:, :-1].sum(axis=-1, keepdims=True)

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    fig = plt.figure(figsize=(12, 7))
    bins = np.arange(0, 1.05, 0.05)

    data = pd.DataFrame({
        "value": np.concatenate([latent_edge_posterior.flatten(),
                                 latent_edge_prior.flatten()]),
        "group": ["posterior"] * len(latent_edge_posterior.flatten()) +
                 ["prior"] * len(latent_edge_prior.flatten())
    })

    sns.histplot(data=data, x="value", hue="group", bins=bins, multiple="layer", legend=True)

    plt.xlim((0, 1))
    plt.xlabel("Probability")
    plt.ylabel("Number of Edges")
    plt.title("Histogram of Latent Edge Probabilities (Prior & Posterior)")

    return fig

def visualize_prior(latent_edge_prior: npt.NDArray, skip_last: bool = True) -> Figure:

    if skip_last:
        latent_edge_prior = latent_edge_prior[:, :1]

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    fig = plt.figure(figsize=(12, 7))
    bins = np.arange(0, 1.05, 0.05)

    data = pd.DataFrame({"value": latent_edge_prior.flatten()})

    sns.histplot(data=data, x="value", bins=bins, multiple="layer")

    plt.xlim((0, 1))
    plt.xlabel("Probability")
    plt.ylabel("Number of Edges")
    plt.title("Histogram of Latent Edge Probabilities (Prior)")

    return fig
