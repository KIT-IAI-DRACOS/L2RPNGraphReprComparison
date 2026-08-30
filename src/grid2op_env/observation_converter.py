from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TypeVar, Generic

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from grid2op.Observation import BaseObservation, ObservationSpace
from grid2op.gym_compat import GymEnv
from gymnasium.spaces import Dict, Box
from gymnasium.wrappers.normalize import RunningMeanStd
from lightsim2grid import LightSimBackend

from grid2op_env.utils import get_attr_list

logger = logging.getLogger(__name__)

T = TypeVar("T")

NODES = "node_features"         # keys node features in the observation [N, x_dim]
NODE_MASK = "node_mask"         # keys boolean mask that masks which rows in the returned node-feature-tensor actually encode node features [N]
EDGES = "edge_features"         # keys edge features in the observation [E, e_dim]
EDGE_MASK = "edge_mask"         # keys boolean mask that masks which rows in the returned edge-feature-tensor actually encode edge features [E]
EDGE_INDEX = "edge_index"       # keys the edge index which notes node pairs that are connected by notes [2, E]
EDGE_TYPE = "edge_type"         # keys edge types per edge [E]
EDGE_WEIGHTS = "edge_weights"    # keys the edge weight per edge [E]
GLOBAL = "global_features"      # keys global features in the observation


_DEFAULT_NODE_FEATURES = [
    "active_power_forecast",
    "reactive_power_forecast",
    "active_power",
    "reactive_power",
    "voltage",
    "voltage_angle_cos",
    "voltage_angle_sin",
    "rho",
]


@dataclass
class _GridDimensions:
    """Static grid dimensions extracted once from the observation space."""

    n_gen: int
    n_load: int
    n_line: int
    n_storage: int

    @classmethod
    def from_obs_space(cls, obs_space: ObservationSpace) -> "_GridDimensions":
        """Extract grid dimensions from a grid2op observation space."""
        return cls(
            n_gen=obs_space.n_gen,
            n_load=obs_space.n_load,
            n_line=obs_space.n_line,
            n_storage=obs_space.n_storage,
        )



class ObservationConverter(ABC, Generic[T]):
    """
    Abstract base for observation converters. Each converter:
    - Exposes a gymnasium observation space for the RL agent via `observation_space`
    - Converts a grid2op observation to that space via `to_gym`
    - Normalizes the resulting observation via `normalize`
    """

    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Space:
        """The gymnasium observation space this converter produces."""

    @abstractmethod
    def to_gym(self, g2op_obs: BaseObservation) -> T:
        """Convert a grid2op observation to a gym-compatible observation."""

    @abstractmethod
    def normalize(self, gym_obs: T) -> T:
        """Normalize a gym observation using a running mean/variance estimate."""

    def close(self):
        pass

    def reset_obs(self):
        pass


class GraphObservationConverter(ObservationConverter[Dict]):
    """
    Converts grid2op observations into a graph-structured Dict observation:
      - NODES:      node feature matrix       [num_nodes, x_dim]
      - EDGE_INDEX: padded adjacency list     [2, max_num_edges]
      - EDGE_MASK:  boolean mask over edges   [max_num_edges]
      - GLOBAL:     global scalar features    [6]

    Node ordering: [line_or | line_ex | gen | load | storage]

    Edges encode:
      - same-substation + same-bus connectivity (dynamic, topology-dependent)
      - line endpoint pairs (static)

    Node features are normalized per-feature using a running mean/variance
    estimate, updated only when update_normalizer=True is passed to to_gym.
    """

    def __init__(
        self,
        g2op_obs_space: ObservationSpace,
        attr_to_observe: Optional[list[str]] = None,
        verbose: bool = False,
    ):
        if attr_to_observe is None:
            attr_to_observe = _DEFAULT_NODE_FEATURES

        self.g2op_obs_space = g2op_obs_space
        self.attr_to_observe = attr_to_observe
        self._dims = _GridDimensions.from_obs_space(g2op_obs_space)
        self._num_nodes = (
            2 * self._dims.n_line + self._dims.n_gen + self._dims.n_load + self._dims.n_storage
        )

        num_connections = g2op_obs_space.sub_info
        self._max_num_edges = int((num_connections * (num_connections - 1)).sum()) + 2 * self._dims.n_line
        x_dim = len(attr_to_observe)

        self._observation_space = Dict({
            NODES: Box(
                low=-np.inf, high=np.inf,
                shape=(self._num_nodes, x_dim),
                dtype=np.float32,
            ),
            EDGE_INDEX: Box(
                low=0, high=self._num_nodes - 1,
                shape=(2, self._max_num_edges),
                dtype=np.int64,
            ),
            EDGE_MASK: Box(
                low=0, high=1,
                shape=(self._max_num_edges,),
                dtype=np.bool_,
            ),
            NODE_MASK: Box(
                low=0, high=1,
                shape=(self._num_nodes,),
                dtype=np.bool_,
            ),
            GLOBAL: Box(
                low=-np.inf, high=np.inf,
                shape=(6,),
                dtype=np.float32,
            ),
        })

        # Normalize per feature, pooling across all nodes and timesteps.
        self._normalizer = RunningMeanStd(shape=(x_dim,))
        self._timings: dict[str, float] = {}

        # Precompute static grid topology (node ordering: [line_or | line_ex | gen | load | storage])
        sub_ids = np.concatenate([
            g2op_obs_space.line_or_to_subid,
            g2op_obs_space.line_ex_to_subid,
            g2op_obs_space.gen_to_subid,
            g2op_obs_space.load_to_subid,
            g2op_obs_space.storage_to_subid,
        ])
        # Candidate topology pairs: all (i, j) with i < j in the same substation.
        # Filtering to same-bus + connected is done dynamically; this eliminates the
        # O(N²) same_sub broadcast that was recomputed on every observation step.
        N = self._num_nodes
        ii, jj = np.triu_indices(N, k=1)
        same_sub_mask = sub_ids[ii] == sub_ids[jj]
        self._topo_cand_src = ii[same_sub_mask].astype(np.int64)
        self._topo_cand_dst = jj[same_sub_mask].astype(np.int64)

        line_or_nodes = np.arange(self._dims.n_line)
        line_ex_nodes = np.arange(self._dims.n_line, 2 * self._dims.n_line)
        self._line_edges = np.stack([
            np.concatenate([line_or_nodes, line_ex_nodes]),
            np.concatenate([line_ex_nodes, line_or_nodes]),
        ]).astype(np.int64)

        if verbose:
            logger.info(
                f"GraphObservationConverter: {self._num_nodes} nodes, "
                f"<= {self._max_num_edges} edges, {x_dim} features per node "
                f"({', '.join(attr_to_observe)})."
            )

    @property
    def observation_space(self) -> Dict:
        return self._observation_space

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def max_nodes(self) -> int:
        """Maximum number of nodes across all observations (= num_nodes for this converter)."""
        return self._num_nodes

    @property
    def max_num_edges(self) -> int:
        return self._max_num_edges

    @property
    def x_dim(self) -> int:
        return self._observation_space[NODES].shape[1]

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Convert a grid2op observation to a graph-structured gym observation.

        Args:
            g2op_obs: The grid2op observation to convert.
        """
        t_total = time.perf_counter()
        node_features = self._get_node_features(g2op_obs)
        t0 = time.perf_counter()
        edge_index = self._get_edge_index(g2op_obs)
        self._timings["edge_index_ms"] = (time.perf_counter() - t0) * 1000

        global_features = self._get_global_features(g2op_obs)

        num_edges = edge_index.shape[1]
        edge_index_padded = np.zeros((2, self._max_num_edges), dtype=np.int64)
        edge_index_padded[:, :num_edges] = edge_index
        edge_mask = np.zeros(self._max_num_edges, dtype=bool)
        edge_mask[:num_edges] = True
        node_mask = np.ones(self._num_nodes, dtype=np.bool_)

        result = self.normalize({
            NODES: node_features,
            EDGE_INDEX: edge_index_padded,
            EDGE_MASK: edge_mask,
            NODE_MASK: node_mask,
            GLOBAL: global_features,
        })
        self._timings["obs_conversion_ms"] = (time.perf_counter() - t_total) * 1000
        return result

    def normalize(self, gym_obs: dict) -> dict:
        """
        Normalize node features using per-feature running statistics.

        Args:
            gym_obs: Graph observation dict with raw node features.
        """
        node_features = gym_obs[NODES]  # (num_nodes, x_dim)
        # Update with each node as an independent sample of shape (x_dim,)
        self._normalizer.update(node_features)

        normalized = (node_features - self._normalizer.mean) / np.sqrt(self._normalizer.var + 1e-8)
        return {
            NODES: normalized.astype(np.float32),
            EDGE_INDEX: gym_obs[EDGE_INDEX],
            EDGE_MASK: gym_obs[EDGE_MASK],
            NODE_MASK: gym_obs[NODE_MASK],
            GLOBAL: gym_obs[GLOBAL],
        }

    def _compute_all_node_features(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray[np.float32]]:
        """
        Compute all supported node features, each as a flat array of length
        num_nodes in order [line_or | line_ex | gen | load | storage].
        """
        dims = self._dims
        zeros_line = np.zeros(dims.n_line, dtype=np.float32)
        zeros_gen = np.zeros(dims.n_gen, dtype=np.float32)
        zeros_load = np.zeros(dims.n_load, dtype=np.float32)
        zeros_storage = np.zeros(dims.n_storage, dtype=np.float32)

        if not g2op_obs._is_done:
            t0 = time.perf_counter()
            load_p, load_q, prod_p, prod_q, _ = g2op_obs.get_forecast_arrays()
            self._timings["forecast_ms"] = (time.perf_counter() - t0) * 1000
            # Index 1 = next timestep forecast; sign convention: gen positive, load negative
            gen_p_forecast = prod_p[1].astype(np.float32)
            gen_q_forecast = prod_q[1].astype(np.float32)
            load_p_forecast = -load_p[1].astype(np.float32)
            load_q_forecast = -load_q[1].astype(np.float32)
        else:
            self._timings["forecast_ms"] = 0.0
            gen_p_forecast = zeros_gen
            gen_q_forecast = zeros_gen
            load_p_forecast = zeros_load
            load_q_forecast = zeros_load

        return {
            "active_power_forecast": np.concatenate([
                zeros_line, zeros_line, gen_p_forecast, load_p_forecast, zeros_storage,
            ]),
            "reactive_power_forecast": np.concatenate([
                zeros_line, zeros_line, gen_q_forecast, load_q_forecast, zeros_storage,
            ]),
            # Generator convention: generation positive, consumption negative
            "active_power": np.concatenate([
                g2op_obs.p_or, g2op_obs.p_ex, g2op_obs.gen_p, -g2op_obs.load_p, g2op_obs.storage_power,
            ]),
            "reactive_power": np.concatenate([
                g2op_obs.q_or, g2op_obs.q_ex, g2op_obs.gen_q, -g2op_obs.load_q, zeros_storage,
            ]),
            "voltage": np.concatenate([
                g2op_obs.v_or, g2op_obs.v_ex, g2op_obs.gen_v, g2op_obs.load_v, zeros_storage,
            ]),
            "voltage_angle_cos": np.cos(np.deg2rad(np.concatenate([
                g2op_obs.theta_or, g2op_obs.theta_ex, g2op_obs.gen_theta, g2op_obs.load_theta, g2op_obs.storage_theta,
            ]))),
            "voltage_angle_sin": np.sin(np.deg2rad(np.concatenate([
                g2op_obs.theta_or, g2op_obs.theta_ex, g2op_obs.gen_theta, g2op_obs.load_theta, g2op_obs.storage_theta,
            ]))),
            # rho is a line-level quantity; non-line nodes get 0
            "rho": np.concatenate([
                g2op_obs.rho, g2op_obs.rho, zeros_gen, zeros_load, zeros_storage,
            ]),
        }

    def _get_node_features(self, g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """Assemble the node feature matrix for the requested attributes."""
        all_features = self._compute_all_node_features(g2op_obs)
        node_features = np.column_stack([all_features[name] for name in self.attr_to_observe])
        return node_features.astype(np.float32)

    def _get_edge_index(self, g2op_obs: BaseObservation) -> npt.NDArray[np.int64]:
        """
        Build the edge index for the current topology.

        Two types of edges:
          1. Same-substation + same-bus pairs (dynamic, changes with topology actions)
          2. Line origin <-> extremity pairs (dynamic, excluded when the line is disconnected)

        Disconnected elements (bus == -1) are excluded from both edge types.

        Complexity: O(N + E) per call instead of O(N²), achieved by grouping
        connected nodes by (substation, bus) key and generating pairs only within
        each group. The old N×N broadcast approach allocates four boolean matrices
        of size N×N on every step, which is ~10× more memory/compute on IEEE36
        (N=177) than on IEEE14 (N=57).
        """
        bus_ids = np.concatenate([
            g2op_obs.line_or_bus, g2op_obs.line_ex_bus,
            g2op_obs.gen_bus, g2op_obs.load_bus,
            g2op_obs.storage_bus,
        ])

        # Filter precomputed same-substation candidate pairs by dynamic bus assignment.
        # Both endpoints must be on the same bus and connected (bus > 0).
        src = self._topo_cand_src
        dst = self._topo_cand_dst
        valid = (bus_ids[src] == bus_ids[dst]) & (bus_ids[src] > 0)
        topo_src = src[valid]
        topo_dst = dst[valid]
        topo_edges = np.stack([
            np.concatenate([topo_src, topo_dst]),
            np.concatenate([topo_dst, topo_src]),
        ])

        # Filter line edges: exclude disconnected lines (bus_ids layout is
        # [line_or | line_ex | gen | load | storage], so line_or_bus = bus_ids[:n_line]
        # and line_ex_bus = bus_ids[n_line:2*n_line]).
        n_line = self._dims.n_line
        line_connected = (bus_ids[:n_line] > 0) & (bus_ids[n_line:2 * n_line] > 0)
        # _line_edges columns: first n_line are or→ex, next n_line are ex→or
        line_mask = np.concatenate([line_connected, line_connected])
        active_line_edges = self._line_edges[:, line_mask]

        return np.concatenate([topo_edges, active_line_edges], axis=1)

    @staticmethod
    def _get_global_features(g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """Extract time-based global features."""
        return np.array([
            g2op_obs.year,
            g2op_obs.month,
            g2op_obs.day,
            g2op_obs.hour_of_day,
            g2op_obs.day_of_week,
            g2op_obs.minute_of_hour,
        ], dtype=np.float32)


class HeterogeneousGraphObservationConverter(GraphObservationConverter):
    """
    Extends GraphObservationConverter with typed edges (heterogeneous graph).

    Edges are split into three types, each processed by a separate conv layer:
      0: powerline edges         (line_or <-> line_ex)
      1: same-substation, same-bus   (active physical connectivity)
      2: same-substation, diff-bus   (optional / latent connectivity)

    Adds EDGE_TYPE: [max_num_edges] int64 to the observation dict.
    Node features, normalization, and padding are inherited unchanged.
    """

    def __init__(
        self,
        g2op_obs_space: ObservationSpace,
        attr_to_observe: Optional[list[str]] = None,
        verbose: bool = False,
    ):
        super().__init__(g2op_obs_space, attr_to_observe, verbose)
        spaces = dict(self._observation_space.spaces)
        spaces[EDGE_TYPE] = Box(
            low=0, high=2,
            shape=(self._max_num_edges,),
            dtype=np.int64,
        )
        self._observation_space = Dict(spaces)

    def _get_edge_index_with_types(self, g2op_obs: BaseObservation) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """
        Build the edge index and per-edge type array for the current topology.

        Returns:
            edge_index: [2, E] connectivity matrix (int64)
            edge_types: [E] type index per edge: 0=powerline, 1=same-bus, 2=optional (int64)
        """
        bus_ids = np.concatenate([
            g2op_obs.line_or_bus, g2op_obs.line_ex_bus,
            g2op_obs.gen_bus, g2op_obs.load_bus,
            g2op_obs.storage_bus,
        ])

        src = self._topo_cand_src
        dst = self._topo_cand_dst
        both_connected = (bus_ids[src] > 0) & (bus_ids[dst] > 0)

        # Type 1: same substation, same bus (physical connectivity)
        same_bus = (bus_ids[src] == bus_ids[dst]) & both_connected
        t1_src, t1_dst = src[same_bus], dst[same_bus]

        # Type 2: same substation, different bus (optional connectivity)
        diff_bus = ~(bus_ids[src] == bus_ids[dst]) & both_connected
        t2_src, t2_dst = src[diff_bus], dst[diff_bus]

        # Make bidirectional (candidates are upper-triangle only)
        def _bidir(s: npt.NDArray, d: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
            return np.concatenate([s, d]), np.concatenate([d, s])

        t1_s, t1_d = _bidir(t1_src, t1_dst)
        t2_s, t2_d = _bidir(t2_src, t2_dst)

        # Type 0: line edges (already bidirectional in _line_edges), filter disconnected
        n_line = self._dims.n_line
        line_connected = (bus_ids[:n_line] > 0) & (bus_ids[n_line:2 * n_line] > 0)
        line_mask = np.concatenate([line_connected, line_connected])
        active_line_edges = self._line_edges[:, line_mask]
        n_line_edges = active_line_edges.shape[1]

        edge_index = np.stack([
            np.concatenate([active_line_edges[0], t1_s, t2_s]),
            np.concatenate([active_line_edges[1], t1_d, t2_d]),
        ])
        edge_types = np.concatenate([
            np.zeros(n_line_edges, dtype=np.int64),
            np.ones(len(t1_s), dtype=np.int64),
            np.full(len(t2_s), 2, dtype=np.int64),
        ])
        return edge_index, edge_types

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Convert a grid2op observation to a heterogeneous graph gym observation.

        Args:
            g2op_obs: The grid2op observation to convert.
        """
        t_total = time.perf_counter()

        node_features = self._get_node_features(g2op_obs)

        t0 = time.perf_counter()
        edge_index, edge_types = self._get_edge_index_with_types(g2op_obs)
        self._timings["edge_index_ms"] = (time.perf_counter() - t0) * 1000

        global_features = self._get_global_features(g2op_obs)

        num_edges = edge_index.shape[1]
        edge_index_padded = np.zeros((2, self._max_num_edges), dtype=np.int64)
        edge_index_padded[:, :num_edges] = edge_index
        edge_mask = np.zeros(self._max_num_edges, dtype=bool)
        edge_mask[:num_edges] = True
        edge_type_padded = np.zeros(self._max_num_edges, dtype=np.int64)
        edge_type_padded[:num_edges] = edge_types
        node_mask = np.ones(self._num_nodes, dtype=np.bool_)

        result = self.normalize({
            NODES: node_features,
            EDGE_INDEX: edge_index_padded,
            EDGE_MASK: edge_mask,
            NODE_MASK: node_mask,
            EDGE_TYPE: edge_type_padded,
            GLOBAL: global_features,
        })
        self._timings["obs_conversion_ms"] = (time.perf_counter() - t_total) * 1000
        return result

    def normalize(self, gym_obs: dict) -> dict:
        """Normalize node features and pass edge types through unchanged."""
        result = super().normalize(gym_obs)
        result[EDGE_TYPE] = gym_obs[EDGE_TYPE]
        return result


class SubstationGraphObservationConverter(GraphObservationConverter):
    """
    Substation-level graph: one node per active busbar, powerlines as edges.

    Each substation contributes 1–2 nodes depending on whether its elements are
    split across bus 1 and bus 2. Padded to max_nodes = 2 * n_sub with NODE_MASK.

    Node features use the same attr_to_observe as GraphObservationConverter,
    aggregated across all elements assigned to each bus:
      - power features (active/reactive): sum
      - voltage, voltage_angle: mean
      - rho: max

    Edges are one bidirectional pair per connected powerline, linking the bus
    nodes at each line's origin and extremity.
    """

    # Aggregation strategy when collapsing per-element features to per-bus nodes.
    # Features not listed here default to "sum".
    _BUS_AGGREGATION: dict[str, str] = {
        "voltage": "mean",
        "voltage_angle_sin": "mean",
        "voltage_angle_cos": "mean",
        "rho": "max",
    }

    def __init__(
        self,
        g2op_obs_space: ObservationSpace,
        attr_to_observe: Optional[list[str]] = None,
        verbose: bool = False,
    ):
        super().__init__(g2op_obs_space, attr_to_observe, verbose)
        # super().__init__ sets: _dims, _timings, _normalizer, attr_to_observe,
        # _observation_space, _topo_cand_src/dst, _line_edges (all for element graph).
        # We override the structure-specific parts below.

        n_sub = g2op_obs_space.n_sub
        self._n_sub = n_sub
        self._max_nodes = 2 * n_sub
        self._max_num_edges = 2 * self._dims.n_line  # bidirectional line edges

        # Substation id for each element in flat ordering [line_or|line_ex|gen|load|storage]
        self._sub_ids_flat = np.concatenate([
            g2op_obs_space.line_or_to_subid,
            g2op_obs_space.line_ex_to_subid,
            g2op_obs_space.gen_to_subid,
            g2op_obs_space.load_to_subid,
            g2op_obs_space.storage_to_subid,
        ]).astype(np.int64)

        # Line endpoint substation ids for edge building
        self._line_or_subid = g2op_obs_space.line_or_to_subid.astype(np.int64)
        self._line_ex_subid = g2op_obs_space.line_ex_to_subid.astype(np.int64)

        # Precompute which feature indices use mean/max aggregation (rest default to sum)
        self._mean_feat_indices: list[int] = [
            i for i, name in enumerate(self.attr_to_observe)
            if self._BUS_AGGREGATION.get(name) == "mean"
        ]
        self._max_feat_indices: list[int] = [
            i for i, name in enumerate(self.attr_to_observe)
            if self._BUS_AGGREGATION.get(name) == "max"
        ]

        x_dim = len(self.attr_to_observe)
        self._observation_space = Dict({
            NODES: Box(low=-np.inf, high=np.inf, shape=(self._max_nodes, x_dim), dtype=np.float32),
            EDGE_INDEX: Box(low=0, high=self._max_nodes - 1, shape=(2, self._max_num_edges), dtype=np.int64),
            EDGE_MASK: Box(low=0, high=1, shape=(self._max_num_edges,), dtype=np.bool_),
            NODE_MASK: Box(low=0, high=1, shape=(self._max_nodes,), dtype=np.bool_),
            GLOBAL: Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })
        self._normalizer = RunningMeanStd(shape=(x_dim,))

        if verbose:
            logger.info(
                f"SubstationGraphObservationConverter: {self._max_nodes} max nodes "
                f"({n_sub} subs × 2 buses), {self._max_num_edges} max edges, "
                f"{x_dim} features per node ({', '.join(self.attr_to_observe)})."
            )

    @property
    def num_nodes(self) -> int:
        """Maximum number of nodes (active count varies per observation)."""
        return self._max_nodes

    @property
    def max_nodes(self) -> int:
        return self._max_nodes

    @property
    def max_num_edges(self) -> int:
        return self._max_num_edges

    def _get_nodes_and_mask(self, g2op_obs: BaseObservation) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """
        Aggregate per-element features into per-bus node features.

        Node slot formula: 2 * sub_id + (bus - 1) for bus ∈ {1, 2}.
        Disconnected elements (bus == -1) are excluded.

        Returns:
            node_features: [max_nodes, x_dim] float32
            node_mask:     [max_nodes] bool — True for slots with ≥1 element
        """
        all_features = self._compute_all_node_features(g2op_obs)

        bus_ids_flat = np.concatenate([
            g2op_obs.line_or_bus, g2op_obs.line_ex_bus,
            g2op_obs.gen_bus, g2op_obs.load_bus,
            g2op_obs.storage_bus,
        ])
        connected = bus_ids_flat > 0
        active_slots = (2 * self._sub_ids_flat + (bus_ids_flat - 1))[connected].astype(np.int64)

        x_dim = len(self.attr_to_observe)
        node_features = np.zeros((self._max_nodes, x_dim), dtype=np.float64)
        node_count = np.zeros(self._max_nodes, dtype=np.int64)
        node_max = np.full((self._max_nodes, x_dim), -np.inf, dtype=np.float64)

        np.add.at(node_count, active_slots, 1)

        for f_idx, feat_name in enumerate(self.attr_to_observe):
            values = all_features[feat_name][connected]
            if self._BUS_AGGREGATION.get(feat_name) == "max":
                np.maximum.at(node_max[:, f_idx], active_slots, values)
            else:  # sum (also used as first step for mean)
                np.add.at(node_features[:, f_idx], active_slots, values)

        node_mask = node_count > 0
        for f_idx in self._mean_feat_indices:
            node_features[node_mask, f_idx] /= node_count[node_mask]
        for f_idx in self._max_feat_indices:
            node_features[node_mask, f_idx] = node_max[node_mask, f_idx]

        return node_features.astype(np.float32), node_mask

    def _get_edge_index(self, g2op_obs: BaseObservation) -> npt.NDArray[np.int64]:
        """
        Build edge index: one bidirectional edge per connected powerline.

        Endpoints map to bus node slots: slot = 2 * sub_id + (bus - 1).
        """
        line_or_bus = g2op_obs.line_or_bus
        line_ex_bus = g2op_obs.line_ex_bus
        connected = (line_or_bus > 0) & (line_ex_bus > 0)

        or_slots = (2 * self._line_or_subid + (line_or_bus - 1))[connected]
        ex_slots = (2 * self._line_ex_subid + (line_ex_bus - 1))[connected]

        return np.stack([
            np.concatenate([or_slots, ex_slots]),
            np.concatenate([ex_slots, or_slots]),
        ]).astype(np.int64)

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Convert a grid2op observation to a substation-level graph gym observation.

        Args:
            g2op_obs: The grid2op observation to convert.
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        node_features, node_mask = self._get_nodes_and_mask(g2op_obs)
        self._timings["node_features_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        edge_index = self._get_edge_index(g2op_obs)
        self._timings["edge_index_ms"] = (time.perf_counter() - t0) * 1000

        global_features = self._get_global_features(g2op_obs)

        num_edges = edge_index.shape[1]
        edge_index_padded = np.zeros((2, self._max_num_edges), dtype=np.int64)
        edge_index_padded[:, :num_edges] = edge_index
        edge_mask = np.zeros(self._max_num_edges, dtype=bool)
        edge_mask[:num_edges] = True

        result = self.normalize({
            NODES: node_features,
            EDGE_INDEX: edge_index_padded,
            EDGE_MASK: edge_mask,
            NODE_MASK: node_mask,
            GLOBAL: global_features,
        })
        self._timings["obs_conversion_ms"] = (time.perf_counter() - t_total) * 1000
        return result

    def normalize(self, gym_obs: dict) -> dict:
        """
        Normalize node features using per-feature running statistics.

        Only active nodes (NODE_MASK=True) update the normalizer. Inactive
        node features are zeroed out after normalization.
        """
        node_features = gym_obs[NODES]   # (max_nodes, x_dim)
        node_mask = gym_obs[NODE_MASK]   # (max_nodes,)

        active_features = node_features[node_mask]
        if active_features.shape[0] > 0:
            self._normalizer.update(active_features)

        normalized = (node_features - self._normalizer.mean) / np.sqrt(self._normalizer.var + 1e-8)
        normalized[~node_mask] = 0.0

        return {
            NODES: normalized.astype(np.float32),
            EDGE_INDEX: gym_obs[EDGE_INDEX],
            EDGE_MASK: gym_obs[EDGE_MASK],
            NODE_MASK: gym_obs[NODE_MASK],
            GLOBAL: gym_obs[GLOBAL],
        }


class ElementGraphObservationConverter(ObservationConverter[Dict]):
    """
    Element-level graph: one node per physical grid element (gen, load, line,
    storage, bus, ground). Fixed static topology with dynamic edge features.

    Node ordering: [gen | load | line | storage | (ground, bus1, bus2) × n_sub]

    Node features (x_dim=28, heterogeneous, zero-padded per type):
      Slots  0- 4: base     — |p|, p, q, |v|, cos(θ)               (all nodes)
      Slots  5-17: gen      — g_norm, g_maxup, g_maxdown, g_minuptime,
                               g_mindowntime, g_cost, g_startcost,
                               g_shutdowncost, g_type×5             (generators)
      Slots 18-21: bus      — b_ground, b_bus1, b_bus2, b_cooldown  (buses)
      Slots 22-25: line     — ρ, p_tsoverflow, p_tscooldown,
                               p_maintenance                         (powerlines)
      Slots 26-27: forecast — p_forecast, q_forecast                (gen/load only)

    Edges are static (precomputed once): each element is connected to every
    busbar and ground node of its substation(s). Edge feature EDGE_ATTR[e, 0]
    is 1 if the element is currently assigned to that specific bus, 0 otherwise.
    EDGE_MASK is all-True (no padding — edge count is fixed).
    """

    # Feature layout for ElementGraphObservationConverter (x_dim = 28).
    # Slots 0-4: base features shared by all node types.
    # Slots 5-17: generator-specific features (zero for non-generators).
    # Slots 18-21: bus-specific features (zero for non-buses).
    # Slots 22-25: powerline-specific features (zero for non-powerlines).
    # Slots 26-27: forecast features for gen/load (zero for all other types).
    _ELEM_X_DIM = 28
    _ELEM_BASE_SLICE = slice(0, 5)        # |p|, p, q, |v|, cos(θ)
    _ELEM_GEN_SLICE = slice(5, 18)        # g_norm, g_maxup, ..., g_type×5
    _ELEM_BUS_SLICE = slice(18, 22)       # b_ground, b_bus1, b_bus2, b_cooldown
    _ELEM_LINE_SLICE = slice(22, 26)      # ρ, p_tsoverflow, p_tscooldown, p_maintenance
    _ELEM_FORECAST_SLICE = slice(26, 28)  # p_forecast, q_forecast

    # One-hot generator type encoding order used in grid2op
    _GEN_TYPE_ORDER = ["solar", "wind", "hydro", "thermal", "nuclear"]

    # Bus-slot offsets within each substation block: (ground=0, bus1=1, bus2=2)
    _GROUND_OFFSET = 0
    _BUS1_OFFSET = 1
    _BUS2_OFFSET = 2

    def __init__(
        self,
        g2op_obs_space: ObservationSpace,
        verbose: bool = False,
    ):
        n_gen = g2op_obs_space.n_gen
        n_load = g2op_obs_space.n_load
        n_line = g2op_obs_space.n_line
        n_storage = g2op_obs_space.n_storage
        n_sub = g2op_obs_space.n_sub

        self._n_gen = n_gen
        self._n_load = n_load
        self._n_line = n_line
        self._n_storage = n_storage
        self._n_sub = n_sub

        # Node index offsets for each element group
        self._gen_offset = 0
        self._load_offset = n_gen
        self._line_offset = n_gen + n_load
        self._storage_offset = n_gen + n_load + n_line
        self._bus_offset = n_gen + n_load + n_line + n_storage  # start of (ground, bus1, bus2) blocks
        self._num_nodes = n_gen + n_load + n_line + n_storage + 3 * n_sub

        # Substation membership for each element group
        self._gen_subid = g2op_obs_space.gen_to_subid.astype(np.int64)
        self._load_subid = g2op_obs_space.load_to_subid.astype(np.int64)
        self._line_or_subid = g2op_obs_space.line_or_to_subid.astype(np.int64)
        self._line_ex_subid = g2op_obs_space.line_ex_to_subid.astype(np.int64)
        self._storage_subid = g2op_obs_space.storage_to_subid.astype(np.int64)

        # Static generator features (don't change per step)
        self._static_gen_features = self._build_static_gen_features(g2op_obs_space)

        # Precompute static edge_index and lookup tables for dynamic edge_attr
        self._edge_index, self._edge_element_idx, self._edge_bus_idx, \
            self._fwd_bus_selector = self._build_static_edges()
        self._num_edges = self._edge_index.shape[1]

        self._observation_space = Dict({
            NODES: Box(low=-np.inf, high=np.inf,
                       shape=(self._num_nodes, self._ELEM_X_DIM), dtype=np.float32),
            EDGE_INDEX: Box(low=0, high=self._num_nodes - 1,
                            shape=(2, self._num_edges), dtype=np.int64),
            EDGE_MASK: Box(low=0, high=1,
                           shape=(self._num_edges,), dtype=np.bool_),
            EDGES: Box(low=0, high=1,
                       shape=(self._num_edges, 1), dtype=np.float32),
            NODE_MASK: Box(low=0, high=1,
                           shape=(self._num_nodes,), dtype=np.bool_),
            GLOBAL: Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })

        self._normalizer = RunningMeanStd(shape=(self._ELEM_X_DIM,))
        self._timings: dict[str, float] = {}

        if verbose:
            logger.info(
                f"ElementGraphObservationConverter: {self._num_nodes} nodes "
                f"({n_gen} gen, {n_load} load, {n_line} line, {n_storage} storage, "
                f"{3 * n_sub} bus/ground), {self._num_edges} edges, "
                f"x_dim={self._ELEM_X_DIM}."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def observation_space(self) -> Dict:
        return self._observation_space

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def max_nodes(self) -> int:
        return self._num_nodes

    @property
    def x_dim(self) -> int:
        return self._ELEM_X_DIM

    @property
    def num_edges(self) -> int:
        return self._num_edges

    # ------------------------------------------------------------------
    # Static precomputation helpers
    # ------------------------------------------------------------------

    def _bus_node(self, sub_id: int, bus: int) -> int:
        """
        Node index for a busbar slot.

        Args:
            sub_id: Substation index.
            bus:    0=ground, 1=bus1, 2=bus2.
        Returns:
            Absolute node index.
        """
        return self._bus_offset + 3 * sub_id + bus

    def _build_static_gen_features(self, obs_space: ObservationSpace) -> npt.NDArray[np.float32]:
        """
        Precompute the time-invariant generator feature columns (slots 5-17).

        Returns:
            [n_gen, 13] float32 array of static generator features.
        """
        n_gen = self._n_gen
        feats = np.zeros((n_gen, 13), dtype=np.float32)

        # g_norm placeholder (slot 0 of gen block) — computed dynamically per step
        # g_maxup … g_shutdowncost (slots 1-8)
        feats[:, 1] = obs_space.gen_max_ramp_up
        feats[:, 2] = obs_space.gen_max_ramp_down
        feats[:, 3] = obs_space.gen_min_uptime
        feats[:, 4] = obs_space.gen_min_downtime
        feats[:, 5] = obs_space.gen_cost_per_MW
        feats[:, 6] = obs_space.gen_startup_cost
        feats[:, 7] = obs_space.gen_shutdown_cost

        # g_type one-hot (slots 8-12)
        for i, gen_type in enumerate(obs_space.gen_type):
            t = gen_type.lower()
            if t in self._GEN_TYPE_ORDER:
                feats[i, 8 + self._GEN_TYPE_ORDER.index(t)] = 1.0

        return feats

    def _build_static_edges(self) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]:
        """
        Build the static (time-invariant) bidirectional edge index.

        Each element is connected to every busbar slot (ground=0, bus1=1, bus2=2)
        at its substation. Lines connect at both endpoints (origin + extremity).
        Loads do NOT connect to ground (disconnecting a load ends the episode).

        Also builds fwd_bus_selector: an index into the per-timestep array
        [gen_bus | load_bus | line_or_bus | line_ex_bus | storage_bus] for each
        forward edge. Line or-side and ex-side edges point to different slices,
        so _get_edge_attr uses the correct bus assignment for each endpoint.

        Returns:
            edge_index:        [2, E] int64 — (src, dst) pairs
            edge_element_idx:  [E//2] int64 — element node index for each forward edge
            edge_bus_idx:      [E//2] int64 — bus node index for each forward edge
            fwd_bus_selector:  [E//2] int64 — index into concat bus array per forward edge
        """
        srcs, dsts = [], []
        elem_idxs, bus_idxs, selectors = [], [], []

        # Offsets into [gen_bus | load_bus | line_or_bus | line_ex_bus | storage_bus]
        sel_gen_base = 0
        sel_load_base = self._n_gen
        sel_line_or_base = self._n_gen + self._n_load
        sel_line_ex_base = self._n_gen + self._n_load + self._n_line
        sel_storage_base = self._n_gen + self._n_load + 2 * self._n_line

        def _add(elem_node: int, bus_node: int, selector: int) -> None:
            srcs.extend([elem_node, bus_node])
            dsts.extend([bus_node, elem_node])
            elem_idxs.append(elem_node)
            bus_idxs.append(bus_node)
            selectors.append(selector)

        # Generators: connect to ground, bus1, bus2 at their substation
        for i in range(self._n_gen):
            s = int(self._gen_subid[i])
            for b in (self._GROUND_OFFSET, self._BUS1_OFFSET, self._BUS2_OFFSET):
                _add(self._gen_offset + i, self._bus_node(s, b), sel_gen_base + i)

        # Loads: connect to bus1, bus2 only (no ground)
        for i in range(self._n_load):
            s = int(self._load_subid[i])
            for b in (self._BUS1_OFFSET, self._BUS2_OFFSET):
                _add(self._load_offset + i, self._bus_node(s, b), sel_load_base + i)

        # Lines: connect to ground, bus1, bus2 at both origin and extremity substations.
        # Or-side edges use line_or_bus; ex-side edges use line_ex_bus.
        for i in range(self._n_line):
            line_node = self._line_offset + i
            for b in (self._GROUND_OFFSET, self._BUS1_OFFSET, self._BUS2_OFFSET):
                _add(line_node, self._bus_node(int(self._line_or_subid[i]), b),
                     sel_line_or_base + i)
            for b in (self._GROUND_OFFSET, self._BUS1_OFFSET, self._BUS2_OFFSET):
                _add(line_node, self._bus_node(int(self._line_ex_subid[i]), b),
                     sel_line_ex_base + i)

        # Storage: connect to ground, bus1, bus2 at their substation
        for i in range(self._n_storage):
            s = int(self._storage_subid[i])
            for b in (self._GROUND_OFFSET, self._BUS1_OFFSET, self._BUS2_OFFSET):
                _add(self._storage_offset + i, self._bus_node(s, b), sel_storage_base + i)

        edge_index = np.array([srcs, dsts], dtype=np.int64)
        edge_element_idx = np.array(elem_idxs, dtype=np.int64)
        edge_bus_idx = np.array(bus_idxs, dtype=np.int64)
        fwd_bus_selector = np.array(selectors, dtype=np.int64)
        return edge_index, edge_element_idx, edge_bus_idx, fwd_bus_selector

    # ------------------------------------------------------------------
    # Per-step feature computation
    # ------------------------------------------------------------------

    def _get_node_features(self, g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """
        Build the [num_nodes, 28] node feature matrix for the current observation.

        Args:
            g2op_obs: Current grid2op observation.
        Returns:
            Node feature matrix [num_nodes, 28] float32.
        """
        X = np.zeros((self._num_nodes, self._ELEM_X_DIM), dtype=np.float32)

        # --- Forecasts (slots 26-27): next-timestep p/q for gen and load ---
        if not g2op_obs._is_done:
            load_p_fc, load_q_fc, prod_p_fc, prod_q_fc, _ = g2op_obs.get_forecast_arrays()
            gen_p_forecast = prod_p_fc[1].astype(np.float32)
            gen_q_forecast = prod_q_fc[1].astype(np.float32)
            load_p_forecast = load_p_fc[1].astype(np.float32)
            load_q_forecast = load_q_fc[1].astype(np.float32)
        else:
            gen_p_forecast = np.zeros(self._n_gen, dtype=np.float32)
            gen_q_forecast = np.zeros(self._n_gen, dtype=np.float32)
            load_p_forecast = np.zeros(self._n_load, dtype=np.float32)
            load_q_forecast = np.zeros(self._n_load, dtype=np.float32)

        # --- Generators ---
        g_p = g2op_obs.gen_p.astype(np.float32)
        g_q = g2op_obs.gen_q.astype(np.float32)
        g_v = g2op_obs.gen_v.astype(np.float32)
        g_theta = g2op_obs.gen_theta.astype(np.float32)
        g_idx = slice(self._gen_offset, self._gen_offset + self._n_gen)
        X[g_idx, 0] = np.abs(g_p)
        X[g_idx, 1] = g_p
        X[g_idx, 2] = g_q
        X[g_idx, 3] = np.abs(g_v)
        X[g_idx, 4] = np.cos(np.deg2rad(g_theta))
        # Static gen features (slots 5-17), g_norm is dynamic (slot 5)
        X[g_idx, self._ELEM_GEN_SLICE] = self._static_gen_features
        p_min = np.abs(g2op_obs.gen_pmin).astype(np.float32)
        p_max = np.abs(g2op_obs.gen_pmax).astype(np.float32)
        denom = p_max - p_min
        g_norm = np.where(denom > 0, (np.abs(g_p) - p_min) / denom, 0.0)
        X[g_idx, 5] = g_norm  # overwrite slot 5 (g_norm)
        X[g_idx, 26] = gen_p_forecast
        X[g_idx, 27] = gen_q_forecast

        # --- Loads ---
        l_p = g2op_obs.load_p.astype(np.float32)
        l_q = g2op_obs.load_q.astype(np.float32)
        l_v = g2op_obs.load_v.astype(np.float32)
        l_theta = g2op_obs.load_theta.astype(np.float32)
        l_idx = slice(self._load_offset, self._load_offset + self._n_load)
        X[l_idx, 0] = np.abs(l_p)
        X[l_idx, 1] = l_p
        X[l_idx, 2] = l_q
        X[l_idx, 3] = np.abs(l_v)
        X[l_idx, 4] = np.cos(np.deg2rad(l_theta))
        X[l_idx, 26] = load_p_forecast
        X[l_idx, 27] = load_q_forecast

        # --- Lines (single node per line, use origin-side electrical quantities) ---
        ln_p = g2op_obs.p_or.astype(np.float32)
        ln_q = g2op_obs.q_or.astype(np.float32)
        ln_v = g2op_obs.v_or.astype(np.float32)
        ln_theta = g2op_obs.theta_or.astype(np.float32)
        ln_idx = slice(self._line_offset, self._line_offset + self._n_line)
        X[ln_idx, 0] = np.abs(ln_p)
        X[ln_idx, 1] = ln_p
        X[ln_idx, 2] = ln_q
        X[ln_idx, 3] = np.abs(ln_v)
        X[ln_idx, 4] = np.cos(np.deg2rad(ln_theta))
        X[ln_idx, 22] = g2op_obs.rho.astype(np.float32)
        X[ln_idx, 23] = g2op_obs.timestep_overflow.astype(np.float32)
        X[ln_idx, 24] = g2op_obs.time_before_cooldown_line.astype(np.float32)
        X[ln_idx, 25] = g2op_obs.duration_next_maintenance.astype(np.float32)

        # --- Storage ---
        if self._n_storage > 0:
            st_p = g2op_obs.storage_power.astype(np.float32)
            st_idx = slice(self._storage_offset, self._storage_offset + self._n_storage)
            # grid2op doesn't expose per-storage voltage/theta directly; use zeros
            X[st_idx, 0] = np.abs(st_p)
            X[st_idx, 1] = st_p

        # --- Bus / Ground nodes ---
        cooldown = g2op_obs.time_before_cooldown_sub.astype(np.float32)  # [n_sub]
        for s in range(self._n_sub):
            ground_node = self._bus_node(s, self._GROUND_OFFSET)
            bus1_node = self._bus_node(s, self._BUS1_OFFSET)
            bus2_node = self._bus_node(s, self._BUS2_OFFSET)
            cd = cooldown[s]
            # one-hot b_type: [b_ground, b_bus1, b_bus2]
            X[ground_node, 18] = 1.0
            X[ground_node, 21] = cd
            X[bus1_node, 19] = 1.0
            X[bus1_node, 21] = cd
            X[bus2_node, 20] = 1.0
            X[bus2_node, 21] = cd

        return X

    def _get_edge_attr(self, g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """
        Build the dynamic edge attribute array: 1 if the element is currently
        connected to the specific bus node on that edge, else 0.

        Uses _fwd_bus_selector to index into a concatenated bus array
        [gen_bus | load_bus | line_or_bus | line_ex_bus | storage_bus], so
        or-side and ex-side line edges get the correct endpoint bus assignment.

        Args:
            g2op_obs: Current grid2op observation.
        Returns:
            Edge attribute array [num_edges, 1] float32.
        """
        def _clamp(arr: npt.NDArray) -> npt.NDArray[np.int64]:
            """Map bus -1 (disconnected) → 0 (ground slot)."""
            return np.where(arr > 0, arr, 0).astype(np.int64)

        storage_bus = (
            _clamp(g2op_obs.storage_bus) if self._n_storage > 0
            else np.zeros(0, dtype=np.int64)
        )
        # Layout: [gen_bus | load_bus | line_or_bus | line_ex_bus | storage_bus]
        all_buses = np.concatenate([
            _clamp(g2op_obs.gen_bus),
            _clamp(g2op_obs.load_bus),
            _clamp(g2op_obs.line_or_bus),
            _clamp(g2op_obs.line_ex_bus),
            storage_bus,
        ])

        bus_slot = (self._edge_bus_idx - self._bus_offset) % 3  # 0=ground, 1=bus1, 2=bus2
        active = (all_buses[self._fwd_bus_selector] == bus_slot).astype(np.float32)

        # edge_index interleaves (elem→bus, bus→elem) per edge pair, so forward
        # edges land at even positions and reverse edges at odd positions.
        attr_full = np.zeros(self._num_edges, dtype=np.float32)
        attr_full[0::2] = active
        attr_full[1::2] = active

        return attr_full[:, np.newaxis]

    # ------------------------------------------------------------------
    # ObservationConverter interface
    # ------------------------------------------------------------------

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Convert a grid2op observation to an element-level graph gym observation.

        Args:
            g2op_obs: The grid2op observation to convert.
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        node_features = self._get_node_features(g2op_obs)
        self._timings["node_features_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        edge_attr = self._get_edge_attr(g2op_obs)
        self._timings["edge_attr_ms"] = (time.perf_counter() - t0) * 1000

        global_features = self._get_global_features(g2op_obs)
        node_mask = np.ones(self._num_nodes, dtype=np.bool_)
        edge_mask = np.ones(self._num_edges, dtype=np.bool_)

        result = self.normalize({
            NODES: node_features,
            EDGE_INDEX: self._edge_index,
            EDGE_MASK: edge_mask,
            EDGES: edge_attr,
            NODE_MASK: node_mask,
            GLOBAL: global_features,
        })
        self._timings["obs_conversion_ms"] = (time.perf_counter() - t_total) * 1000
        return result

    def normalize(self, gym_obs: dict) -> dict:
        """
        Normalize node features using per-feature running statistics.

        Args:
            gym_obs: Graph observation dict with raw node features.
        """
        node_features = gym_obs[NODES]  # [num_nodes, 26]
        self._normalizer.update(node_features)
        normalized = (node_features - self._normalizer.mean) / np.sqrt(self._normalizer.var + 1e-8)
        return {
            NODES: normalized.astype(np.float32),
            EDGE_INDEX: gym_obs[EDGE_INDEX],
            EDGE_MASK: gym_obs[EDGE_MASK],
            EDGES: gym_obs[EDGES],
            NODE_MASK: gym_obs[NODE_MASK],
            GLOBAL: gym_obs[GLOBAL],
        }

    @staticmethod
    def _get_global_features(g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """Extract time-based global features."""
        return np.array([
            g2op_obs.year, g2op_obs.month, g2op_obs.day,
            g2op_obs.hour_of_day, g2op_obs.day_of_week, g2op_obs.minute_of_hour,
        ], dtype=np.float32)


class ElementLODFGraphObservationConverter(ElementGraphObservationConverter):
    """
    Element-level graph extended with LODF contingency-coupling edges.

    Inherits the element-to-busbar static topology from
    :class:`ElementGraphObservationConverter`:

    * Type 0 — element-bus connections (binary edge attr: 1 if active bus assignment)

    Adds a second edge type:

    * Type 1 — directed LODF coupling edges (line[i] → line[j], i ≠ j)

    Powerline nodes sit at indices ``line_offset + i`` = ``n_gen + n_load + i``.
    Edge weight ``EDGES[e, 0]`` for a type-1 edge is ``|LODF[i, j]|`` — the
    fractional flow change on line i when line j trips.

    All type-0 edges carry the binary bus-assignment attribute from the parent so
    that ``EDGES`` is uniform in shape across both types.  Type-0 edges are always
    active; type-1 LODF edges are active only when both line i and line j are
    connected (``line_status[i] & line_status[j]``).

    LODF depends only on network topology, so it is cached per ``line_status``
    and recomputed only on topology changes via the shared
    :func:`_compute_lodf` module-level helper.

    ``EDGE_TYPE`` is declared as ``Box(low=0, high=1, …)``; the downstream
    :class:`~rl.gnn_ppo_model.GNNModel` derives
    ``num_edge_types = high + 1 = 2`` automatically.

    Observation keys
    ----------------
    NODES      : [num_nodes, 28]         — element node features (identical to parent)
    EDGE_INDEX : [2, max_num_edges]      — padded connectivity
    EDGE_MASK  : [max_num_edges]         — True for active edges
    EDGES      : [max_num_edges, 1]      — bus assignment (type 0) or |LODF| (type 1)
    EDGE_TYPE  : [max_num_edges]         — 0/1 per edge, 0 for padding
    NODE_MASK  : [num_nodes]             — all True (element nodes always present)
    GLOBAL     : [6]                     — time-based global features

    :param backend: LightSimBackend instance (for LODF via ``get_lodf()``).
    :param g2op_obs_space: grid2op observation space.
    :param verbose: log converter dimensions on construction.
    """

    def __init__(
        self,
        backend: LightSimBackend,
        g2op_obs_space: ObservationSpace,
        verbose: bool = False,
    ):
        super().__init__(g2op_obs_space, verbose)
        self._grid = backend._grid

        n_line = self._n_line

        # Directed LODF edges: line[i] → line[j] for all i ≠ j.
        # Powerline nodes start at self._line_offset = n_gen + n_load.
        self._n_lodf_edges = n_line * (n_line - 1)
        idx = np.arange(n_line)
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        no_self_loop = ii != jj
        self._lodf_edge_index = np.stack([
            ii[no_self_loop] + self._line_offset,
            jj[no_self_loop] + self._line_offset,
        ]).astype(np.int64)  # [2, n_lodf_edges]

        self._max_num_edges = self._num_edges + self._n_lodf_edges

        # Rebuild obs space with padded shapes and new EDGE_TYPE key.
        self._observation_space = Dict({
            NODES: Box(low=-np.inf, high=np.inf,
                       shape=(self._num_nodes, self._ELEM_X_DIM), dtype=np.float32),
            EDGE_INDEX: Box(low=0, high=self._num_nodes - 1,
                            shape=(2, self._max_num_edges), dtype=np.int64),
            EDGE_MASK: Box(low=0, high=1,
                           shape=(self._max_num_edges,), dtype=np.bool_),
            EDGES: Box(low=-np.inf, high=np.inf,
                       shape=(self._max_num_edges, 1), dtype=np.float32),
            EDGE_TYPE: Box(low=0, high=1,
                           shape=(self._max_num_edges,), dtype=np.int64),
            NODE_MASK: Box(low=0, high=1,
                           shape=(self._num_nodes,), dtype=np.bool_),
            GLOBAL: Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })

        self._edge_normalizer = RunningMeanStd(shape=(1,))
        self._cached_D: Optional[np.ndarray] = None
        self._cached_line_status: Optional[np.ndarray] = None

        if verbose:
            logger.info(
                f"ElementLODFGraphObservationConverter: {self._num_nodes} nodes, "
                f"<= {self._max_num_edges} edges "
                f"({self._num_edges} element-bus + {self._n_lodf_edges} LODF), "
                f"x_dim={self._ELEM_X_DIM}."
            )

    def _get_lodf_cached(self, line_status: np.ndarray) -> None:
        """
        Update _cached_D when line_status changes. Delegates to :func:`_compute_lodf`.

        Args:
            line_status: Boolean array [n_line] of current line connection state.
        """
        self._cached_D, self._cached_line_status = _compute_lodf(
            self._grid, self._n_line, line_status,
            self._cached_line_status, self._cached_D,
        )

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Convert a grid2op observation to an element+LODF graph observation.

        Type-0 edges are the static element-bus connections (binary bus-assignment
        attr); type-1 LODF edges are directed powerline-to-powerline edges.

        Args:
            g2op_obs: Current grid2op observation.
        Returns:
            Dict with keys NODES, EDGE_INDEX, EDGE_MASK, EDGES, EDGE_TYPE,
            NODE_MASK, GLOBAL.
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        node_features = self._get_node_features(g2op_obs)
        self._timings["node_features_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        edge_attr = self._get_edge_attr(g2op_obs)   # [num_edges, 1] binary
        self._timings["edge_attr_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self._get_lodf_cached(g2op_obs.line_status)
        self._timings["lodf_ms"] = (time.perf_counter() - t0) * 1000

        # Local line indices (0-based) for LODF src/dst lookup.
        lodf_src_local = self._lodf_edge_index[0] - self._line_offset
        lodf_dst_local = self._lodf_edge_index[1] - self._line_offset
        raw_lodf = self._cached_D[lodf_src_local, lodf_dst_local].astype(np.float32)
        line_status = g2op_obs.line_status.astype(bool)
        lodf_mask = line_status[lodf_src_local] & line_status[lodf_dst_local]

        global_features = self._get_global_features(g2op_obs)

        n0 = self._num_edges
        n1 = self._n_lodf_edges
        E = self._max_num_edges

        edge_index_padded = np.zeros((2, E), dtype=np.int64)
        edge_index_padded[:, :n0] = self._edge_index
        edge_index_padded[:, n0:n0 + n1] = self._lodf_edge_index

        edge_mask = np.zeros(E, dtype=bool)
        edge_mask[:n0] = True                      # type-0 always active
        edge_mask[n0:n0 + n1] = lodf_mask

        edge_type = np.zeros(E, dtype=np.int64)
        edge_type[n0:n0 + n1] = 1

        edges_padded = np.zeros((E, 1), dtype=np.float32)
        edges_padded[:n0] = edge_attr
        edges_padded[n0:n0 + n1, 0] = raw_lodf

        node_mask = np.ones(self._num_nodes, dtype=np.bool_)

        result = self.normalize({
            NODES: node_features,
            EDGE_INDEX: edge_index_padded,
            EDGE_MASK: edge_mask,
            EDGES: edges_padded,
            EDGE_TYPE: edge_type,
            NODE_MASK: node_mask,
            GLOBAL: global_features,
        })
        self._timings["obs_conversion_ms"] = (time.perf_counter() - t_total) * 1000
        return result

    def normalize(self, gym_obs: dict) -> dict:
        """
        Normalize node features and LODF edge attributes with separate running statistics.

        * Nodes: running normalizer from parent (all element nodes always present).
        * Type-1 (LODF) edges: separate running normalizer on active |LODF| values.
        * Type-0 (element-bus) edges: binary attribute, passed through unchanged.
        * EDGE_TYPE, EDGE_INDEX, NODE_MASK, GLOBAL pass through unchanged.

        Args:
            gym_obs: Raw observation dict from to_gym.
        """
        node_features = gym_obs[NODES]
        self._normalizer.update(node_features)
        normalized_nodes = (node_features - self._normalizer.mean) / np.sqrt(self._normalizer.var + 1e-8)

        edges = gym_obs[EDGES]          # (E, 1)
        edge_mask = gym_obs[EDGE_MASK]  # (E,)
        edge_type = gym_obs[EDGE_TYPE]  # (E,)

        lodf_active = edge_mask & (edge_type == 1)
        active_lodf_vals = edges[lodf_active]
        if active_lodf_vals.shape[0] > 0:
            self._edge_normalizer.update(active_lodf_vals)

        normalized_edges = np.zeros_like(edges)
        normalized_edges[edge_type == 0] = edges[edge_type == 0]   # pass binary attrs through
        if lodf_active.any():
            normalized_edges[lodf_active] = (
                (active_lodf_vals - self._edge_normalizer.mean)
                / np.sqrt(self._edge_normalizer.var + 1e-8)
            )

        return {
            NODES: normalized_nodes.astype(np.float32),
            EDGE_INDEX: gym_obs[EDGE_INDEX],
            EDGE_MASK: edge_mask,
            EDGES: normalized_edges.astype(np.float32),
            EDGE_TYPE: edge_type,
            NODE_MASK: gym_obs[NODE_MASK],
            GLOBAL: gym_obs[GLOBAL],
        }


class FlatObservationConverter(ObservationConverter[gym.spaces.Dict]):
    """
    Converts grid2op observations into a flat Dict observation where each key
    corresponds to a named g2op attribute (e.g. rho, p_or, load_p).

    Features are normalized online using a running mean and variance estimate
    over the concatenated flat observation vector.
    """

    def __init__(
        self,
        gym_env: GymEnv,
        attr_to_observe: list[str],
        verbose: bool = False,
    ):
        self.attr_to_observe = attr_to_observe

        # Use the gym env's obs space only to derive attribute shapes and raw extraction
        self._raw_obs_space = gym_env.observation_space.keep_only_attr(attr_to_observe)

        # Build observation space: unbounded boxes since normalization is applied at runtime
        self._observation_space = gym.spaces.Dict({
            attr: Box(low=-np.inf, high=np.inf, shape=space.shape, dtype=np.float32)
            for attr, space in self._raw_obs_space.spaces.items()
        })

        # Single running normalizer over the full concatenated flat vector
        total_dim = sum(int(np.prod(space.shape)) for space in self._raw_obs_space.spaces.values())
        self.normalizer = RunningMeanStd(shape=(total_dim,))

        if verbose:
            logger.info(
                f"FlatObservationConverter: {len(attr_to_observe)} attributes, "
                f"{total_dim} total features ({', '.join(attr_to_observe)})."
            )

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._observation_space

    def to_gym(self, g2op_obs: BaseObservation) -> dict:
        raw = dict(self._raw_obs_space.to_gym(g2op_obs))
        return self.normalize(raw)

    def normalize(self, gym_obs: dict) -> dict:
        flat = np.concatenate([
            np.asarray(gym_obs[attr], dtype=np.float64).ravel()
            for attr in self.attr_to_observe
        ])
        self.normalizer.update(flat[np.newaxis])
        normalized = (flat - self.normalizer.mean) / np.sqrt(self.normalizer.var + 1e-8)

        result = {}
        offset = 0
        for attr in self.attr_to_observe:
            shape = self._observation_space[attr].shape
            size = int(np.prod(shape))
            result[attr] = normalized[offset:offset + size].reshape(shape).astype(np.float32)
            offset += size
        return result


def _compute_ptdf_distance(
    grid,
    line_status: np.ndarray,
    cached_line_status: Optional[np.ndarray],
    cached_ptdf: Optional[np.ndarray],
    cached_D: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (ptdf, D, line_status_copy), recomputing only when line_status changes.

    D[i, j] = sum_l |PTDF[l, i] - PTDF[l, j]|  (L1 PTDF distance, [2*n_sub, 2*n_sub]).

    DC PF is called directly on the C++ grid model, which is safe because
    lightsim2grid keeps separate DC and AC solver objects.

    Args:
        grid: lightsim2grid GridModel instance (backend._grid).
        line_status: Boolean array [n_line] of current line connection state.
        cached_line_status: Previously cached line_status (or None).
        cached_ptdf: Previously cached PTDF matrix (or None).
        cached_D: Previously cached distance matrix (or None).
    Returns:
        (ptdf, D, line_status_copy) — ptdf is [n_branches, 2*n_sub],
        D is [2*n_sub, 2*n_sub] symmetric, non-negative.
    Raises:
        RuntimeError: if the DC power flow diverges.
    """
    if cached_line_status is not None and np.array_equal(line_status, cached_line_status):
        return cached_ptdf, cached_D, cached_line_status
    Vinit = np.ones(grid.total_bus(), dtype=complex)
    Vdc = grid.dc_pf(Vinit, 10, 1e-8)
    if Vdc.shape[0] == 0:
        raise RuntimeError(
            "DC power flow diverged; PTDF cannot be computed for current topology."
        )
    ptdf = grid.get_ptdf()
    diff = ptdf[:, :, np.newaxis] - ptdf[:, np.newaxis, :]  # (n_br, n_bus, n_bus)
    D = np.nan_to_num(np.abs(diff).sum(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    return ptdf, D, line_status.copy()


class PTDFGraphObservationConverter(GraphObservationConverter):
    """
    PTDF-based electrical distance graph.

    Nodes represent busbars (up to 2 per substation → max_nodes = 2*n_sub).
    Edges are fully connected over all bus-slot pairs; the edge weight is the
    L1 PTDF distance:

        D[i, j] = sum_l |PTDF[l, i] - PTDF[l, j]|

    which measures how differently a unit injection at bus i vs bus j
    redistributes power across the transmission lines.

    NODE_MASK marks buses with ≥1 connected element (active buses).
    EDGE_MASK marks directed edges where both endpoint buses are active.
    EDGES carries the running-normalized PTDF distance as a scalar [E, 1].

    PTDF is topology-dependent but operating-point-independent (DC linearization),
    so it is cached per line-status vector and only recomputed on topology changes.

    Node features use the same attr_to_observe as GraphObservationConverter,
    aggregated across all elements assigned to each bus:
      - power features (active/reactive): sum
      - voltage, voltage_angle: mean
      - rho: max

    Bus slot convention (aligns with PTDF column indices from lightsim2grid):
        slot = (busbar - 1) * n_sub + sub_id    for busbar ∈ {1, 2}
    Slots 0..n_sub-1 → busbar 1; slots n_sub..2*n_sub-1 → busbar 2.
    """

    _BUS_AGGREGATION: dict[str, str] = {
        "voltage": "mean",
        "voltage_angle_sin": "mean",
        "voltage_angle_cos": "mean",
        "rho": "max",
    }

    def __init__(
        self,
        backend: LightSimBackend,
        g2op_obs_space: ObservationSpace,
        attr_to_observe: Optional[list[str]] = None,
        verbose: bool = False,
    ):
        # Call parent to initialize _dims, _timings, attr_to_observe, _normalizer,
        # and _compute_all_node_features (element-level feature extraction).
        # Structure-specific attributes (_observation_space, _max_num_edges, etc.)
        # are overridden below.
        super().__init__(g2op_obs_space, attr_to_observe, verbose)

        self._grid = backend._grid
        self._n_sub = g2op_obs_space.n_sub

        n_sub = self._n_sub
        self._max_nodes = 2 * n_sub
        self._max_num_edges = self._max_nodes * (self._max_nodes - 1)  # directed, no self-loops

        # Substation id per element in flat order [line_or|line_ex|gen|load|storage],
        # used to compute PTDF-convention bus slots: slot = (busbar-1)*n_sub + sub_id.
        self._sub_ids_flat = np.concatenate([
            g2op_obs_space.line_or_to_subid,
            g2op_obs_space.line_ex_to_subid,
            g2op_obs_space.gen_to_subid,
            g2op_obs_space.load_to_subid,
            g2op_obs_space.storage_to_subid,
        ]).astype(np.int64)

        # Feature aggregation indices (same strategy as SubstationGraphObservationConverter)
        self._mean_feat_indices: list[int] = [
            i for i, name in enumerate(self.attr_to_observe)
            if self._BUS_AGGREGATION.get(name) == "mean"
        ]
        self._max_feat_indices: list[int] = [
            i for i, name in enumerate(self.attr_to_observe)
            if self._BUS_AGGREGATION.get(name) == "max"
        ]

        # Static fully-connected directed edge index over all bus slots (no self-loops)
        idx = np.arange(self._max_nodes)
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        no_self_loop = ii != jj
        self._edge_index_full = np.stack([ii[no_self_loop], jj[no_self_loop]]).astype(np.int64)

        x_dim = len(self.attr_to_observe)
        self._observation_space = Dict({
            NODES: Box(low=-np.inf, high=np.inf, shape=(self._max_nodes, x_dim), dtype=np.float32),
            EDGE_INDEX: Box(low=0, high=self._max_nodes - 1, shape=(2, self._max_num_edges), dtype=np.int64),
            EDGE_MASK: Box(low=0, high=1, shape=(self._max_num_edges,), dtype=np.bool_),
            EDGES: Box(low=-np.inf, high=np.inf, shape=(self._max_num_edges, 1), dtype=np.float32),
            NODE_MASK: Box(low=0, high=1, shape=(self._max_nodes,), dtype=np.bool_),
            GLOBAL: Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })

        # Override node normalizer from parent; add scalar edge normalizer.
        self._normalizer = RunningMeanStd(shape=(x_dim,))
        self._edge_normalizer = RunningMeanStd(shape=(1,))

        # PTDF cache: only recompute when line_status changes.
        # _cached_D is derived from _cached_ptdf and shares the same lifetime.
        self._cached_ptdf: Optional[np.ndarray] = None
        self._cached_D: Optional[np.ndarray] = None
        self._cached_line_status: Optional[np.ndarray] = None

        if verbose:
            logger.info(
                f"PTDFGraphObservationConverter: {self._max_nodes} max nodes "
                f"({n_sub} subs × 2 buses), {self._max_num_edges} directed edges, "
                f"{x_dim} features per node ({', '.join(self.attr_to_observe)})."
            )

    @property
    def num_nodes(self) -> int:
        """Total bus slots (2 * n_sub); same as max_nodes for this converter."""
        return self._max_nodes

    @property
    def max_nodes(self) -> int:
        """Maximum number of nodes across all observations (= 2 * n_sub)."""
        return self._max_nodes

    @property
    def max_num_edges(self) -> int:
        """Number of directed edges in the fully-connected graph (max_nodes * (max_nodes - 1))."""
        return self._max_num_edges

    def _get_ptdf_cached(self, line_status: np.ndarray) -> np.ndarray:
        """
        Return the PTDF matrix, recomputing only when line_status has changed.

        Delegates to the module-level :func:`_compute_ptdf_distance` helper which
        is shared with :class:`SubstationPTDFGraphObservationConverter`.

        Args:
            line_status: Boolean array [n_line] of current line connection state.
        Returns:
            PTDF matrix [n_branches, 2*n_sub].
        """
        self._cached_ptdf, self._cached_D, self._cached_line_status = _compute_ptdf_distance(
            self._grid, line_status,
            self._cached_line_status, self._cached_ptdf, self._cached_D,
        )
        return self._cached_ptdf

    def _get_nodes_and_mask(
        self, g2op_obs: BaseObservation
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """
        Aggregate per-element features into per-bus node features.

        Bus slot formula: slot = (busbar - 1) * n_sub + sub_id, busbar ∈ {1, 2}.
        Disconnected elements (bus == -1) are excluded.

        Args:
            g2op_obs: Current grid2op observation.
        Returns:
            node_features: [max_nodes, x_dim] float32
            node_mask:     [max_nodes] bool — True for slots with ≥1 connected element
        """
        all_features = self._compute_all_node_features(g2op_obs)

        bus_ids_flat = np.concatenate([
            g2op_obs.line_or_bus, g2op_obs.line_ex_bus,
            g2op_obs.gen_bus, g2op_obs.load_bus,
            g2op_obs.storage_bus,
        ])
        connected = bus_ids_flat > 0
        # PTDF bus slot: (busbar - 1) * n_sub + sub_id
        active_slots = (
            (bus_ids_flat[connected] - 1) * self._n_sub + self._sub_ids_flat[connected]
        ).astype(np.int64)

        x_dim = len(self.attr_to_observe)
        node_features = np.zeros((self._max_nodes, x_dim), dtype=np.float64)
        node_count = np.zeros(self._max_nodes, dtype=np.int64)
        node_max = np.full((self._max_nodes, x_dim), -np.inf, dtype=np.float64)

        np.add.at(node_count, active_slots, 1)

        for f_idx, feat_name in enumerate(self.attr_to_observe):
            values = all_features[feat_name][connected]
            if self._BUS_AGGREGATION.get(feat_name) == "max":
                np.maximum.at(node_max[:, f_idx], active_slots, values)
            else:
                np.add.at(node_features[:, f_idx], active_slots, values)

        node_mask = node_count > 0
        for f_idx in self._mean_feat_indices:
            node_features[node_mask, f_idx] /= node_count[node_mask]
        for f_idx in self._max_feat_indices:
            node_features[node_mask, f_idx] = node_max[node_mask, f_idx]

        return node_features.astype(np.float32), node_mask

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Convert a grid2op observation to a PTDF-based electrical distance graph.

        Args:
            g2op_obs: The grid2op observation to convert.
        Returns:
            Dict with keys:
              NODES:      [max_nodes, x_dim] float32, aggregated bus features (zeroed for inactive).
              EDGE_INDEX: [2, max_num_edges] int64, static fully-connected directed index.
              EDGE_MASK:  [max_num_edges] bool, True where both endpoint buses are active.
              EDGES:      [max_num_edges, 1] float32, normalized L1 PTDF distance (zeroed for inactive).
              NODE_MASK:  [max_nodes] bool, True for buses with ≥1 connected element.
              GLOBAL:     [6] float32, time-based features.
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        node_features, node_mask = self._get_nodes_and_mask(g2op_obs)
        self._timings["node_features_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self._get_ptdf_cached(g2op_obs.line_status)
        D = self._cached_D
        self._timings["ptdf_ms"] = (time.perf_counter() - t0) * 1000

        global_features = self._get_global_features(g2op_obs)

        src, dst = self._edge_index_full[0], self._edge_index_full[1]
        edge_weights = D[src, dst].astype(np.float32)[:, np.newaxis]  # [max_num_edges, 1]
        edge_mask = node_mask[src] & node_mask[dst]

        result = self.normalize({
            NODES: node_features,
            EDGE_INDEX: self._edge_index_full,
            EDGE_MASK: edge_mask,
            EDGES: edge_weights,
            NODE_MASK: node_mask,
            GLOBAL: global_features,
        })
        self._timings["obs_conversion_ms"] = (time.perf_counter() - t_total) * 1000
        return result

    def normalize(self, gym_obs: dict) -> dict:
        """
        Normalize node features and edge weights using separate running statistics.

        Only active nodes (NODE_MASK=True) update the node normalizer; inactive
        node features are zeroed out after normalization.  Only active edges
        (EDGE_MASK=True) update the edge normalizer; inactive edges are zeroed.

        Args:
            gym_obs: PTDF graph observation dict with raw features.
        """
        node_features = gym_obs[NODES]   # (max_nodes, x_dim)
        node_mask = gym_obs[NODE_MASK]   # (max_nodes,)
        edge_weights = gym_obs[EDGES]    # (max_num_edges, 1)
        edge_mask = gym_obs[EDGE_MASK]   # (max_num_edges,)

        active_features = node_features[node_mask]
        if active_features.shape[0] > 0:
            self._normalizer.update(active_features)
        normalized_nodes = (node_features - self._normalizer.mean) / np.sqrt(self._normalizer.var + 1e-8)
        normalized_nodes[~node_mask] = 0.0

        active_edges = edge_weights[edge_mask]  # (n_active, 1)
        if active_edges.shape[0] > 0:
            self._edge_normalizer.update(active_edges)
        normalized_edges = (edge_weights - self._edge_normalizer.mean) / np.sqrt(self._edge_normalizer.var + 1e-8)
        normalized_edges[~edge_mask] = 0.0

        if np.any(np.isnan(normalized_nodes)) or np.any(np.isnan(normalized_edges)):
            logger.warning("NaN detected in PTDFGraphObservationConverter.normalize(); replacing with 0.")
            normalized_nodes = np.nan_to_num(normalized_nodes, nan=0.0, posinf=0.0, neginf=0.0)
            normalized_edges = np.nan_to_num(normalized_edges, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            NODES: normalized_nodes.astype(np.float32),
            EDGE_INDEX: gym_obs[EDGE_INDEX],
            EDGE_MASK: edge_mask,
            EDGES: normalized_edges.astype(np.float32),
            NODE_MASK: node_mask,
            GLOBAL: gym_obs[GLOBAL],
        }


def _compute_lodf(
    grid,
    n_line: int,
    line_status: np.ndarray,
    cached_line_status: Optional[np.ndarray],
    cached_D: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (D, new_line_status) where D = |LODF[:n_line, :n_line]|.

    Recomputes only when *line_status* differs from *cached_line_status*.
    DC PF is called directly on the C++ grid model, which is safe because
    lightsim2grid keeps separate DC and AC solver objects.

    Args:
        grid: lightsim2grid GridModel instance (backend._grid).
        n_line: number of AC transmission lines.
        line_status: boolean array [n_line] of current connectivity.
        cached_line_status: previously cached line_status (or None).
        cached_D: previously cached distance matrix (or None).
    Returns:
        (D, line_status_copy) — D is [n_line, n_line] float64, non-negative.
    Raises:
        RuntimeError: if the DC power flow diverges.
    """
    if cached_line_status is not None and np.array_equal(line_status, cached_line_status):
        return cached_D, cached_line_status
    Vinit = np.ones(grid.total_bus(), dtype=complex)
    Vdc = grid.dc_pf(Vinit, 10, 1e-8)
    if Vdc.shape[0] == 0:
        raise RuntimeError(
            "DC power flow diverged; LODF cannot be computed for current topology."
        )
    lodf = grid.get_lodf()
    lodf_sub = np.nan_to_num(lodf[:n_line, :n_line], nan=0.0, posinf=0.0, neginf=0.0)
    return np.abs(lodf_sub), line_status.copy()


class LODFGraphObservationConverter(ObservationConverter[Dict]):
    """
    LODF-based contingency coupling graph.

    Nodes represent powerlines (n_line nodes, fixed count — no padding).
    Edges are fully connected between all line pairs; the edge weight is:

        W[i, j] = |LODF[i, j]|

    where LODF[i, j] is the Line Outage Distribution Factor — the fractional
    change in flow on line i when line j is removed.  High |LODF| means the
    two lines are tightly coupled under contingency: tripping one strongly
    affects the other.

    NODE_MASK is True for connected lines (obs.line_status).
    EDGE_MASK is True only when both endpoint lines are connected.
    EDGES carries the running-normalized |LODF| value as a scalar [E, 1].

    Like PTDF, LODF depends only on network topology (not on operating point),
    so it is cached per line-status vector and recomputed only on topology changes.

    Node features per line (x_dim = 7):
      0: rho              — relative line loading
      1: p_or             — active power flow (origin side)
      2: q_or             — reactive power flow (origin side)
      3: v_or             — voltage magnitude (origin side)
      4: cos(theta_or)    — voltage angle cosine (origin side)
      5: sin(theta_or)    — voltage angle sine (origin side)
      6: line_status      — 1.0 if connected, 0.0 if disconnected
    """

    _X_DIM = 7

    def __init__(
        self,
        backend: LightSimBackend,
        g2op_obs_space: ObservationSpace,
        verbose: bool = False,
    ):
        if g2op_obs_space.n_line == 0:
            raise ValueError("LODFGraphObservationConverter requires n_line > 0.")

        self._grid = backend._grid
        self._n_line = g2op_obs_space.n_line

        n_line = self._n_line
        self._num_edges = n_line * (n_line - 1)  # directed, no self-loops

        # Static fully-connected directed edge index (no self-loops)
        idx = np.arange(n_line)
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        no_self_loop = ii != jj
        self._edge_index = np.stack([ii[no_self_loop], jj[no_self_loop]]).astype(np.int64)

        self._observation_space = Dict({
            NODES: Box(low=-np.inf, high=np.inf, shape=(n_line, self._X_DIM), dtype=np.float32),
            EDGE_INDEX: Box(low=0, high=n_line - 1, shape=(2, self._num_edges), dtype=np.int64),
            EDGE_MASK: Box(low=0, high=1, shape=(self._num_edges,), dtype=np.bool_),
            EDGES: Box(low=-np.inf, high=np.inf, shape=(self._num_edges, 1), dtype=np.float32),
            NODE_MASK: Box(low=0, high=1, shape=(n_line,), dtype=np.bool_),
            GLOBAL: Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })

        self._normalizer = RunningMeanStd(shape=(self._X_DIM,))
        self._edge_normalizer = RunningMeanStd(shape=(1,))
        self._timings: dict[str, float] = {}

        # LODF cache: _cached_D = |LODF[:n_line, :n_line]|, shares lifetime with LODF.
        self._cached_D: Optional[np.ndarray] = None
        self._cached_line_status: Optional[np.ndarray] = None

        if verbose:
            logger.info(
                f"LODFGraphObservationConverter: {n_line} nodes (powerlines), "
                f"{self._num_edges} directed edges, x_dim={self._X_DIM}."
            )

    @property
    def observation_space(self) -> Dict:
        return self._observation_space

    @property
    def num_nodes(self) -> int:
        """Number of powerline nodes (= n_line, fixed)."""
        return self._n_line

    @property
    def max_nodes(self) -> int:
        """Maximum number of nodes across all observations (= n_line)."""
        return self._n_line

    @property
    def num_edges(self) -> int:
        """Total directed edges in the fully-connected line graph."""
        return self._num_edges

    def _get_lodf_cached(self, line_status: np.ndarray) -> None:
        """
        Update _cached_D and _cached_line_status if line_status has changed.

        Delegates to the module-level :func:`_compute_lodf` helper which is
        shared with :class:`ElementLODFGraphObservationConverter`.

        Args:
            line_status: Boolean array [n_line] of current line connection state.
        """
        self._cached_D, self._cached_line_status = _compute_lodf(
            self._grid, self._n_line, line_status,
            self._cached_line_status, self._cached_D,
        )

    def _get_node_features(self, g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """
        Build the [n_line, 7] node feature matrix for the current observation.

        Features for disconnected lines (line_status == False) are zeroed out.

        Args:
            g2op_obs: Current grid2op observation.
        Returns:
            Node feature matrix [n_line, 7] float32.
        """
        line_status = g2op_obs.line_status  # dtype bool, shape (n_line,)
        theta_or = np.deg2rad(g2op_obs.theta_or).astype(np.float32)
        X = np.column_stack([
            g2op_obs.rho.astype(np.float32),
            g2op_obs.p_or.astype(np.float32),
            g2op_obs.q_or.astype(np.float32),
            g2op_obs.v_or.astype(np.float32),
            np.cos(theta_or),
            np.sin(theta_or),
            line_status.astype(np.float32),
        ])  # (n_line, 7)
        X[~line_status] = 0.0
        return X

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Convert a grid2op observation to an LODF-based contingency coupling graph.

        Args:
            g2op_obs: The grid2op observation to convert.
        Returns:
            Dict with keys:
              NODES:      [n_line, 7] float32, per-line electrical features (zeroed for disconnected).
              EDGE_INDEX: [2, n_line*(n_line-1)] int64, static fully-connected directed index.
              EDGE_MASK:  [n_line*(n_line-1)] bool, True where both endpoint lines are connected.
              EDGES:      [n_line*(n_line-1), 1] float32, normalized |LODF[i,j]| (zeroed for inactive).
              NODE_MASK:  [n_line] bool, True for connected lines.
              GLOBAL:     [6] float32, time-based features.
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        node_features = self._get_node_features(g2op_obs)
        self._timings["node_features_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self._get_lodf_cached(g2op_obs.line_status)
        self._timings["lodf_ms"] = (time.perf_counter() - t0) * 1000
        if self._cached_D is None:
            raise RuntimeError("LODF cache is uninitialized after _get_lodf_cached — DC PF likely failed.")

        node_mask = g2op_obs.line_status.astype(bool)
        global_features = self._get_global_features(g2op_obs)

        src, dst = self._edge_index[0], self._edge_index[1]
        edge_weights = self._cached_D[src, dst].astype(np.float32)[:, np.newaxis]  # [num_edges, 1]
        edge_mask = node_mask[src] & node_mask[dst]

        result = self.normalize({
            NODES: node_features,
            EDGE_INDEX: self._edge_index,
            EDGE_MASK: edge_mask,
            EDGES: edge_weights,
            NODE_MASK: node_mask,
            GLOBAL: global_features,
        })
        self._timings["obs_conversion_ms"] = (time.perf_counter() - t_total) * 1000
        return result

    def normalize(self, gym_obs: dict[str, npt.NDArray]) -> dict[str, npt.NDArray]:
        """
        Normalize node features and edge weights using separate running statistics.

        Only connected lines (NODE_MASK=True) update the node normalizer; disconnected
        line features are zeroed after normalization.  Only active edges (EDGE_MASK=True)
        update the edge normalizer; inactive edges are zeroed.

        Args:
            gym_obs: LODF graph observation dict with raw features.
        """
        node_features = gym_obs[NODES]   # (n_line, 7)
        node_mask = gym_obs[NODE_MASK]   # (n_line,)
        edge_weights = gym_obs[EDGES]    # (num_edges, 1)
        edge_mask = gym_obs[EDGE_MASK]   # (num_edges,)

        active_features = node_features[node_mask]
        if active_features.shape[0] > 0:
            self._normalizer.update(active_features)
        normalized_nodes = (node_features - self._normalizer.mean) / np.sqrt(self._normalizer.var + 1e-8)
        normalized_nodes[~node_mask] = 0.0

        active_edges = edge_weights[edge_mask]  # (n_active, 1)
        if active_edges.shape[0] > 0:
            self._edge_normalizer.update(active_edges)
        normalized_edges = (edge_weights - self._edge_normalizer.mean) / np.sqrt(self._edge_normalizer.var + 1e-8)
        normalized_edges[~edge_mask] = 0.0

        return {
            NODES: normalized_nodes.astype(np.float32),
            EDGE_INDEX: gym_obs[EDGE_INDEX],
            EDGE_MASK: edge_mask,
            EDGES: normalized_edges.astype(np.float32),
            NODE_MASK: node_mask,
            GLOBAL: gym_obs[GLOBAL],
        }

    @staticmethod
    def _get_global_features(g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """Extract time-based global features."""
        return np.array([
            g2op_obs.year, g2op_obs.month, g2op_obs.day,
            g2op_obs.hour_of_day, g2op_obs.day_of_week, g2op_obs.minute_of_hour,
        ], dtype=np.float32)


def _compute_zbus_admittance(
    grid,
    topo_vect: np.ndarray,
    cached_topo_vect: Optional[np.ndarray],
    cached_D: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (D, topo_vect_copy) where D[i,j] = 1 / (1 + |Zbus[i,j]|).

    Values lie in (0, 1]: 1 for electrically identical buses (|Zbus|=0),
    approaching 0 for electrically isolated pairs (|Zbus|→∞).  The bounded
    range prevents gradient explosion when bus-split actions create near-zero
    Zbus entries, which the old 1/(|Zbus|+eps) formula mapped to 1/eps = 1e6.

    Recomputes only when topo_vect differs from cached_topo_vect.
    No DC PF is required — get_Ybus() reads directly from network parameters.

    Args:
        grid: lightsim2grid GridModel instance (backend._grid).
        topo_vect: Integer array [dim_topo] encoding current busbar assignments.
        cached_topo_vect: Previously cached topo_vect (or None).
        cached_D: Previously cached admittance matrix (or None).
    Returns:
        (D, topo_vect_copy) — D is [2*n_sub, 2*n_sub] float in (0, 1].
    """
    if cached_topo_vect is not None and np.array_equal(topo_vect, cached_topo_vect):
        return cached_D, cached_topo_vect
    import scipy.linalg
    Ybus = grid.get_Ybus()
    Zbus = scipy.linalg.pinv(Ybus.toarray())
    D = 1.0 / (1.0 + np.abs(Zbus))
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
    return D, topo_vect.copy()


class ZbusGraphObservationConverter(PTDFGraphObservationConverter):
    """
    Zbus-based electrical admittance graph.

    Nodes represent busbars (up to 2 per substation → max_nodes = 2*n_sub).
    Edges are fully connected over all bus-slot pairs; the edge weight is the
    bus-to-bus admittance derived from the bus impedance matrix:

        W[i, j] = 1 / (|Zbus[i, j]| + ε)

    where Zbus = pinv(Ybus) (Moore-Penrose pseudoinverse of the bus admittance
    matrix).  High admittance means buses are electrically close; near-zero
    admittance means they are electrically isolated.

    NODE_MASK, EDGE_MASK, EDGES, node features, and normalization are inherited
    unchanged from PTDFGraphObservationConverter.

    Unlike PTDF/LODF, computing Ybus does not require running a DC power flow —
    it is derived directly from network topology and line parameters.  Zbus
    depends on both line connectivity (line_status) and busbar assignments
    (topo_vect), so the cache key is obs.topo_vect (which encodes both).
    """

    def __init__(
        self,
        backend: LightSimBackend,
        g2op_obs_space: ObservationSpace,
        attr_to_observe: Optional[list[str]] = None,
        verbose: bool = False,
    ):
        # Inherit full bus-level structure (obs space, edge index, normalizers,
        # _get_nodes_and_mask, normalize, _get_global_features).
        # Structure-specific cache variables are overridden below.
        super().__init__(backend, g2op_obs_space, attr_to_observe, verbose)

        # Replace PTDF cache with Zbus cache keyed on the full topology vector.
        # topo_vect encodes both line status and busbar assignments, so any
        # topology change that affects Ybus is detected.
        del self._cached_ptdf
        self._cached_D = None
        self._cached_line_status = None          # unused; kept to avoid AttributeError on parent methods
        self._cached_topo_vect: Optional[np.ndarray] = None

        if verbose:
            logger.info(
                f"ZbusGraphObservationConverter: {self._max_nodes} max nodes "
                f"({self._n_sub} subs × 2 buses), {self._max_num_edges} directed edges, "
                f"{len(self.attr_to_observe)} features per node ({', '.join(self.attr_to_observe)})."
            )

    def _get_zbus_cached(self, topo_vect: np.ndarray) -> None:
        """
        Recompute the admittance matrix 1/(|Zbus| + ε) when topology changes.

        Delegates to the module-level :func:`_compute_zbus_admittance` helper
        which is shared with :class:`SubstationZbusGraphObservationConverter`.

        Args:
            topo_vect: Integer array [dim_topo] encoding current busbar assignments.
        """
        self._cached_D, self._cached_topo_vect = _compute_zbus_admittance(
            self._grid, topo_vect,
            self._cached_topo_vect, self._cached_D,
        )

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Convert a grid2op observation to a Zbus-based electrical admittance graph.

        Args:
            g2op_obs: The grid2op observation to convert.
        Returns:
            Dict with keys:
              NODES:      [max_nodes, x_dim] float32, aggregated bus features (zeroed for inactive).
              EDGE_INDEX: [2, max_num_edges] int64, static fully-connected directed index.
              EDGE_MASK:  [max_num_edges] bool, True where both endpoint buses are active.
              EDGES:      [max_num_edges, 1] float32, normalized 1/|Zbus[i,j]| (zeroed for inactive).
              NODE_MASK:  [max_nodes] bool, True for buses with ≥1 connected element.
              GLOBAL:     [6] float32, time-based features.
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        node_features, node_mask = self._get_nodes_and_mask(g2op_obs)
        self._timings["node_features_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self._get_zbus_cached(g2op_obs.topo_vect)
        self._timings["zbus_ms"] = (time.perf_counter() - t0) * 1000

        global_features = self._get_global_features(g2op_obs)

        src, dst = self._edge_index_full[0], self._edge_index_full[1]
        edge_weights = self._cached_D[src, dst].astype(np.float32)[:, np.newaxis]  # [max_num_edges, 1]
        edge_mask = node_mask[src] & node_mask[dst]

        result = self.normalize({
            NODES: node_features,
            EDGE_INDEX: self._edge_index_full,
            EDGE_MASK: edge_mask,
            EDGES: edge_weights,
            NODE_MASK: node_mask,
            GLOBAL: global_features,
        })
        self._timings["obs_conversion_ms"] = (time.perf_counter() - t_total) * 1000
        return result


class SubstationPTDFGraphObservationConverter(SubstationGraphObservationConverter):
    """
    Substation-level graph extended with PTDF electrical-distance edges.

    Inherits the sparse powerline topology from
    :class:`SubstationGraphObservationConverter`:

    * Type 0 — powerline edges (bus_or[l] ↔ bus_ex[l], one per connected line)

    Adds a second edge type:

    * Type 1 — fully-connected directed PTDF-distance edges between all bus slots

    Edge weight ``EDGES[e, 0]`` for a type-1 edge (i → j) is the L1 PTDF distance:

        D[i, j] = sum_l |PTDF[l, i] - PTDF[l, j]|

    Only active edges (EDGE_MASK=True) carry a non-zero attribute; inactive
    type-0 edges and padding positions are zeroed.

    PTDF depends only on network topology (not on the operating point), so it
    is cached per ``line_status`` and recomputed only on topology changes using
    the shared :func:`_compute_ptdf_distance` module-level helper.

    Slot convention: this converter inherits the Substation slot ordering
    (slot = 2*sub_id + (bus - 1)), while lightsim2grid returns PTDF in the
    PTDF/Ybus convention (slot = (bus - 1)*n_sub + sub_id).  A one-time
    permutation ``_sub_to_ptdf`` is applied when caching D so that all
    subsequent indexing uses Substation slot order consistently.
    ``_sub_to_ptdf[s] = s//2 + (s%2)*n_sub`` maps each Substation slot s to
    its corresponding PTDF slot, so
    ``D_sub = D_ptdf[np.ix_(_sub_to_ptdf, _sub_to_ptdf)]`` gives
    ``D_sub[i, j] = D_ptdf[ptdf(i), ptdf(j)]`` — the correct reindexing.

    ``EDGE_TYPE`` is declared as ``Box(low=0, high=1, …)``; the downstream
    :class:`~rl.gnn_ppo_model.GNNModel` derives
    ``num_edge_types = high + 1 = 2`` automatically.

    Observation keys
    ----------------
    NODES      : [max_nodes, x_dim]   — aggregated bus features (zeroed for inactive)
    EDGE_INDEX : [2, max_num_edges]   — padded connectivity
    EDGE_MASK  : [max_num_edges]      — True for active edges
    EDGES      : [max_num_edges, 1]   — 0.0 for type-0, PTDF distance for type-1
    EDGE_TYPE  : [max_num_edges]      — 0/1 per edge, 0 for padding
    NODE_MASK  : [max_nodes]          — True for active bus slots
    GLOBAL     : [6]                  — time-based global features

    :param backend: LightSimBackend instance.
    :param g2op_obs_space: grid2op observation space.
    :param attr_to_observe: node feature names (defaults to ``_DEFAULT_NODE_FEATURES``).
    :param verbose: log converter dimensions on construction.
    """

    def __init__(
        self,
        backend: LightSimBackend,
        g2op_obs_space: ObservationSpace,
        attr_to_observe: Optional[list[str]] = None,
        verbose: bool = False,
    ):
        super().__init__(g2op_obs_space, attr_to_observe, verbose)
        self._grid = backend._grid
        n_sub = self._n_sub

        # Permutation: PTDF slot (busbar-1)*n_sub+sub_id  →  Substation slot 2*sub_id+(busbar-1)
        s = np.arange(2 * n_sub)
        # Maps Substation slot s → PTDF slot: s//2 + (s%2)*n_sub.
        # Used as D_sub = D_ptdf[ix_(sub_to_ptdf, sub_to_ptdf)] so that
        # D_sub[i, j] = D_ptdf[ptdf(i), ptdf(j)] (correct reindexing).
        self._sub_to_ptdf: npt.NDArray[np.int64] = (s // 2 + (s % 2) * n_sub).astype(np.int64)

        # Fully-connected directed edge index over all bus slots (no self-loops).
        self._n_phys_edges: int = self._max_nodes * (self._max_nodes - 1)
        idx = np.arange(self._max_nodes)
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        no_self_loop = ii != jj
        self._phys_edge_index: npt.NDArray[np.int64] = np.stack(
            [ii[no_self_loop], jj[no_self_loop]]
        ).astype(np.int64)  # [2, n_phys_edges]

        # Topology edges from parent: 2 * n_line (bidirectional powerline edges).
        # Expand to include physics edges.
        self._max_num_edges = self._max_num_edges + self._n_phys_edges

        x_dim = len(self.attr_to_observe)
        self._observation_space = Dict({
            NODES: Box(low=-np.inf, high=np.inf,
                       shape=(self._max_nodes, x_dim), dtype=np.float32),
            EDGE_INDEX: Box(low=0, high=self._max_nodes - 1,
                            shape=(2, self._max_num_edges), dtype=np.int64),
            EDGE_MASK: Box(low=0, high=1,
                           shape=(self._max_num_edges,), dtype=np.bool_),
            EDGES: Box(low=-np.inf, high=np.inf,
                       shape=(self._max_num_edges, 1), dtype=np.float32),
            EDGE_TYPE: Box(low=0, high=1,
                           shape=(self._max_num_edges,), dtype=np.int64),
            NODE_MASK: Box(low=0, high=1,
                           shape=(self._max_nodes,), dtype=np.bool_),
            GLOBAL: Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })

        self._edge_normalizer = RunningMeanStd(shape=(1,))

        # PTDF cache (in Substation slot order after permutation).
        self._cached_ptdf: Optional[np.ndarray] = None
        self._cached_D: Optional[np.ndarray] = None
        self._cached_line_status: Optional[np.ndarray] = None

        if verbose:
            logger.info(
                f"SubstationPTDFGraphObservationConverter: {self._max_nodes} max nodes "
                f"({n_sub} subs × 2 buses), {self._max_num_edges} edges "
                f"({self._max_num_edges - self._n_phys_edges} topology + "
                f"{self._n_phys_edges} PTDF), x_dim={x_dim}."
            )

    def _get_ptdf_cached(self, line_status: np.ndarray) -> None:
        """
        Update _cached_D (in Substation slot order) when line_status changes.

        Delegates to :func:`_compute_ptdf_distance`, then applies the
        PTDF-to-Substation slot permutation so subsequent indexing is
        consistent with the node ordering used by this converter.

        Args:
            line_status: Boolean array [n_line] of current line connection state.
        """
        prev_D = self._cached_D
        self._cached_ptdf, D_ptdf, self._cached_line_status = _compute_ptdf_distance(
            self._grid, line_status,
            self._cached_line_status, self._cached_ptdf, self._cached_D,
        )
        if D_ptdf is not prev_D:
            # Cache was stale — apply slot permutation before storing.
            self._cached_D = D_ptdf[np.ix_(self._sub_to_ptdf, self._sub_to_ptdf)]

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Convert a grid2op observation to a substation+PTDF graph observation.

        Args:
            g2op_obs: Current grid2op observation.
        Returns:
            Dict with keys NODES, EDGE_INDEX, EDGE_MASK, EDGES, EDGE_TYPE,
            NODE_MASK, GLOBAL.
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        node_features, node_mask = self._get_nodes_and_mask(g2op_obs)
        self._timings["node_features_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        edge_index_topo = self._get_edge_index(g2op_obs)   # [2, n_topo_edges]
        self._timings["edge_index_ms"] = (time.perf_counter() - t0) * 1000
        n_topo = edge_index_topo.shape[1]

        t0 = time.perf_counter()
        self._get_ptdf_cached(g2op_obs.line_status)
        self._timings["ptdf_ms"] = (time.perf_counter() - t0) * 1000

        global_features = self._get_global_features(g2op_obs)

        src, dst = self._phys_edge_index[0], self._phys_edge_index[1]
        phys_weights = self._cached_D[src, dst].astype(np.float32)  # [n_phys_edges]
        phys_mask = node_mask[src] & node_mask[dst]                  # [n_phys_edges]

        n_total = n_topo + self._n_phys_edges
        E = self._max_num_edges

        edge_index_padded = np.zeros((2, E), dtype=np.int64)
        edge_index_padded[:, :n_topo] = edge_index_topo
        edge_index_padded[:, n_topo:n_total] = self._phys_edge_index

        edge_mask = np.zeros(E, dtype=bool)
        edge_mask[:n_topo] = True
        edge_mask[n_topo:n_total] = phys_mask

        edge_type = np.zeros(E, dtype=np.int64)
        edge_type[n_topo:n_total] = 1

        edges_padded = np.zeros((E, 1), dtype=np.float32)
        edges_padded[n_topo:n_total, 0] = phys_weights

        result = self.normalize({
            NODES: node_features,
            EDGE_INDEX: edge_index_padded,
            EDGE_MASK: edge_mask,
            EDGES: edges_padded,
            EDGE_TYPE: edge_type,
            NODE_MASK: node_mask,
            GLOBAL: global_features,
        })
        self._timings["obs_conversion_ms"] = (time.perf_counter() - t_total) * 1000
        return result

    def normalize(self, gym_obs: dict) -> dict:
        """
        Normalize node features and PTDF edge weights with separate running statistics.

        Node normalization is inherited from the parent (active nodes only).
        Type-1 (PTDF) edge attributes are normalized using a separate running
        normalizer; only active type-1 edges update the statistics.
        Type-0 (topology) edge attributes remain 0.0.

        Args:
            gym_obs: Raw observation dict from to_gym.
        """
        base = super().normalize(gym_obs)   # handles NODES, passes rest through

        edges = gym_obs[EDGES]              # (E, 1)
        edge_mask = gym_obs[EDGE_MASK]      # (E,)
        edge_type = gym_obs[EDGE_TYPE]      # (E,)

        phys_active = edge_mask & (edge_type == 1)
        active_vals = edges[phys_active]    # (n_active, 1)
        if active_vals.shape[0] > 0:
            self._edge_normalizer.update(active_vals)
        normalized_edges = np.zeros_like(edges)
        if phys_active.any():
            normalized_edges[phys_active] = (
                (active_vals - self._edge_normalizer.mean)
                / np.sqrt(self._edge_normalizer.var + 1e-8)
            )

        if np.any(np.isnan(normalized_edges)):
            logger.warning("NaN detected in SubstationPTDFGraphObservationConverter.normalize(); replacing with 0.")
            normalized_edges = np.nan_to_num(normalized_edges, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            **base,
            EDGES: normalized_edges.astype(np.float32),
            EDGE_TYPE: edge_type,
        }


class SubstationZbusGraphObservationConverter(SubstationGraphObservationConverter):
    """
    Substation-level graph extended with Zbus electrical-admittance edges.

    Inherits the sparse powerline topology from
    :class:`SubstationGraphObservationConverter`:

    * Type 0 — powerline edges (bus_or[l] ↔ bus_ex[l], one per connected line)

    Adds a second edge type:

    * Type 1 — fully-connected directed Zbus-admittance edges between all bus slots

    Edge weight ``EDGES[e, 0]`` for a type-1 edge (i → j) is:

        D[i, j] = 1 / (|Zbus[i, j]| + ε)

    where ``Zbus = pinv(Ybus)`` (Moore-Penrose pseudoinverse of the bus admittance
    matrix).  High admittance means the two buses are electrically close.

    Unlike PTDF, Zbus depends on both line connectivity and busbar assignments,
    so it is cached per ``topo_vect`` using the shared
    :func:`_compute_zbus_admittance` module-level helper.

    The same PTDF-to-Substation slot permutation as
    :class:`SubstationPTDFGraphObservationConverter` is applied when caching
    so that all indexing uses Substation slot order consistently.

    ``EDGE_TYPE`` is declared as ``Box(low=0, high=1, …)``; the downstream
    :class:`~rl.gnn_ppo_model.GNNModel` derives
    ``num_edge_types = high + 1 = 2`` automatically.

    :param backend: LightSimBackend instance.
    :param g2op_obs_space: grid2op observation space.
    :param attr_to_observe: node feature names (defaults to ``_DEFAULT_NODE_FEATURES``).
    :param verbose: log converter dimensions on construction.
    """


    def __init__(
        self,
        backend: LightSimBackend,
        g2op_obs_space: ObservationSpace,
        attr_to_observe: Optional[list[str]] = None,
        verbose: bool = False,
    ):
        super().__init__(g2op_obs_space, attr_to_observe, verbose)
        self._grid = backend._grid
        n_sub = self._n_sub

        # Permutation: PTDF slot (busbar-1)*n_sub+sub_id  →  Substation slot 2*sub_id+(busbar-1)
        s = np.arange(2 * n_sub)
        # Maps Substation slot s → PTDF slot: s//2 + (s%2)*n_sub.
        # Used as D_sub = D_ptdf[ix_(sub_to_ptdf, sub_to_ptdf)] so that
        # D_sub[i, j] = D_ptdf[ptdf(i), ptdf(j)] (correct reindexing).
        self._sub_to_ptdf: npt.NDArray[np.int64] = (s // 2 + (s % 2) * n_sub).astype(np.int64)

        # Fully-connected directed edge index (no self-loops).
        self._n_phys_edges: int = self._max_nodes * (self._max_nodes - 1)
        idx = np.arange(self._max_nodes)
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        no_self_loop = ii != jj
        self._phys_edge_index: npt.NDArray[np.int64] = np.stack(
            [ii[no_self_loop], jj[no_self_loop]]
        ).astype(np.int64)

        self._max_num_edges = self._max_num_edges + self._n_phys_edges

        x_dim = len(self.attr_to_observe)
        self._observation_space = Dict({
            NODES: Box(low=-np.inf, high=np.inf,
                       shape=(self._max_nodes, x_dim), dtype=np.float32),
            EDGE_INDEX: Box(low=0, high=self._max_nodes - 1,
                            shape=(2, self._max_num_edges), dtype=np.int64),
            EDGE_MASK: Box(low=0, high=1,
                           shape=(self._max_num_edges,), dtype=np.bool_),
            EDGES: Box(low=-np.inf, high=np.inf,
                       shape=(self._max_num_edges, 1), dtype=np.float32),
            EDGE_TYPE: Box(low=0, high=1,
                           shape=(self._max_num_edges,), dtype=np.int64),
            NODE_MASK: Box(low=0, high=1,
                           shape=(self._max_nodes,), dtype=np.bool_),
            GLOBAL: Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })

        self._edge_normalizer = RunningMeanStd(shape=(1,))

        # Zbus cache (in Substation slot order after permutation).
        self._cached_D: Optional[np.ndarray] = None
        self._cached_topo_vect: Optional[np.ndarray] = None

        if verbose:
            logger.info(
                f"SubstationZbusGraphObservationConverter: {self._max_nodes} max nodes "
                f"({n_sub} subs × 2 buses), {self._max_num_edges} edges "
                f"({self._max_num_edges - self._n_phys_edges} topology + "
                f"{self._n_phys_edges} Zbus), x_dim={x_dim}."
            )

    def _get_zbus_cached(self, topo_vect: np.ndarray) -> None:
        """
        Update _cached_D (in Substation slot order) when topo_vect changes.

        Delegates to :func:`_compute_zbus_admittance`, then applies the
        PTDF-to-Substation slot permutation.

        Args:
            topo_vect: Integer array [dim_topo] encoding current busbar assignments.
        """
        prev_D = self._cached_D
        D_ptdf, self._cached_topo_vect = _compute_zbus_admittance(
            self._grid, topo_vect,
            self._cached_topo_vect, self._cached_D,
        )
        if D_ptdf is not prev_D:
            self._cached_D = D_ptdf[np.ix_(self._sub_to_ptdf, self._sub_to_ptdf)]

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Convert a grid2op observation to a substation+Zbus graph observation.

        Args:
            g2op_obs: Current grid2op observation.
        Returns:
            Dict with keys NODES, EDGE_INDEX, EDGE_MASK, EDGES, EDGE_TYPE,
            NODE_MASK, GLOBAL.
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        node_features, node_mask = self._get_nodes_and_mask(g2op_obs)
        self._timings["node_features_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        edge_index_topo = self._get_edge_index(g2op_obs)
        self._timings["edge_index_ms"] = (time.perf_counter() - t0) * 1000
        n_topo = edge_index_topo.shape[1]

        t0 = time.perf_counter()
        self._get_zbus_cached(g2op_obs.topo_vect)
        self._timings["zbus_ms"] = (time.perf_counter() - t0) * 1000

        global_features = self._get_global_features(g2op_obs)

        src, dst = self._phys_edge_index[0], self._phys_edge_index[1]
        phys_weights = self._cached_D[src, dst].astype(np.float32)
        phys_mask = node_mask[src] & node_mask[dst]

        n_total = n_topo + self._n_phys_edges
        E = self._max_num_edges

        edge_index_padded = np.zeros((2, E), dtype=np.int64)
        edge_index_padded[:, :n_topo] = edge_index_topo
        edge_index_padded[:, n_topo:n_total] = self._phys_edge_index

        edge_mask = np.zeros(E, dtype=bool)
        edge_mask[:n_topo] = True
        edge_mask[n_topo:n_total] = phys_mask

        edge_type = np.zeros(E, dtype=np.int64)
        edge_type[n_topo:n_total] = 1

        edges_padded = np.zeros((E, 1), dtype=np.float32)
        edges_padded[n_topo:n_total, 0] = phys_weights

        result = self.normalize({
            NODES: node_features,
            EDGE_INDEX: edge_index_padded,
            EDGE_MASK: edge_mask,
            EDGES: edges_padded,
            EDGE_TYPE: edge_type,
            NODE_MASK: node_mask,
            GLOBAL: global_features,
        })
        self._timings["obs_conversion_ms"] = (time.perf_counter() - t_total) * 1000
        return result

    def normalize(self, gym_obs: dict) -> dict:
        """
        Normalize node features and Zbus edge weights with separate running statistics.

        Args:
            gym_obs: Raw observation dict from to_gym.
        """
        base = super().normalize(gym_obs)

        edges = gym_obs[EDGES]
        edge_mask = gym_obs[EDGE_MASK]
        edge_type = gym_obs[EDGE_TYPE]

        phys_active = edge_mask & (edge_type == 1)
        active_vals = edges[phys_active]
        if active_vals.shape[0] > 0:
            self._edge_normalizer.update(active_vals)
        normalized_edges = np.zeros_like(edges)
        if phys_active.any():
            normalized_edges[phys_active] = (
                (active_vals - self._edge_normalizer.mean)
                / np.sqrt(self._edge_normalizer.var + 1e-8)
            )

        if np.any(np.isnan(normalized_edges)):
            logger.warning("NaN detected in SubstationZbusGraphObservationConverter.normalize(); replacing with 0.")
            normalized_edges = np.nan_to_num(normalized_edges, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            **base,
            EDGES: normalized_edges.astype(np.float32),
            EDGE_TYPE: edge_type,
        }


# --- Factory ---
def make_observation_converter(gym_env: GymEnv, env_config: dict) -> ObservationConverter:
    """Construct the appropriate ObservationConverter from env_config."""
    mode = env_config.get("observation_space", "FlatSpace")
    if mode == "GraphObsSpace" or mode == "BusConnectivityGraphObsSpace":
        return GraphObservationConverter(
            g2op_obs_space=gym_env.init_env.observation_space,
            attr_to_observe=env_config.get("attr_to_observe"),
            verbose=env_config.get("verbose", False),
        )
    elif mode == "HeterogeneousGraphObsSpace":
        return HeterogeneousGraphObservationConverter(
            g2op_obs_space=gym_env.init_env.observation_space,
            attr_to_observe=env_config.get("attr_to_observe"),
            verbose=env_config.get("verbose", False),
        )
    elif mode == "SubstationGraphObsSpace":
        return SubstationGraphObservationConverter(
            g2op_obs_space=gym_env.init_env.observation_space,
            attr_to_observe=env_config.get("attr_to_observe"),
            verbose=env_config.get("verbose", False),
        )
    elif mode == "ElementGraphObsSpace":
        return ElementGraphObservationConverter(
            g2op_obs_space=gym_env.init_env.observation_space,
            verbose=env_config.get("verbose", False),
        )
    elif mode == "PTDFGraphObsSpace":
        return PTDFGraphObservationConverter(
            backend=gym_env.init_env.backend,
            g2op_obs_space=gym_env.init_env.observation_space,
            attr_to_observe=env_config.get("attr_to_observe"),
            verbose=env_config.get("verbose", False),
        )
    elif mode == "LODFGraphObsSpace":
        return LODFGraphObservationConverter(
            backend=gym_env.init_env.backend,
            g2op_obs_space=gym_env.init_env.observation_space,
            verbose=env_config.get("verbose", False),
        )
    elif mode == "ElementLODFGraphObsSpace":
        return ElementLODFGraphObservationConverter(
            backend=gym_env.init_env.backend,
            g2op_obs_space=gym_env.init_env.observation_space,
            verbose=env_config.get("verbose", False),
        )
    elif mode == "ZbusGraphObsSpace":
        return ZbusGraphObservationConverter(
            backend=gym_env.init_env.backend,
            g2op_obs_space=gym_env.init_env.observation_space,
            attr_to_observe=env_config.get("attr_to_observe"),
            verbose=env_config.get("verbose", False),
        )
    elif mode == "SubstationPTDFGraphObsSpace":
        return SubstationPTDFGraphObservationConverter(
            backend=gym_env.init_env.backend,
            g2op_obs_space=gym_env.init_env.observation_space,
            attr_to_observe=env_config.get("attr_to_observe"),
            verbose=env_config.get("verbose", False),
        )
    elif mode == "SubstationZbusGraphObsSpace":
        return SubstationZbusGraphObservationConverter(
            backend=gym_env.init_env.backend,
            g2op_obs_space=gym_env.init_env.observation_space,
            attr_to_observe=env_config.get("attr_to_observe"),
            verbose=env_config.get("verbose", False),
        )
    elif mode == "FlatObsSpace":
        return FlatObservationConverter(
            gym_env=gym_env,
            attr_to_observe=get_attr_list(env_config.get("g2op_input", ["p_i", "p_l", "q_i", "q_l", "a", "v_i", "v_l", "theta_i", "theta_l","r", "t"])),
            verbose=env_config.get("verbose", False),
        )
    else:
        raise ValueError(f"Unknown observation space type: {mode}")

if __name__ == "__main__":
    import grid2op
    env = grid2op.make("l2rpn_wcci_2022")
    graph = GraphObservationConverter(env.observation_space)
    print(graph.num_nodes)
