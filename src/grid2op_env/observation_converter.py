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
NODE_SUBSTATIONS = "node_substations"  # keys the substation(s) of every node [N, 2], -1 = none (readout per substation)
FLAT_FEATURES = "flat_features"  # keys the node features plus bus per node of FlatGraphObservationConverter [N, x_dim]


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

# Features read from get_forecast_arrays(), which is only called when one of them is observed.
_FORECAST_NODE_FEATURES = ("active_power_forecast", "reactive_power_forecast")

# Node features of CompleteGraphObservationConverter besides forecasts and the node-type one-hot.
# Voltage angles are left out: obs.to_vect() of l2rpn_wcci_2020 does not contain them, so they
# cannot be restored from imitation data.
_LINE_STATE_FEATURES = [
    "rho",
    "timestep_overflow",
    "time_before_cooldown_line",
    "time_next_maintenance",
    "duration_next_maintenance",
]
_COMPLETE_STATE_NODE_FEATURES = ["active_power", "reactive_power", "voltage", *_LINE_STATE_FEATURES, "time_before_cooldown_sub"]
# Node types in node order; each gets a one-hot column `node_type_<type>`.
_NODE_TYPES = ("line_or", "line_ex", "gen", "load", "storage")
# Node types of ElementGraphObservationConverter in node order (ground/bus1/bus2 repeat per substation).
_ELEMENT_NODE_TYPES = ("gen", "load", "line", "storage", "ground", "bus1", "bus2")
# Per-bus net injection of generators, loads and storage, read by LODF line nodes at both line ends.
_INJECTION_FEATURES = ["active_power", "reactive_power"]
# time_next_maintenance is clipped to this many steps (one day of 5-minute steps).
MAINTENANCE_HORIZON_STEPS = 288

# How bus-level converters merge the features of the elements on a bus. Features not listed are
# summed: powers and forecasts (net injection of the bus) and node-type columns (element counts).
_BUS_AGGREGATION: dict[str, str] = {
    "voltage": "mean",
    "voltage_angle_sin": "mean",
    "voltage_angle_cos": "mean",
    "time_before_cooldown_sub": "mean",
    "rho": "max",
    "timestep_overflow": "max",
    "time_before_cooldown_line": "max",
    "duration_next_maintenance": "max",
    "time_next_maintenance": "min",
}


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


@dataclass
class _Forecasts:
    """Next-step forecasts in the generator sign convention (loads negative)."""

    gen_p: npt.NDArray[np.float32]  # [n_gen]
    gen_q: npt.NDArray[np.float32]  # [n_gen]; grid2op forecasts no gen q, this holds the voltage setpoint
    load_p: npt.NDArray[np.float32]  # [n_load]
    load_q: npt.NDArray[np.float32]  # [n_load]


def zero_forecasts(dims: _GridDimensions) -> _Forecasts:
    """All-zero forecasts, for converters that observe none."""
    return _Forecasts(*(np.zeros(n, dtype=np.float32) for n in (dims.n_gen, dims.n_gen, dims.n_load, dims.n_load)))


def next_step_forecasts(g2op_obs: BaseObservation, dims: _GridDimensions) -> _Forecasts:
    """Forecasts of the next step (index 1 of `get_forecast_arrays`); zeros once the episode is done.

    Args:
        g2op_obs: Current observation.
        dims: Grid dimensions (for the zero arrays).

    Returns:
        Generator p and voltage setpoint, load p and q (negated).
    """
    if g2op_obs._is_done:
        return zero_forecasts(dims)
    load_p, load_q, prod_p, prod_v, _ = g2op_obs.get_forecast_arrays()
    return _Forecasts(
        gen_p=prod_p[1].astype(np.float32),
        gen_q=prod_v[1].astype(np.float32),
        load_p=-load_p[1].astype(np.float32),
        load_q=-load_q[1].astype(np.float32),
    )


def aggregate_to_buses(
    features: dict[str, npt.NDArray],
    names: list[str],
    slots: npt.NDArray[np.int64],
    n_slots: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Merge per-element features into bus nodes, per feature by `_BUS_AGGREGATION` (default: sum).

    Args:
        features: Feature name -> value per element [n_elements] (connected elements only).
        names: Features to merge, in column order.
        slots: Bus slot of every element [n_elements].
        n_slots: Number of bus slots.

    Returns:
        node_features [n_slots, len(names)] (0 on empty slots) and node_mask [n_slots] (slots with an element).
    """
    count = np.bincount(slots, minlength=n_slots)
    node_mask = count > 0
    node_features = np.zeros((n_slots, len(names)), dtype=np.float64)
    for col, name in enumerate(names):
        values = np.asarray(features[name], dtype=np.float64)
        rule = _BUS_AGGREGATION.get(name, "sum")
        if rule in ("sum", "mean"):
            np.add.at(node_features[:, col], slots, values)
            if rule == "mean":
                node_features[node_mask, col] /= count[node_mask]
        else:
            merged = np.full(n_slots, -np.inf if rule == "max" else np.inf)
            (np.maximum if rule == "max" else np.minimum).at(merged, slots, values)
            node_features[node_mask, col] = merged[node_mask]
    return node_features.astype(np.float32), node_mask



class ObservationConverter(ABC, Generic[T]):
    """
    Abstract base for observation converters. Each converter:
    - Exposes a gymnasium observation space for the RL agent via `observation_space`
    - Converts a grid2op observation to that space via `to_gym`
    - Normalizes the resulting observation via `normalize`
    """

    # True if `to_gym` also reads the backend's internal grid model (PTDF/LODF/Zbus), which only
    # matches the observation in the env that produced it. To convert stored observations, first
    # call `sync_backend_to_observation`.
    needs_backend_state: bool = False

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
        self._use_forecasts = any(name in _FORECAST_NODE_FEATURES for name in attr_to_observe)
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
        self._sub_ids = sub_ids
        self._node_type_features = self._build_node_type_features()
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

    @property
    def node_substations(self) -> npt.NDArray[np.int64]:
        """Substation of every node [num_nodes, 2]; second column -1 (each node is in one substation)."""
        return np.stack([self._sub_ids, np.full_like(self._sub_ids, -1)], axis=1).astype(np.int64)

    @staticmethod
    def node_buses(g2op_obs: BaseObservation) -> npt.NDArray[np.int64]:
        """Bus of every node [num_nodes] in node order [line_or | line_ex | gen | load | storage]; -1 = disconnected."""
        return np.concatenate([
            g2op_obs.line_or_bus, g2op_obs.line_ex_bus, g2op_obs.gen_bus, g2op_obs.load_bus, g2op_obs.storage_bus,
        ]).astype(np.int64)

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

    def _build_node_type_features(self) -> dict[str, npt.NDArray[np.float32]]:
        """Static one-hot columns: `node_type_<type>` -> [num_nodes] (1 on nodes of that type)."""
        dims = self._dims
        counts = [dims.n_line, dims.n_line, dims.n_gen, dims.n_load, dims.n_storage]
        node_type = np.repeat(np.arange(len(_NODE_TYPES)), counts)
        return {f"node_type_{name}": (node_type == i).astype(np.float32) for i, name in enumerate(_NODE_TYPES)}

    def _on_line_ends(self, values: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """Per-line values [n_line] as a node column [num_nodes]: on both line ends, 0 elsewhere."""
        values = np.asarray(values, dtype=np.float32)
        rest = np.zeros(self._num_nodes - 2 * self._dims.n_line, dtype=np.float32)
        return np.concatenate([values, values, rest])

    def _compute_all_node_features(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray[np.float32]]:
        """
        Compute all supported node features, each as a flat array of length
        num_nodes in order [line_or | line_ex | gen | load | storage].

        Line features (`rho`, `timestep_overflow`, ...) sit on both line ends, `time_before_cooldown_sub`
        on every node of the substation, `node_type_<type>` is a static one-hot.
        """
        dims = self._dims
        zeros_line = np.zeros(dims.n_line, dtype=np.float32)
        zeros_storage = np.zeros(dims.n_storage, dtype=np.float32)

        t0 = time.perf_counter()
        fc = next_step_forecasts(g2op_obs, dims) if self._use_forecasts else zero_forecasts(dims)
        self._timings["forecast_ms"] = (time.perf_counter() - t0) * 1000

        return {
            "active_power_forecast": np.concatenate([zeros_line, zeros_line, fc.gen_p, fc.load_p, zeros_storage]),
            # grid2op forecasts no gen q; the generator slot holds the voltage setpoint forecast.
            "reactive_power_forecast": np.concatenate([zeros_line, zeros_line, fc.gen_q, fc.load_q, zeros_storage]),
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
            # Line-level quantities sit on both line ends; non-line nodes get 0
            "rho": self._on_line_ends(g2op_obs.rho),
            "timestep_overflow": self._on_line_ends(g2op_obs.timestep_overflow),
            "time_before_cooldown_line": self._on_line_ends(g2op_obs.time_before_cooldown_line),
            "time_next_maintenance": self._on_line_ends(scale_time_next_maintenance(g2op_obs.time_next_maintenance)),
            "duration_next_maintenance": self._on_line_ends(g2op_obs.duration_next_maintenance),
            "time_before_cooldown_sub": g2op_obs.time_before_cooldown_sub[self._sub_ids].astype(np.float32),
            **self._node_type_features,
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
        bus_ids = self.node_buses(g2op_obs)

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


def complete_node_features(use_forecasts: bool, has_storage: bool) -> list[str]:
    """Node features of CompleteGraphObservationConverter, in column order.

    Args:
        use_forecasts: Include the forecast features first.
        has_storage: Include the `node_type_storage` column (only grids with storage units).

    Returns:
        Feature names: forecasts, `_COMPLETE_STATE_NODE_FEATURES`, then one `node_type_<type>` per node type.
    """
    forecasts = list(_FORECAST_NODE_FEATURES) if use_forecasts else []
    types = [t for t in _NODE_TYPES if t != "storage" or has_storage]
    return forecasts + _COMPLETE_STATE_NODE_FEATURES + [f"node_type_{t}" for t in types]


def element_node_features(use_forecasts: bool, has_storage: bool) -> list[str]:
    """Node features of ElementGraphObservationConverter, in column order.

    The information of `complete_node_features` on the element graph: one node per line holds the
    state of both line ends (`*_ex` columns), the ground/bus nodes hold the substation cooldown.

    Args:
        use_forecasts: Include the forecast features.
        has_storage: Include the `node_type_storage` column (only grids with storage units).

    Returns:
        Feature names: element state, line extremity state, forecasts, line state, substation
        cooldown, then one `node_type_<type>` per node type.
    """
    forecasts = list(_FORECAST_NODE_FEATURES) if use_forecasts else []
    types = [t for t in _ELEMENT_NODE_TYPES if t != "storage" or has_storage]
    return (["active_power", "reactive_power", "voltage", "active_power_ex", "reactive_power_ex", "voltage_ex"]
            + forecasts + _LINE_STATE_FEATURES + ["time_before_cooldown_sub"] + [f"node_type_{t}" for t in types])


def lodf_node_features(use_forecasts: bool) -> list[str]:
    """Node (= powerline) features of LODFGraphObservationConverter, in column order.

    Args:
        use_forecasts: Include the forecasts of the injections at the line ends.

    Returns:
        Feature names: line state, `line_status`, then per line end (`_or`, `_ex`) the substation
        cooldown and the net injection of the bus the end is connected to (`injection_*`).
    """
    injections = _INJECTION_FEATURES + (list(_FORECAST_NODE_FEATURES) if use_forecasts else [])
    end = ["time_before_cooldown_sub"] + [f"injection_{name}" for name in injections]
    line = ["active_power_or", "reactive_power_or", "voltage_or", "active_power_ex", "reactive_power_ex",
            "voltage_ex", *_LINE_STATE_FEATURES, "line_status"]
    return line + [f"{name}_{side}" for side in ("or", "ex") for name in end]


def scale_time_next_maintenance(steps: npt.ArrayLike, horizon: int = MAINTENANCE_HORIZON_STEPS) -> npt.NDArray[np.float32]:
    """Steps until the next maintenance, clipped to `horizon` and scaled to [0, 1].

    Args:
        steps: grid2op `time_next_maintenance` per line; -1 means none planned, 0 means ongoing.

    Returns:
        min(steps, horizon) / horizon per line; -1 maps to 1 (not within the horizon).
    """
    steps = np.asarray(steps, dtype=np.float32)
    return np.where(steps < 0, 1.0, np.minimum(steps, horizon) / horizon).astype(np.float32)


class CompleteGraphObservationConverter(GraphObservationConverter):
    """GraphObservationConverter with the operational state of lines and substations.

    Same graph (nodes, edges, global features) as the parent. Node features differ:
      - no voltage angles (not restorable from `obs.to_vect()` of imitation data)
      - line features on both line-end nodes, 0 elsewhere: `timestep_overflow`,
        `time_before_cooldown_line`, `time_next_maintenance` (see `scale_time_next_maintenance`),
        `duration_next_maintenance`
      - `time_before_cooldown_sub` on every node of the substation
      - a node-type one-hot (line origin, line extremity, generator, load, storage if the grid has any)

    The feature set is fixed (`complete_node_features`); `use_forecasts` only adds or drops the
    forecast columns. Features are normalized like the parent's, by a running mean/variance.
    The same features are available to every converter built on GraphObservationConverter via
    `attr_to_observe` (obs_space config `node_features: complete`).
    """

    def __init__(self, g2op_obs_space: ObservationSpace, use_forecasts: bool = True, verbose: bool = False):
        """Args:
            g2op_obs_space: Observation space of the grid2op env.
            use_forecasts: Include the forecast node features.
            verbose: Log the graph size and features.
        """
        features = complete_node_features(use_forecasts, has_storage=g2op_obs_space.n_storage > 0)
        super().__init__(g2op_obs_space, attr_to_observe=features, verbose=verbose)


class EdgelessGraphObservationConverter(GraphObservationConverter):
    """GraphObservationConverter without edges: same nodes and node features, no message passing.

    Control of the graph comparison: a GNN on it can only combine node features in its readout,
    so the gap to a graph with edges measures what the graph structure adds.
    """

    def _get_edge_index(self, g2op_obs: BaseObservation) -> npt.NDArray[np.int64]:
        """No edges: [2, 0]."""
        return np.zeros((2, 0), dtype=np.int64)


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
    merged over all elements assigned to each bus by `aggregate_to_buses`
    (`_BUS_AGGREGATION`: e.g. powers summed, voltage averaged, rho maximum).

    Edges are one bidirectional pair per connected powerline, linking the bus
    nodes at each line's origin and extremity.
    """

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

    @property
    def node_substations(self) -> npt.NDArray[np.int64]:
        """Substation of every bus slot [max_nodes, 2] (slot // 2); second column -1."""
        slots = np.arange(self._max_nodes)
        return np.stack([slots // 2, np.full_like(slots, -1)], axis=1).astype(np.int64)

    def _get_nodes_and_mask(self, g2op_obs: BaseObservation) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """
        Aggregate per-element features into per-bus node features (see `aggregate_to_buses`).

        Node slot formula: 2 * sub_id + (bus - 1) for bus ∈ {1, 2}.
        Disconnected elements (bus == -1) are excluded.

        Returns:
            node_features: [max_nodes, x_dim] float32
            node_mask:     [max_nodes] bool — True for slots with ≥1 element
        """
        all_features = self._compute_all_node_features(g2op_obs)
        bus_ids_flat = self.node_buses(g2op_obs)
        connected = bus_ids_flat > 0
        active_slots = (2 * self._sub_ids_flat + (bus_ids_flat - 1))[connected]
        connected_features = {name: all_features[name][connected] for name in self.attr_to_observe}
        return aggregate_to_buses(connected_features, self.attr_to_observe, active_slots, self._max_nodes)

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

    Node features (`element_node_features`, zero where a feature does not apply to the node type):
      - gens, loads, storage: p, q, v (generator sign convention: loads negative), forecasts
      - lines: p, q, v of the origin (same columns) and of the extremity, rho, overflow steps,
        line cooldown, steps to maintenance (`scale_time_next_maintenance`), maintenance duration
      - ground/bus nodes: substation cooldown
      - node-type one-hot (gen, load, line, storage if the grid has any, ground, bus1, bus2)

    Edges are static (precomputed once): each element is connected to every
    busbar and ground node of its substation(s). Edge feature EDGE_ATTR[e, 0]
    is 1 if the element is currently assigned to that specific bus, 0 otherwise.
    EDGE_MASK is all-True (no padding — edge count is fixed).
    """

    # Bus-slot offsets within each substation block: (ground=0, bus1=1, bus2=2)
    _GROUND_OFFSET = 0
    _BUS1_OFFSET = 1
    _BUS2_OFFSET = 2

    def __init__(
        self,
        g2op_obs_space: ObservationSpace,
        verbose: bool = False,
        use_forecasts: bool = True,
    ):
        # If False, get_forecast_arrays() is never called and the forecast columns are left out.
        self._use_forecasts = use_forecasts
        self._dims = _GridDimensions.from_obs_space(g2op_obs_space)
        self.feature_names = element_node_features(use_forecasts, has_storage=g2op_obs_space.n_storage > 0)
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

        self._node_type_features = self._build_node_type_features()

        # Precompute static edge_index and lookup tables for dynamic edge_attr
        self._edge_index, self._edge_element_idx, self._edge_bus_idx, \
            self._fwd_bus_selector = self._build_static_edges()
        self._num_edges = self._edge_index.shape[1]

        self._observation_space = Dict({
            NODES: Box(low=-np.inf, high=np.inf,
                       shape=(self._num_nodes, self.x_dim), dtype=np.float32),
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

        self._normalizer = RunningMeanStd(shape=(self.x_dim,))
        self._timings: dict[str, float] = {}

        if verbose:
            logger.info(
                f"ElementGraphObservationConverter: {self._num_nodes} nodes "
                f"({n_gen} gen, {n_load} load, {n_line} line, {n_storage} storage, "
                f"{3 * n_sub} bus/ground), {self._num_edges} edges, "
                f"x_dim={self.x_dim} ({', '.join(self.feature_names)})."
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
        return len(self.feature_names)

    @property
    def num_edges(self) -> int:
        return self._num_edges

    @property
    def node_substations(self) -> npt.NDArray[np.int64]:
        """Substation(s) of every node [num_nodes, 2]: lines have both ends, other nodes -1 in column 2."""
        first = self._place(gen=self._gen_subid, load=self._load_subid, line=self._line_or_subid,
                            storage=self._storage_subid, bus=np.repeat(np.arange(self._n_sub), 3))
        second = np.full(self._num_nodes, -1.0, dtype=np.float32)
        second[self._line_offset:self._line_offset + self._n_line] = self._line_ex_subid
        return np.stack([first, second], axis=1).astype(np.int64)

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

    def _place(
        self,
        gen: Optional[npt.ArrayLike] = None,
        load: Optional[npt.ArrayLike] = None,
        line: Optional[npt.ArrayLike] = None,
        storage: Optional[npt.ArrayLike] = None,
        bus: Optional[npt.ArrayLike] = None,
    ) -> npt.NDArray[np.float32]:
        """One node column [num_nodes] from values per node type; types left out get 0.

        Args:
            gen, load, line, storage: Values per element of that type ([n_gen], [n_load], ...).
            bus: Values per ground/bus node [3 * n_sub], in node order (ground, bus1, bus2) per substation.
        """
        column = np.zeros(self._num_nodes, dtype=np.float32)
        blocks = ((self._gen_offset, gen), (self._load_offset, load), (self._line_offset, line),
                  (self._storage_offset, storage), (self._bus_offset, bus))
        for offset, values in blocks:
            if values is not None:
                values = np.asarray(values, dtype=np.float32)
                column[offset:offset + len(values)] = values
        return column

    def _build_node_type_features(self) -> dict[str, npt.NDArray[np.float32]]:
        """Static one-hot columns: `node_type_<type>` -> [num_nodes] (1 on nodes of that type)."""
        counts = [self._n_gen, self._n_load, self._n_line, self._n_storage]
        node_type = np.concatenate([np.repeat(np.arange(4), counts), np.tile([4, 5, 6], self._n_sub)])
        return {f"node_type_{name}": (node_type == i).astype(np.float32) for i, name in enumerate(_ELEMENT_NODE_TYPES)}

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

    def _compute_node_features(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray[np.float32]]:
        """All node features of `element_node_features`, each as a column [num_nodes] in node order.

        Args:
            g2op_obs: Current grid2op observation.
        Returns:
            Feature name -> column; 0 on node types the feature does not apply to.
        """
        obs = g2op_obs
        fc = next_step_forecasts(obs, self._dims) if self._use_forecasts else zero_forecasts(self._dims)
        return {
            # Generator sign convention: generation positive, consumption negative.
            "active_power": self._place(gen=obs.gen_p, load=-obs.load_p, line=obs.p_or, storage=obs.storage_power),
            "reactive_power": self._place(gen=obs.gen_q, load=-obs.load_q, line=obs.q_or),
            "voltage": self._place(gen=obs.gen_v, load=obs.load_v, line=obs.v_or),
            "active_power_ex": self._place(line=obs.p_ex),
            "reactive_power_ex": self._place(line=obs.q_ex),
            "voltage_ex": self._place(line=obs.v_ex),
            "active_power_forecast": self._place(gen=fc.gen_p, load=fc.load_p),
            # grid2op forecasts no gen q; the generator slot holds the voltage setpoint forecast.
            "reactive_power_forecast": self._place(gen=fc.gen_q, load=fc.load_q),
            "rho": self._place(line=obs.rho),
            "timestep_overflow": self._place(line=obs.timestep_overflow),
            "time_before_cooldown_line": self._place(line=obs.time_before_cooldown_line),
            "time_next_maintenance": self._place(line=scale_time_next_maintenance(obs.time_next_maintenance)),
            "duration_next_maintenance": self._place(line=obs.duration_next_maintenance),
            "time_before_cooldown_sub": self._place(bus=np.repeat(obs.time_before_cooldown_sub, 3)),
            **self._node_type_features,
        }

    def _get_node_features(self, g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """Node feature matrix [num_nodes, x_dim], columns in `feature_names` order."""
        features = self._compute_node_features(g2op_obs)
        return np.column_stack([features[name] for name in self.feature_names]).astype(np.float32)

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
        node_features = gym_obs[NODES]  # [num_nodes, x_dim]
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

    LODF depends only on network topology, so it is cached per ``topo_vect``
    and recomputed only on topology changes via the shared
    :func:`_compute_lodf` module-level helper.

    ``EDGE_TYPE`` is declared as ``Box(low=0, high=1, …)``; the downstream
    :class:`~rl.gnn_ppo_model.GNNModel` derives
    ``num_edge_types = high + 1 = 2`` automatically.

    Observation keys
    ----------------
    NODES      : [num_nodes, x_dim]      — element node features (identical to parent)
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

    needs_backend_state = True

    def __init__(
        self,
        backend: LightSimBackend,
        g2op_obs_space: ObservationSpace,
        verbose: bool = False,
        use_forecasts: bool = True,
    ):
        super().__init__(g2op_obs_space, verbose, use_forecasts)
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
                       shape=(self._num_nodes, self.x_dim), dtype=np.float32),
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
        self._cached_topo_vect: Optional[np.ndarray] = None

        if verbose:
            logger.info(
                f"ElementLODFGraphObservationConverter: {self._num_nodes} nodes, "
                f"<= {self._max_num_edges} edges "
                f"({self._num_edges} element-bus + {self._n_lodf_edges} LODF), "
                f"x_dim={self.x_dim}."
            )

    def _get_lodf_cached(self, topo_vect: np.ndarray) -> None:
        """
        Update _cached_D when topo_vect changes. Delegates to :func:`_compute_lodf`.

        Args:
            topo_vect: Bus of every element ([dim_topo], -1 = disconnected); covers bus splits and line status.
        """
        self._cached_D, self._cached_topo_vect = _compute_lodf(
            self._grid, self._n_line, topo_vect,
            self._cached_topo_vect, self._cached_D,
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
        self._get_lodf_cached(g2op_obs.topo_vect)
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
    topo_vect: np.ndarray,
    cached_topo_vect: Optional[np.ndarray],
    cached_ptdf: Optional[np.ndarray],
    cached_D: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (ptdf, D, topo_vect_copy), recomputing only when topo_vect changes (bus splits change PTDF without any line switching).

    D[i, j] = sum_l |PTDF[l, i] - PTDF[l, j]|  (L1 PTDF distance, [2*n_sub, 2*n_sub]).

    DC PF is called directly on the C++ grid model, which is safe because
    lightsim2grid keeps separate DC and AC solver objects.

    Args:
        grid: lightsim2grid GridModel instance (backend._grid).
        topo_vect: Bus of every element ([dim_topo], -1 = disconnected); covers bus splits and line status.
        cached_topo_vect: Previously cached topo_vect (or None).
        cached_ptdf: Previously cached PTDF matrix (or None).
        cached_D: Previously cached distance matrix (or None).
    Returns:
        (ptdf, D, topo_vect_copy) — ptdf is [n_branches, 2*n_sub],
        D is [2*n_sub, 2*n_sub] symmetric, non-negative.
    Raises:
        RuntimeError: if the DC power flow diverges.
    """
    if cached_topo_vect is not None and np.array_equal(topo_vect, cached_topo_vect):
        return cached_ptdf, cached_D, cached_topo_vect
    Vinit = np.ones(grid.total_bus(), dtype=complex)
    Vdc = grid.dc_pf(Vinit, 10, 1e-8)
    if Vdc.shape[0] == 0:
        raise RuntimeError(
            "DC power flow diverged; PTDF cannot be computed for current topology."
        )
    ptdf = grid.get_ptdf()
    diff = ptdf[:, :, np.newaxis] - ptdf[:, np.newaxis, :]  # (n_br, n_bus, n_bus)
    D = np.nan_to_num(np.abs(diff).sum(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    return ptdf, D, topo_vect.copy()


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
    merged over all elements assigned to each bus by `aggregate_to_buses`
    (`_BUS_AGGREGATION`: e.g. powers summed, voltage averaged, rho maximum).

    Bus slot convention (aligns with PTDF column indices from lightsim2grid):
        slot = (busbar - 1) * n_sub + sub_id    for busbar ∈ {1, 2}
    Slots 0..n_sub-1 → busbar 1; slots n_sub..2*n_sub-1 → busbar 2.
    """

    needs_backend_state = True

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

        # PTDF cache: only recompute when topo_vect changes.
        # _cached_D is derived from _cached_ptdf and shares the same lifetime.
        self._cached_ptdf: Optional[np.ndarray] = None
        self._cached_D: Optional[np.ndarray] = None
        self._cached_topo_vect: Optional[np.ndarray] = None

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

    @property
    def node_substations(self) -> npt.NDArray[np.int64]:
        """Substation of every bus slot [max_nodes, 2] (slot % n_sub); second column -1."""
        slots = np.arange(self._max_nodes)
        return np.stack([slots % self._n_sub, np.full_like(slots, -1)], axis=1).astype(np.int64)

    def _get_ptdf_cached(self, topo_vect: np.ndarray) -> np.ndarray:
        """
        Return the PTDF matrix, recomputing only when topo_vect has changed.

        Delegates to the module-level :func:`_compute_ptdf_distance` helper which
        is shared with :class:`SubstationPTDFGraphObservationConverter`.

        Args:
            topo_vect: Bus of every element ([dim_topo], -1 = disconnected); covers bus splits and line status.
        Returns:
            PTDF matrix [n_branches, 2*n_sub].
        """
        self._cached_ptdf, self._cached_D, self._cached_topo_vect = _compute_ptdf_distance(
            self._grid, topo_vect,
            self._cached_topo_vect, self._cached_ptdf, self._cached_D,
        )
        return self._cached_ptdf

    def _get_nodes_and_mask(
        self, g2op_obs: BaseObservation
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """
        Aggregate per-element features into per-bus node features (see `aggregate_to_buses`).

        Bus slot formula: slot = (busbar - 1) * n_sub + sub_id, busbar ∈ {1, 2}.
        Disconnected elements (bus == -1) are excluded.

        Args:
            g2op_obs: Current grid2op observation.
        Returns:
            node_features: [max_nodes, x_dim] float32
            node_mask:     [max_nodes] bool — True for slots with ≥1 connected element
        """
        all_features = self._compute_all_node_features(g2op_obs)
        bus_ids_flat = self.node_buses(g2op_obs)
        connected = bus_ids_flat > 0
        active_slots = ((bus_ids_flat - 1) * self._n_sub + self._sub_ids_flat)[connected]
        connected_features = {name: all_features[name][connected] for name in self.attr_to_observe}
        return aggregate_to_buses(connected_features, self.attr_to_observe, active_slots, self._max_nodes)

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
        self._get_ptdf_cached(g2op_obs.topo_vect)
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
    topo_vect: np.ndarray,
    cached_topo_vect: Optional[np.ndarray],
    cached_D: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (D, topo_vect_copy) where D = |LODF[:n_line, :n_line]|.

    Recomputes only when *topo_vect* differs from *cached_topo_vect*.
    DC PF is called directly on the C++ grid model, which is safe because
    lightsim2grid keeps separate DC and AC solver objects.

    Args:
        grid: lightsim2grid GridModel instance (backend._grid).
        n_line: number of AC transmission lines.
        topo_vect: Bus of every element ([dim_topo], -1 = disconnected); covers bus splits and line status.
        cached_topo_vect: previously cached topo_vect (or None).
        cached_D: previously cached distance matrix (or None).
    Returns:
        (D, topo_vect_copy) — D is [n_line, n_line] float64, non-negative.
    Raises:
        RuntimeError: if the DC power flow diverges.
    """
    if cached_topo_vect is not None and np.array_equal(topo_vect, cached_topo_vect):
        return cached_D, cached_topo_vect
    Vinit = np.ones(grid.total_bus(), dtype=complex)
    Vdc = grid.dc_pf(Vinit, 10, 1e-8)
    if Vdc.shape[0] == 0:
        raise RuntimeError(
            "DC power flow diverged; LODF cannot be computed for current topology."
        )
    lodf = grid.get_lodf()
    lodf_sub = np.nan_to_num(lodf[:n_line, :n_line], nan=0.0, posinf=0.0, neginf=0.0)
    return np.abs(lodf_sub), topo_vect.copy()


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

    NODE_MASK is all True: disconnected lines stay nodes (their cooldown and maintenance stay
    visible, `line_status` = 0), but have no edges.
    EDGE_MASK is True only when both endpoint lines are connected.
    EDGES carries the running-normalized |LODF| value as a scalar [E, 1].

    Like PTDF, LODF depends only on network topology (not on operating point),
    so it is cached per ``topo_vect`` and recomputed only on topology changes.

    Node features per line (`lodf_node_features`): p, q, v of both line ends, the line state of
    `complete_node_features` (rho, overflow, cooldown, maintenance), `line_status`, and per line end
    the substation cooldown and the net injection (generators − loads + storage: p, q, forecasts)
    of the bus the end is connected to (0 if disconnected).
    """

    needs_backend_state = True

    def __init__(
        self,
        backend: LightSimBackend,
        g2op_obs_space: ObservationSpace,
        verbose: bool = False,
        use_forecasts: bool = True,
    ):
        if g2op_obs_space.n_line == 0:
            raise ValueError("LODFGraphObservationConverter requires n_line > 0.")

        self._grid = backend._grid
        self._n_line = g2op_obs_space.n_line
        self._n_sub = g2op_obs_space.n_sub
        self._dims = _GridDimensions.from_obs_space(g2op_obs_space)
        self._use_forecasts = use_forecasts
        self.feature_names = lodf_node_features(use_forecasts)
        self._line_or_subid = g2op_obs_space.line_or_to_subid.astype(np.int64)
        self._line_ex_subid = g2op_obs_space.line_ex_to_subid.astype(np.int64)
        # Substation of every injection element, in order [gen | load | storage].
        self._injection_subid = np.concatenate([
            g2op_obs_space.gen_to_subid, g2op_obs_space.load_to_subid, g2op_obs_space.storage_to_subid,
        ]).astype(np.int64)

        n_line = self._n_line
        self._num_edges = n_line * (n_line - 1)  # directed, no self-loops

        # Static fully-connected directed edge index (no self-loops)
        idx = np.arange(n_line)
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        no_self_loop = ii != jj
        self._edge_index = np.stack([ii[no_self_loop], jj[no_self_loop]]).astype(np.int64)

        self._observation_space = Dict({
            NODES: Box(low=-np.inf, high=np.inf, shape=(n_line, self.x_dim), dtype=np.float32),
            EDGE_INDEX: Box(low=0, high=n_line - 1, shape=(2, self._num_edges), dtype=np.int64),
            EDGE_MASK: Box(low=0, high=1, shape=(self._num_edges,), dtype=np.bool_),
            EDGES: Box(low=-np.inf, high=np.inf, shape=(self._num_edges, 1), dtype=np.float32),
            NODE_MASK: Box(low=0, high=1, shape=(n_line,), dtype=np.bool_),
            GLOBAL: Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })

        self._normalizer = RunningMeanStd(shape=(self.x_dim,))
        self._edge_normalizer = RunningMeanStd(shape=(1,))
        self._timings: dict[str, float] = {}

        # LODF cache: _cached_D = |LODF[:n_line, :n_line]|, shares lifetime with LODF.
        self._cached_D: Optional[np.ndarray] = None
        self._cached_topo_vect: Optional[np.ndarray] = None

        if verbose:
            logger.info(
                f"LODFGraphObservationConverter: {n_line} nodes (powerlines), "
                f"{self._num_edges} directed edges, x_dim={self.x_dim} ({', '.join(self.feature_names)})."
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

    @property
    def x_dim(self) -> int:
        """Node features per line."""
        return len(self.feature_names)

    @property
    def node_substations(self) -> npt.NDArray[np.int64]:
        """Substations of both ends of every line [n_line, 2]."""
        return np.stack([self._line_or_subid, self._line_ex_subid], axis=1)

    def _get_lodf_cached(self, topo_vect: np.ndarray) -> None:
        """
        Update _cached_D and _cached_topo_vect if topo_vect has changed.

        Delegates to the module-level :func:`_compute_lodf` helper which is
        shared with :class:`ElementLODFGraphObservationConverter`.

        Args:
            topo_vect: Bus of every element ([dim_topo], -1 = disconnected); covers bus splits and line status.
        """
        self._cached_D, self._cached_topo_vect = _compute_lodf(
            self._grid, self._n_line, topo_vect,
            self._cached_topo_vect, self._cached_D,
        )

    def _bus_injections(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray[np.float32]]:
        """Net injection of generators, loads and storage per bus slot 2 * sub + (bus - 1).

        Same values as the summed power columns of the bus-level converters, restricted to injections.

        Returns:
            `injection_<feature>` -> [2 * n_sub] for every feature of `_INJECTION_FEATURES` and forecasts.
        """
        obs, dims = g2op_obs, self._dims
        fc = next_step_forecasts(obs, dims) if self._use_forecasts else zero_forecasts(dims)
        zeros_storage = np.zeros(dims.n_storage, dtype=np.float32)
        values = {
            "active_power": np.concatenate([obs.gen_p, -obs.load_p, obs.storage_power]),
            "reactive_power": np.concatenate([obs.gen_q, -obs.load_q, zeros_storage]),
        }
        if self._use_forecasts:
            values["active_power_forecast"] = np.concatenate([fc.gen_p, fc.load_p, zeros_storage])
            values["reactive_power_forecast"] = np.concatenate([fc.gen_q, fc.load_q, zeros_storage])
        buses = np.concatenate([obs.gen_bus, obs.load_bus, obs.storage_bus]).astype(np.int64)
        connected = buses > 0
        slots = (2 * self._injection_subid + buses - 1)[connected]
        names = list(values)
        sums, _ = aggregate_to_buses({name: values[name][connected] for name in names}, names, slots, 2 * self._n_sub)
        return {f"injection_{name}": sums[:, i] for i, name in enumerate(names)}

    def _line_end_features(
        self, g2op_obs: BaseObservation, side: str, injections: dict[str, npt.NDArray[np.float32]],
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Substation cooldown and bus injections at one end of every line.

        Args:
            g2op_obs: Current grid2op observation.
            side: "or" (origin) or "ex" (extremity).
            injections: Output of `_bus_injections`.
        Returns:
            `<feature>_<side>` -> [n_line]; injections are 0 for a disconnected end.
        """
        bus = (g2op_obs.line_or_bus if side == "or" else g2op_obs.line_ex_bus).astype(np.int64)
        sub = self._line_or_subid if side == "or" else self._line_ex_subid
        connected = bus > 0
        slot = np.where(connected, 2 * sub + bus - 1, 0)
        features = {"time_before_cooldown_sub": g2op_obs.time_before_cooldown_sub[sub].astype(np.float32)}
        features.update({name: np.where(connected, per_bus[slot], 0.0) for name, per_bus in injections.items()})
        return {f"{name}_{side}": values for name, values in features.items()}

    def _get_node_features(self, g2op_obs: BaseObservation) -> npt.NDArray[np.float32]:
        """
        Build the [n_line, x_dim] node feature matrix (columns in `feature_names` order).

        Disconnected lines keep their state (cooldown, maintenance); grid2op reports 0 flow for them.

        Args:
            g2op_obs: Current grid2op observation.
        Returns:
            Node feature matrix [n_line, x_dim] float32.
        """
        obs = g2op_obs
        features = {
            "active_power_or": obs.p_or, "reactive_power_or": obs.q_or, "voltage_or": obs.v_or,
            "active_power_ex": obs.p_ex, "reactive_power_ex": obs.q_ex, "voltage_ex": obs.v_ex,
            "rho": obs.rho,
            "timestep_overflow": obs.timestep_overflow,
            "time_before_cooldown_line": obs.time_before_cooldown_line,
            "time_next_maintenance": scale_time_next_maintenance(obs.time_next_maintenance),
            "duration_next_maintenance": obs.duration_next_maintenance,
            "line_status": obs.line_status,
        }
        injections = self._bus_injections(obs)
        for side in ("or", "ex"):
            features.update(self._line_end_features(obs, side, injections))
        return np.column_stack([features[name] for name in self.feature_names]).astype(np.float32)

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """
        Convert a grid2op observation to an LODF-based contingency coupling graph.

        Args:
            g2op_obs: The grid2op observation to convert.
        Returns:
            Dict with keys:
              NODES:      [n_line, x_dim] float32, per-line features (see `lodf_node_features`).
              EDGE_INDEX: [2, n_line*(n_line-1)] int64, static fully-connected directed index.
              EDGE_MASK:  [n_line*(n_line-1)] bool, True where both endpoint lines are connected.
              EDGES:      [n_line*(n_line-1), 1] float32, normalized |LODF[i,j]| (zeroed for inactive).
              NODE_MASK:  [n_line] bool, all True (disconnected lines stay nodes without edges).
              GLOBAL:     [6] float32, time-based features.
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        node_features = self._get_node_features(g2op_obs)
        self._timings["node_features_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self._get_lodf_cached(g2op_obs.topo_vect)
        self._timings["lodf_ms"] = (time.perf_counter() - t0) * 1000
        if self._cached_D is None:
            raise RuntimeError("LODF cache is uninitialized after _get_lodf_cached — DC PF likely failed.")

        line_status = g2op_obs.line_status.astype(bool)
        node_mask = np.ones(self._n_line, dtype=np.bool_)
        global_features = self._get_global_features(g2op_obs)

        src, dst = self._edge_index[0], self._edge_index[1]
        edge_weights = self._cached_D[src, dst].astype(np.float32)[:, np.newaxis]  # [num_edges, 1]
        edge_mask = line_status[src] & line_status[dst]

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

        Only nodes with NODE_MASK=True update the node normalizer; masked nodes are zeroed
        after normalization.  Only active edges (EDGE_MASK=True) update the edge normalizer;
        inactive edges are zeroed.

        Args:
            gym_obs: LODF graph observation dict with raw features.
        """
        node_features = gym_obs[NODES]   # (n_line, x_dim)
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
    is cached per ``topo_vect`` and recomputed only on topology changes using
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

    needs_backend_state = True

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
        self._cached_topo_vect: Optional[np.ndarray] = None

        if verbose:
            logger.info(
                f"SubstationPTDFGraphObservationConverter: {self._max_nodes} max nodes "
                f"({n_sub} subs × 2 buses), {self._max_num_edges} edges "
                f"({self._max_num_edges - self._n_phys_edges} topology + "
                f"{self._n_phys_edges} PTDF), x_dim={x_dim}."
            )

    def _get_ptdf_cached(self, topo_vect: np.ndarray) -> None:
        """
        Update _cached_D (in Substation slot order) when topo_vect changes.

        Delegates to :func:`_compute_ptdf_distance`, then applies the
        PTDF-to-Substation slot permutation so subsequent indexing is
        consistent with the node ordering used by this converter.

        Args:
            topo_vect: Bus of every element ([dim_topo], -1 = disconnected); covers bus splits and line status.
        """
        prev_D = self._cached_D
        self._cached_ptdf, D_ptdf, self._cached_topo_vect = _compute_ptdf_distance(
            self._grid, topo_vect,
            self._cached_topo_vect, self._cached_ptdf, self._cached_D,
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
        self._get_ptdf_cached(g2op_obs.topo_vect)
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

    needs_backend_state = True

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


def sync_backend_to_observation(env, obs: BaseObservation) -> bool:
    """Put `env`'s backend into the grid state of `obs` (from any env of the same grid).

    Sets every element's bus (incl. line status) and the injections, then runs an AC power flow,
    so converters with `needs_backend_state` compute the same PTDF/LODF/Zbus as in the env that
    produced `obs`. `backend.update_from_obs` cannot be used: observations carry no shunt data.

    Args:
        env: Grid2Op environment whose backend the converter reads.
        obs: Observation to reproduce, e.g. rebuilt with `observation_space.from_vect`.

    Returns:
        Whether the power flow converged.
    """
    action = env.action_space({
        "set_bus": obs.topo_vect,
        "injection": {"load_p": obs.load_p, "load_q": obs.load_q, "prod_p": obs.gen_p, "prod_v": obs.gen_v},
    })
    backend_action = env._backend_action_class()
    backend_action += action
    env.backend.apply_action(backend_action)
    converged, _ = env.backend.runpf(is_dc=False)
    return bool(converged)


class FlatGraphObservationConverter(ObservationConverter[Dict]):
    """The node features of a graph converter plus the bus of every node, as one input for an MLP.

    Same normalized numbers as the graph converter's NODES, without the constant node-type columns,
    plus the bus of every node (-1 = disconnected, 1, 2) in place of the edges. The MLP baseline of
    the graph comparison: the same information as the Complete graph, without graph structure.

    Observation: {FLAT_FEATURES: [num_nodes, x_dim]}; `FlatMLPModel` flattens it.
    """

    def __init__(self, graph: GraphObservationConverter, verbose: bool = False):
        """Args:
            graph: Converter that computes and normalizes the node features.
            verbose: Log the input size.
        """
        self._graph = graph
        self._kept_columns = [i for i, name in enumerate(graph.attr_to_observe) if not name.startswith("node_type_")]
        self._observation_space = Dict({
            FLAT_FEATURES: Box(low=-np.inf, high=np.inf, shape=(graph.num_nodes, len(self._kept_columns) + 1),
                               dtype=np.float32),
        })
        if verbose:
            logger.info(f"FlatGraphObservationConverter: {graph.num_nodes} nodes x "
                        f"{len(self._kept_columns) + 1} features (node features without types, bus).")

    @property
    def observation_space(self) -> Dict:
        """{FLAT_FEATURES: Box [num_nodes, x_dim]}."""
        return self._observation_space

    @property
    def _timings(self) -> dict[str, float]:
        """Timings of the graph converter (read by the env)."""
        return self._graph._timings

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """Normalized node features of the graph converter plus the bus column: {FLAT_FEATURES: [N, x_dim]}."""
        nodes = self._graph.to_gym(g2op_obs)[NODES][:, self._kept_columns]
        buses = self._graph.node_buses(g2op_obs).astype(np.float32)
        return {FLAT_FEATURES: np.column_stack([nodes, buses]).astype(np.float32)}

    def normalize(self, gym_obs: dict) -> dict:
        """Identity: the graph converter normalizes the node features in `to_gym`."""
        return gym_obs


class NodeSubstationsConverter(ObservationConverter[Dict]):
    """Adds NODE_SUBSTATIONS, the static substation(s) of every node, to a graph converter's output.

    Needed by the per-substation readout of `GNNModel` (`rl.readout.substation_readout`).
    The wrapped converter must have a `node_substations` property ([max_nodes, 2], -1 = none).
    """

    def __init__(self, converter: ObservationConverter[Dict], n_sub: int):
        """Args:
            converter: Graph converter to wrap.
            n_sub: Substations of the grid (upper bound of the substation ids).
        """
        self.converter = converter
        self.needs_backend_state = converter.needs_backend_state
        self._node_substations = converter.node_substations
        spaces = dict(converter.observation_space.spaces)
        spaces[NODE_SUBSTATIONS] = Box(low=-1, high=n_sub - 1, shape=self._node_substations.shape, dtype=np.int64)
        self._observation_space = Dict(spaces)

    @property
    def observation_space(self) -> Dict:
        """Observation space of the wrapped converter plus NODE_SUBSTATIONS [max_nodes, 2]."""
        return self._observation_space

    @property
    def _timings(self) -> dict[str, float]:
        """Timings of the wrapped converter (read by the env)."""
        return self.converter._timings

    def to_gym(self, g2op_obs: BaseObservation) -> dict[str, npt.NDArray]:
        """Observation of the wrapped converter plus NODE_SUBSTATIONS."""
        return {**self.converter.to_gym(g2op_obs), NODE_SUBSTATIONS: self._node_substations}

    def normalize(self, gym_obs: dict) -> dict:
        """Normalize with the wrapped converter, keeping NODE_SUBSTATIONS."""
        return {**self.converter.normalize(gym_obs), NODE_SUBSTATIONS: self._node_substations}

    def close(self) -> None:
        """Close the wrapped converter."""
        self.converter.close()

    def reset_obs(self) -> None:
        """Reset the wrapped converter at the start of an episode."""
        self.converter.reset_obs()


# --- Factory ---
def resolve_node_features(env_config: dict, g2op_obs_space: ObservationSpace) -> Optional[list[str]]:
    """Node features of the attribute-list converters (Graph, Heterogeneous, Substation, PTDF, Zbus, ...).

    Args:
        env_config: Env config; reads `attr_to_observe`, `node_features` ("default" or "complete")
            and `use_forecasts`.
        g2op_obs_space: Observation space of the grid2op env.

    Returns:
        Explicit `attr_to_observe` if set; `complete_node_features` for `node_features: complete`;
        the default features without forecasts if `use_forecasts` is off; else None (converter default).
    """
    use_forecasts = env_config.get("use_forecasts", True)
    node_features = env_config.get("node_features", "default")
    if node_features not in ("default", "complete"):
        raise ValueError(f"Unknown node_features {node_features!r}; expected 'default' or 'complete'.")
    if env_config.get("attr_to_observe") is not None:
        return env_config["attr_to_observe"]
    if node_features == "complete":
        return complete_node_features(use_forecasts, has_storage=g2op_obs_space.n_storage > 0)
    if not use_forecasts:
        return [f for f in _DEFAULT_NODE_FEATURES if f not in _FORECAST_NODE_FEATURES]
    return None


def make_observation_converter(gym_env: GymEnv, env_config: dict, backend: LightSimBackend) -> ObservationConverter:
    """Construct the appropriate ObservationConverter from env_config.

    Args:
        gym_env: GymEnv wrapper of the grid2op env (for the observation space).
        env_config: Env config, see `core.rllib_config.build_env_config`. `node_substations: true`
            adds NODE_SUBSTATIONS to the observations (`NodeSubstationsConverter`).
        backend: Backend of the grid2op env that is actually stepped. Converters with
            `needs_backend_state` read its grid model. Not `gym_env.init_env.backend`: GymEnv works
            on a copy of the env, whose backend never follows the episode.
    """
    converter = _make_base_converter(gym_env, env_config, backend)
    if env_config.get("node_substations", False):
        converter = NodeSubstationsConverter(converter, n_sub=gym_env.init_env.observation_space.n_sub)
    return converter


def _make_base_converter(gym_env: GymEnv, env_config: dict, backend: LightSimBackend) -> ObservationConverter:
    """The converter selected by `env_config["observation_space"]`; arguments as `make_observation_converter`."""
    mode = env_config.get("observation_space", "FlatSpace")
    g2op_obs_space = gym_env.init_env.observation_space
    verbose = env_config.get("verbose", False)
    use_forecasts = env_config.get("use_forecasts", True)
    # use_forecasts=False removes forecasts from every converter: the default node features of the
    # attribute-list converters lose their forecast columns, the element and LODF converters drop theirs.
    attr_to_observe = resolve_node_features(env_config, g2op_obs_space)
    if mode == "GraphObsSpace" or mode == "BusConnectivityGraphObsSpace":
        return GraphObservationConverter(g2op_obs_space, attr_to_observe=attr_to_observe, verbose=verbose)
    elif mode == "CompleteGraphObsSpace":
        return CompleteGraphObservationConverter(g2op_obs_space, use_forecasts=use_forecasts, verbose=verbose)
    elif mode == "EdgelessGraphObsSpace":
        return EdgelessGraphObservationConverter(g2op_obs_space, attr_to_observe=attr_to_observe, verbose=verbose)
    elif mode == "FlatGraphObsSpace":
        graph = GraphObservationConverter(g2op_obs_space, attr_to_observe=attr_to_observe)
        return FlatGraphObservationConverter(graph, verbose=verbose)
    elif mode == "HeterogeneousGraphObsSpace":
        return HeterogeneousGraphObservationConverter(g2op_obs_space, attr_to_observe=attr_to_observe, verbose=verbose)
    elif mode == "SubstationGraphObsSpace":
        return SubstationGraphObservationConverter(g2op_obs_space, attr_to_observe=attr_to_observe, verbose=verbose)
    elif mode == "ElementGraphObsSpace":
        return ElementGraphObservationConverter(g2op_obs_space, verbose=verbose, use_forecasts=use_forecasts)
    elif mode == "PTDFGraphObsSpace":
        return PTDFGraphObservationConverter(backend, g2op_obs_space, attr_to_observe=attr_to_observe, verbose=verbose)
    elif mode == "LODFGraphObsSpace":
        return LODFGraphObservationConverter(backend, g2op_obs_space, verbose=verbose, use_forecasts=use_forecasts)
    elif mode == "ElementLODFGraphObsSpace":
        return ElementLODFGraphObservationConverter(backend, g2op_obs_space, verbose=verbose, use_forecasts=use_forecasts)
    elif mode == "ZbusGraphObsSpace":
        return ZbusGraphObservationConverter(backend, g2op_obs_space, attr_to_observe=attr_to_observe, verbose=verbose)
    elif mode == "SubstationPTDFGraphObsSpace":
        return SubstationPTDFGraphObservationConverter(backend, g2op_obs_space, attr_to_observe=attr_to_observe,
                                                       verbose=verbose)
    elif mode == "SubstationZbusGraphObsSpace":
        return SubstationZbusGraphObservationConverter(backend, g2op_obs_space, attr_to_observe=attr_to_observe,
                                                       verbose=verbose)
    elif mode == "FlatObsSpace":
        g2op_input = env_config.get("g2op_input", ["p_i", "p_l", "q_i", "q_l", "a", "v_i", "v_l", "theta_i", "theta_l", "r", "t"])
        # "vect": every attribute of obs.to_vect(), i.e. exactly what imitation data stores.
        attrs = (list(g2op_obs_space.attr_list_vect) if "vect" in g2op_input
                 else get_attr_list(g2op_input))
        return FlatObservationConverter(gym_env=gym_env, attr_to_observe=attrs, verbose=verbose)
    else:
        raise ValueError(f"Unknown observation space type: {mode}")

if __name__ == "__main__":
    import grid2op
    env = grid2op.make("l2rpn_wcci_2022")
    graph = GraphObservationConverter(env.observation_space)
    print(graph.num_nodes)
