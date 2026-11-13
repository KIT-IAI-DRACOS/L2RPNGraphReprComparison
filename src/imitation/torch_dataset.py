"""
Imitation data as model inputs: stored `obs.to_vect()` rows converted by the RL env's own
observation converter, so the policy network sees exactly what it sees during RL.

Conversion runs once per (dataset, converter config) and is cached as `.npz`, since rebuilding
~100k grid2op observations takes minutes. The converter normalizes with running statistics (as in
the RL env); a warm-up pass over a random subset lets them settle before any state is stored.

Edge arrays are padded to the converter's maximum edge count, which is far above what occurs.
They are trimmed to the largest edge count in the dataset before caching; the models only read
edges selected by EDGE_MASK, so the trimmed width does not change their output.

Converters that read the backend's grid model (PTDF, LODF, Zbus; `needs_backend_state`) get the
backend put into each stored state first (`sync_backend_to_observation`), so they compute the
same matrices as in the env that recorded the state.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from grid2op.gym_compat import GymEnv
from torch.utils.data import Dataset

from grid2op_env.observation_converter import (
    EDGE_INDEX,
    EDGE_MASK,
    EDGE_TYPE,
    EDGE_WEIGHTS,
    EDGES,
    make_observation_converter,
    sync_backend_to_observation,
)
from grid2op_env.utils import make_g2op_env
from imitation.dataset import ImitationData

logger = logging.getLogger(__name__)

# Converter outputs with one entry per (padded) edge.
_EDGE_KEYS = (EDGE_INDEX, EDGE_MASK, EDGES, EDGE_TYPE, EDGE_WEIGHTS)

# env_config entries that determine the converter's output.
_CONVERTER_KEYS = ("env_name", "observation_space", "use_forecasts", "attr_to_observe", "g2op_input", "custom_input")


def converter_cache_key(dataset_path: Path, env_config: dict[str, Any], warmup: int, seed: int) -> str:
    """Hash of everything that determines a conversion, used as cache file name."""
    stat = Path(dataset_path).stat()
    payload = {
        "dataset": [str(Path(dataset_path).resolve()), stat.st_size, stat.st_mtime_ns],
        "converter": {k: env_config.get(k) for k in _CONVERTER_KEYS},
        "warmup": warmup,
        "seed": seed,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _trim_edges(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Cut padded edge axes down to the largest edge count present (see module docstring)."""
    if EDGE_MASK not in arrays:
        return arrays
    full = arrays[EDGE_MASK].shape[1]
    keep = max(int(arrays[EDGE_MASK].sum(axis=1).max()), 1)
    if not arrays[EDGE_MASK][:, keep:].any():
        for key in _EDGE_KEYS:
            if key in arrays:
                axis = list(arrays[key].shape[1:]).index(full) + 1
                arrays[key] = np.take(arrays[key], np.arange(keep), axis=axis)
    return arrays


def convert_observations(obs: np.ndarray, env_config: dict[str, Any], warmup: int, seed: int) -> dict[str, np.ndarray]:
    """Convert `obs.to_vect()` rows into the RL policy's observation dict, stacked per key.

    Args:
        obs: [n, obs_dim] observation vectors.
        env_config: The RL env config (as from `core.rllib_config.build_env_config`).
        warmup: States converted (and discarded) first so the running normalizer settles.
        seed: Seed for drawing the warm-up states.

    Returns:
        Key -> array [n, ...]; int64 arrays are stored as int32.
    """
    g2op_env = make_g2op_env(env_config)
    try:
        converter = make_observation_converter(GymEnv(g2op_env, with_forecast=True), env_config, backend=g2op_env.backend)
        n_diverged = 0

        def to_gym(row: np.ndarray) -> dict:
            nonlocal n_diverged
            g2op_obs = g2op_env.observation_space.from_vect(row)
            if converter.needs_backend_state and not sync_backend_to_observation(g2op_env, g2op_obs):
                n_diverged += 1
            return converter.to_gym(g2op_obs)

        rng = np.random.default_rng(seed)
        for i in rng.choice(len(obs), size=min(warmup, len(obs)), replace=False):
            to_gym(obs[i])

        parts: dict[str, list[np.ndarray]] = {}
        for i, row in enumerate(obs):
            for key, value in to_gym(row).items():
                value = np.asarray(value)
                parts.setdefault(key, []).append(value.astype(np.int32) if value.dtype == np.int64 else value)
            if (i + 1) % 10000 == 0:
                logger.info(f"Converted {i + 1}/{len(obs)} observations")
        if n_diverged:
            logger.warning(f"AC power flow did not converge for {n_diverged} synced states (incl. warm-up)")
    finally:
        g2op_env.close()
    return _trim_edges({k: np.stack(v) for k, v in parts.items()})


def load_or_convert(
    dataset_path: Path,
    data: ImitationData,
    env_config: dict[str, Any],
    cache_dir: Path,
    warmup: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Converted observations of a whole dataset, from cache if available."""
    cache_path = Path(cache_dir) / f"{Path(dataset_path).stem}_{converter_cache_key(dataset_path, env_config, warmup, seed)}.npz"
    if cache_path.exists():
        logger.info(f"Loading converted observations from {cache_path}")
        with np.load(cache_path) as cached:
            return {k: cached[k] for k in cached.files}
    logger.info(f"Converting {len(data)} observations with {env_config['observation_space']} -> {cache_path}")
    arrays = convert_observations(data.obs, env_config, warmup, seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp.npz")
    np.savez(tmp_path, **arrays)
    tmp_path.replace(cache_path)
    return arrays


class ImitationTorchDataset(Dataset):
    """Samples `{"obs": {key: tensor}, "target": [A], "max_rho": [A], "valid": [A]}` for a subset.

    `max_rho`/`valid` are present only if the source recorded them (greedy Tutor), and are what
    the regret metrics need.

    Args:
        obs: Converted observations, as from `load_or_convert`.
        data: The dataset they were converted from.
        indices: Samples of this subset.
    """

    def __init__(self, obs: dict[str, np.ndarray], data: ImitationData, indices: np.ndarray):
        self.indices = np.asarray(indices)
        self.obs = {k: torch.from_numpy(v[self.indices]) for k, v in obs.items()}
        self.target = torch.from_numpy(data.policy_target[self.indices])
        self.max_rho = None if data.max_rho is None else torch.from_numpy(data.max_rho[self.indices])
        self.valid = torch.from_numpy(data.action_mask[self.indices])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = {
            "obs": {k: (v[i].long() if v.dtype == torch.int32 else v[i]) for k, v in self.obs.items()},
            "target": self.target[i],
        }
        if self.max_rho is not None:
            item["max_rho"] = self.max_rho[i]
            item["valid"] = self.valid[i]
        return item
