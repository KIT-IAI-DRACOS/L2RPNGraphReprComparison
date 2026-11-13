"""
Utilities in the grid2op and gym convertion.
"""

import os
import zlib
from pathlib import Path
from typing import Any

import grid2op
import numpy as np
from grid2op.Chronics import MultifolderWithCache
from grid2op.Environment import BaseEnv
from lightsim2grid import LightSimBackend


def make_train_test_val_split(
    library_directory: str,
    env_name: str,
    pct_val: float,
    pct_test: float,
) -> None:
    """
    Function that splits an environment into a train, test and validation set.
    """
    if not os.path.exists(os.path.join(library_directory, env_name, "_train")):
        env = grid2op.make(os.path.join(library_directory, env_name))
        env.train_val_split_random(
            pct_val=pct_val, pct_test=pct_test, add_for_test="test"
        )


def rename_env(env: BaseEnv):
    # if the path contains _per_day or _train or _test or _val, then ignore this part of the string
    env_name = env.env_name
    if "_per_day" in env_name:
        env_name = env_name.replace("_per_day", "")
    if "_train" in env_name:
        env_name = env_name.replace("_train", "")
    if "_test" in env_name:
        env_name = env_name.replace("_test", "")
    if "_val" in env_name:
        env_name = env_name.replace("_val", "")
    if "_small" in env_name:
        env_name = env_name.replace("_small", "")
    if "_large" in env_name:
        env_name = env_name.replace("_large", "")
    env.set_env_name(env_name)


def chronic_seed(base_seed: int, chronic_id: str) -> int:
    """Seed for one chronic: depends on the chronic's name, not on its position in any list.

    grid2op draws maintenance from the env seed, so a single seed for every chronic gives every
    chronic the same maintenance schedule. Evaluating another agent with the same seed per chronic
    reproduces the Tutor's maintenance exactly.

    Args:
        base_seed: Run-level seed.
        chronic_id: Chronic path or name.

    Returns:
        A seed in [0, 2**31).
    """
    return (base_seed + zlib.crc32(Path(chronic_id).name.encode())) % 2**31


def worker_seed(seed: int, worker_index: int, vector_index: int = 0) -> int:
    """Env seed of one env copy of an RLlib run, different for every rollout worker and vectorized env.

    Args:
        seed: Run-level seed.
        worker_index: RLlib worker index (0 = local worker, 1..n = remote workers).
        vector_index: Index of the env within a worker that vectorizes several envs.

    Returns:
        A seed in [0, 2**31).
    """
    return int(np.random.SeedSequence([seed, worker_index, vector_index]).generate_state(1)[0] % 2**31)


def shuffle_chronics(env: BaseEnv, seed: int) -> list[str]:
    """Shuffle the order in which `env.reset()` goes through the chronics; grid2op's default is sorted by name.

    Args:
        env: Grid2Op env with a multi-folder chronics handler.
        seed: Seed of the permutation.

    Returns:
        Chronic paths in the new order.
    """
    rng = np.random.default_rng(seed)
    return [str(path) for path in env.chronics_handler.shuffle(shuffler=lambda order: order[rng.permutation(len(order))])]


def chronic_shard(env: BaseEnv, shard: int, n_shards: int) -> list[str]:
    """One of `n_shards` disjoint parts of the env's chronics: chronic `shard`, `shard + n_shards`, ... by name.

    Args:
        env: Grid2Op env with a multi-folder chronics handler.
        shard: Index of the part, in [0, n_shards).
        n_shards: Number of parts, e.g. the number of evaluation workers.

    Returns:
        Chronic paths, as accepted by `env.set_id`.
    """
    return sorted(str(path) for path in env.chronics_handler.subpaths)[shard::n_shards]


def make_g2op_env(env_config: dict[str, Any]) -> BaseEnv:
    """
    Function that makes a grid2op environment.
    """
    chronics_dir = env_config.get("chronics_dir", None)
    use_chronics_cache = env_config.get("use_chronics_cache", False)
    extra_kwargs = {"chronics_class": MultifolderWithCache} if use_chronics_cache else {}

    # Pass full path directly to avoid grid2op.change_local_dir(), which writes to
    # ~/.grid2opconfig.json on shared NFS and causes race conditions between concurrent jobs.
    env_name = os.path.join(chronics_dir, env_config["env_name"]) if chronics_dir else env_config["env_name"]

    env = grid2op.make(
        env_name,
        **env_config["grid2op_kwargs"],
        **extra_kwargs,
        backend=LightSimBackend(),
    )
    if use_chronics_cache:
        env.chronics_handler.set_filter(lambda x: True)
        env.chronics_handler.reset()
    else:
        env.chronics_handler.set_chunk_size(100)

    if "seed" in env_config:
        env.seed(int(env_config["seed"]))

    # *** RENAME THE ENVIRONMENT *** excl _train / _val etc
    # such that it can gather the action space and normalization/scaling parameters
    rename_env(env)

    if env.env_name == "rte_case14_realistic":
        env.set_thermal_limit(np.array(
            [
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                1000,
                760,
                450,
                760,
                380,
                380,
                760,
                380,
                760,
                380,
                380,
                380,
                2000,
                2000,
            ])
        )
    return env


def get_attr_list(attr_abbreviated: list):
    if "all" in attr_abbreviated:
        attr = ["topo_vect", "line_status", "load_p", "gen_p", "p_ex", "p_or", "load_q", "gen_q", "q_ex", "q_or", "load_v", "gen_v", "v_ex", "v_or", "load_theta", "gen_theta", "theta_ex", "theta_or", "a_ex", "a_or", "rho", "timestep_overflow", "time_next_maintenance"]
    else:
        attr = []
        if "t" in attr_abbreviated:
            attr = ["topo_vect"]
        if "l" in attr_abbreviated:
            # include line status
            attr.extend(["line_status"])
        if "p_i" in attr_abbreviated:
            # include active power input
            attr.extend(["load_p", "gen_p"])
        if "p_l" in attr_abbreviated:
            # include active power line flows
            attr.extend(["p_ex", "p_or"])
        if "q_i" in attr_abbreviated:
            # include reactive power input
            attr.extend(["load_q", "gen_q"])
        if "q_l" in attr_abbreviated:
            # include reactive power line flows
            attr.extend(["q_ex", "q_or"])
        if "v_i" in attr_abbreviated:
            # include voltage input
            attr.extend(["load_v", "gen_v"])
        if "v_l" in attr_abbreviated:
            # include voltage line flows
            attr.extend(["v_ex", "v_or"])
        if "theta_i" in attr_abbreviated:
            # include voltage angle input
            attr.extend(["load_theta", "gen_theta"])
        if "theta_l" in attr_abbreviated:
            # include voltage angle line flows
            attr.extend(["theta_ex", "theta_or"])
        if "a" in attr_abbreviated:
            # include current line flows
            attr.extend(["a_ex", "a_or"])
        if "r" in attr_abbreviated:
            # include rho (power flow / thermal limit lines)
            attr.append("rho")
        if "o" in attr_abbreviated:
            # include ts since overflow
            attr.append("timestep_overflow")
        if "m" in attr_abbreviated:
            attr.append("time_next_maintenance")
    return attr
