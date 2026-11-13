"""
RL fine-tuning of a pretrained policy ("Senior"): Hydra overrides for experiments/train.py that rebuild
the network of a Junior run (experiments/pretrain_policy.py or one trial of experiments/pretrain_hpo.py).

`ImitationInitCallback` loads the Junior's weights strictly, so the RL model must have the Junior's
architecture and observation space. The overrides repeat every value of the `model` and `obs_space`
sections of the Junior's `config.yaml`; the RL baseline uses them too, so all agents share one network.
Dropout is replaced (default 0): RLlib samples in eval mode but updates in train mode, so dropout would
make PPO's probability ratio noisy. Dropout has no parameters, so the weights still load.
"""
from __future__ import annotations

import shlex
from typing import Any

from omegaconf import DictConfig, OmegaConf

ARCHITECTURE_SECTIONS = ("model", "obs_space")
# Machine-specific paths, filled by the RL run's own config instead.
SKIPPED_KEYS = frozenset({"model.gnn.edge_probs_path"})
DROPOUT_KEY = "dropout_prob"


def flatten(values: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Nested dict -> `{dotted.key: leaf value}`; lists are leaves.

    Args:
        values: Nested config values (plain containers, e.g. from `OmegaConf.to_container`).
        prefix: Dotted path of `values` itself, e.g. "model".
    """
    flat = {}
    for key, value in values.items():
        path = f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat


def format_value(value: Any) -> str:
    """A config leaf in Hydra's override grammar, e.g. `[256,256]`, `null`, `true`."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(format_value(v) for v in value) + "]"
    return str(value)


def architecture_overrides(junior_cfg: DictConfig, dropout_prob: float | None = 0.0) -> list[str]:
    """Hydra overrides that give an RL run the Junior's network and observation space.

    Args:
        junior_cfg: The Junior run's `config.yaml`.
        dropout_prob: Replaces every `dropout_prob` of the model; None keeps the Junior's.

    Returns:
        Overrides like `++model.gnn.hidden_dim=512` (`++` adds keys the RL config may lack).
    """
    overrides = []
    for section in ARCHITECTURE_SECTIONS:
        values = OmegaConf.to_container(junior_cfg[section], resolve=False)
        for key, value in flatten(values, section).items():
            if key in SKIPPED_KEYS:
                continue
            if dropout_prob is not None and key.endswith(f".{DROPOUT_KEY}"):
                value = dropout_prob
            overrides.append(f"++{key}={format_value(value)}")
    return overrides


def shell_overrides(overrides: list[str]) -> str:
    """Overrides as one shell-quoted line (protects `[...]` from globbing)."""
    return " ".join(shlex.quote(o) for o in overrides)


VAL_SURVIVAL_KEY = "grid2op_end_mean"  # mean last grid2op step of the val episodes of one evaluation


def select_best_val_checkpoint(results: dict[str, dict[str, Any]], evaluation_interval: int) -> str:
    """The kept checkpoint with the highest val survival among those evaluated with their own weights.

    RLlib evaluates every `evaluation_interval` iterations and repeats the last evaluation's results in
    between, so a checkpoint from another iteration (e.g. the one at the end of training) carries the
    val survival of older weights and is left out.

    Args:
        results: Checkpoint name -> its metrics with `training_iteration` and `VAL_SURVIVAL_KEY`
            (the trial's `checkpoint_results.json`, written by `core.train`).
        evaluation_interval: Iterations between two evaluations.

    Returns:
        Name of the selected checkpoint; the later one on ties.
    """
    evaluated = {name: metrics[VAL_SURVIVAL_KEY] for name, metrics in results.items()
                 if metrics.get("training_iteration") and metrics["training_iteration"] % evaluation_interval == 0
                 and metrics.get(VAL_SURVIVAL_KEY) is not None}
    if not evaluated:
        raise ValueError(f"No checkpoint from an evaluation iteration (every {evaluation_interval}) in {sorted(results)}")
    return max(sorted(evaluated, reverse=True), key=evaluated.get)
