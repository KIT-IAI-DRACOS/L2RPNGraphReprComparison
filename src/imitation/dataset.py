"""
Source-independent imitation dataset format.

The Tutor's raw output is converted into this layout, so training code never needs to know
where a dataset came from. Targets are built here, not in the Tutor, so e.g. a different
soft-label temperature only needs a rebuild, not a new rollout.

Per-sample fields (n samples, A = policy actions incl. do-nothing at index 0):
    obs            [n, obs_dim] float32  `obs.to_vect()`; rebuild with `observation_space.from_vect`.
    policy_target  [n, A] float32        Distribution over policy actions (rows sum to 1).
    action_mask    [n, A] bool           Actions known to be valid in that state (all True if unknown).
    chronic        [n] str               Chronic name, for chronic-level train/val/test splits.
    step           [n] int32             Step within the chronic (-1 if unknown).
    max_rho        [n, A] float32        Optional: simulated max rho per action (NaN if invalid),
                                         for regret evaluation. Absent if the source lacks it.
Metadata (scalars): source, target, temperature, action_space, n_actions.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from imitation.tutor import DO_NOTHING_INDEX, DO_NOTHING_MARGIN

GREEDY_TUTOR = "greedy_tutor"    # imitation.tutor, one .npz per chronic

HARD = "hard"
SOFT = "soft"

SPLITS = ("train", "val", "test")


@dataclass
class ImitationData:
    """In-memory imitation dataset in the shared format (see module docstring)."""

    obs: np.ndarray
    policy_target: np.ndarray
    action_mask: np.ndarray
    chronic: np.ndarray
    step: np.ndarray
    max_rho: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.obs)

    def save(self, path: Path) -> None:
        """Write the dataset to a single `.npz`."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "obs": self.obs,
            "policy_target": self.policy_target,
            "action_mask": self.action_mask,
            "chronic": self.chronic,
            "step": self.step,
        }
        if self.max_rho is not None:
            arrays["max_rho"] = self.max_rho
        arrays.update({f"meta_{k}": np.asarray(v) for k, v in self.metadata.items()})
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "ImitationData":
        """Read a dataset written by `save`."""
        with np.load(path) as data:
            return cls(
                obs=data["obs"],
                policy_target=data["policy_target"],
                action_mask=data["action_mask"],
                chronic=data["chronic"],
                step=data["step"],
                max_rho=data["max_rho"] if "max_rho" in data.files else None,
                metadata={k[len("meta_"):]: data[k].item() for k in data.files if k.startswith("meta_")},
            )


def one_hot(index: np.ndarray, n_actions: int) -> np.ndarray:
    """One-hot targets [n, n_actions] from action indices [n]."""
    target = np.zeros((len(index), n_actions), dtype=np.float32)
    target[np.arange(len(index)), index] = 1.0
    return target


def soft_targets(
    max_rho: np.ndarray, valid: np.ndarray, temperature: float, margin: float = DO_NOTHING_MARGIN
) -> np.ndarray:
    """Boltzmann targets over do-nothing and the actions that beat it by more than `margin`.

    The support is do-nothing plus every valid action with rho_a < rho_dn - margin, the Tutor's
    rule for preferring a topology action (`select_action`). Within it, p(a|s) ∝ exp(-rho_a / T),
    with do-nothing scored as rho_dn - margin. So do-nothing keeps mass when the best gain over it
    is small and loses it smoothly as the gain grows, while actions that do not beat it by the
    margin (including the many that simulate to exactly rho_dn) get 0. The argmax is the Tutor's
    executed action. If do-nothing is invalid, all valid actions form the support; if no action is
    valid, the target is do-nothing.

    The temperature is in rho units, not normalised per state: T = 0.01 gives actions within
    ~0.01 of the best noticeable mass in every state.

    Args:
        max_rho: Simulated max rho [n, A], NaN where invalid; index 0 is do-nothing.
        valid: Validity mask [n, A].
        temperature: Softmax temperature in rho units, > 0.
        margin: Gain over do-nothing an action needs to enter the support (the Tutor's
            `do_nothing_margin`).

    Returns:
        Targets [n, A] float32, rows summing to 1.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}; use hard targets instead")
    rho = np.where(valid, max_rho, np.inf).astype(np.float64)
    rho_dn = rho[:, DO_NOTHING_INDEX:DO_NOTHING_INDEX + 1]
    support = rho < rho_dn - margin
    support[:, DO_NOTHING_INDEX] = valid[:, DO_NOTHING_INDEX]
    rho[:, DO_NOTHING_INDEX] -= margin

    # Subtracting the per-state minimum keeps the best weight at exp(0) = 1 (no underflow).
    rho = np.where(support, rho, np.inf)
    rho_min = rho.min(axis=1, keepdims=True)
    empty = ~np.isfinite(rho_min[:, 0])
    logits = -(rho - np.where(np.isfinite(rho_min), rho_min, 0.0)) / temperature
    weights = np.where(support, np.exp(logits), 0.0)
    weights[empty, DO_NOTHING_INDEX] = 1.0
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def from_greedy_tutor(experience_dir: Path, target: str, temperature: float = 0.0) -> ImitationData:
    """Convert `imitation.tutor` files (`results/tutor/*/experience/*.npz`).

    Args:
        experience_dir: Directory of per-chronic `.npz` files.
        target: `"hard"` (one-hot of the executed action) or `"soft"` (see `soft_targets`, with
            the margin the Tutor ran with).
        temperature: Softmax temperature for `"soft"`.

    Returns:
        The dataset in the shared format.
    """
    if target not in (HARD, SOFT):
        raise ValueError(f"Unknown target '{target}', expected '{HARD}' or '{SOFT}'")
    parts: dict[str, list] = {k: [] for k in ("obs", "max_rho", "valid", "action", "chronic", "step")}
    action_space = None
    margins = set()
    for path in sorted(experience_dir.glob("*.npz")):
        with np.load(path) as data:
            action_space = str(data["action_space"])
            margins.add(float(data["do_nothing_margin"]))
            if len(data["action"]) == 0:
                continue
            for key in ("obs", "max_rho", "valid", "action", "step"):
                parts[key].append(data[key])
            parts["chronic"].append(np.full(len(data["action"]), str(data["chronic"])))
    if len(margins) != 1:
        raise ValueError(f"Expected one do_nothing_margin across all chronics, found {sorted(margins)}")
    margin = margins.pop()
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    n_actions = arrays["max_rho"].shape[1]

    if target == HARD:
        policy_target = one_hot(arrays["action"].astype(np.int64), n_actions)
    else:
        policy_target = soft_targets(arrays["max_rho"], arrays["valid"], temperature, margin)

    # When no action is valid the Tutor falls back to do-nothing; keep that target unmasked so a
    # masked policy can still assign it probability.
    action_mask = arrays["valid"] | (policy_target > 0)

    return ImitationData(
        obs=arrays["obs"],
        policy_target=policy_target,
        action_mask=action_mask,
        chronic=arrays["chronic"],
        step=arrays["step"].astype(np.int32),
        max_rho=arrays["max_rho"],
        metadata={"source": GREEDY_TUTOR, "target": target, "temperature": float(temperature),
                  "do_nothing_margin": margin, "action_space": action_space, "n_actions": n_actions},
    )


def chronic_group(chronic: str) -> str:
    """Stratification group of a chronic: its month for `Scenario_<month>_<id>` names, else the name."""
    parts = Path(chronic).name.split("_")
    return parts[1] if len(parts) == 3 and parts[0] == "Scenario" else Path(chronic).name


def make_split(chronics: list[str], fractions: tuple[float, float, float], seed: int) -> dict[str, list[str]]:
    """Assign chronics to train/val/test, stratified by month so every split covers all seasons.

    Args:
        chronics: Chronic names or paths.
        fractions: Shares of (train, val, test); the remainder after rounding goes to train.
        seed: Seed of the shuffle within each month.

    Returns:
        Split name -> sorted chronic names.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1, got {fractions}")
    groups: dict[str, list[str]] = defaultdict(list)
    for chronic in sorted({Path(c).name for c in chronics}):
        groups[chronic_group(chronic)].append(chronic)
    rng = np.random.default_rng(seed)
    split: dict[str, list[str]] = {name: [] for name in SPLITS}
    for members in groups.values():
        members = list(rng.permutation(members))
        n_val = round(fractions[1] * len(members))
        n_test = round(fractions[2] * len(members))
        split["val"] += members[:n_val]
        split["test"] += members[n_val:n_val + n_test]
        split["train"] += members[n_val + n_test:]
    return {name: sorted(members) for name, members in split.items()}


def save_split(split: dict[str, list[str]], path: Path) -> None:
    """Write a split as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1))


def load_split(path: Path) -> dict[str, list[str]]:
    """Read a split written by `save_split`."""
    return json.loads(Path(path).read_text())


def split_indices(chronic: np.ndarray, split: dict[str, list[str]], name: str) -> np.ndarray:
    """Indices of the samples whose chronic belongs to split `name`."""
    return np.flatnonzero(np.isin(chronic, split[name]))
