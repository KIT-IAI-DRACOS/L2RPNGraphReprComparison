"""
Imitation pretraining of the RL policy network: cross-entropy between the policy's logits and the
dataset's `policy_target` (one-hot for hard labels, a distribution for soft labels).

Model selection uses `kl` = cross-entropy minus the entropy of the targets, i.e. KL(target || policy):
the part of the loss the network can still reduce. It is comparable between hard labels (entropy 0)
and soft labels, whose cross-entropy can never fall below their entropy.

The network is built through RLlib's ModelCatalog exactly as PPO builds the RL_POLICY model, so
the trained `state_dict` loads into the policy unchanged (`core.callback.ImitationInitCallback`).
Only the policy logits are trained; the value branch is left to the critic warm-up in RL.
"""
from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch.utils.data import DataLoader, Dataset

from imitation.tutor import DO_NOTHING_INDEX

logger = logging.getLogger(__name__)

# An action counts as "good" if its simulated max rho is within this of the best valid action.
GOOD_ACTION_TOLERANCE = 0.01
# k of the top-k accuracies of `evaluate`.
TOP_K = (5, 20)


@dataclass
class PretrainConfig:
    """Optimisation settings.

    Attributes:
        epochs: Maximum epochs.
        batch_size: Samples per gradient step.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        patience: Stop after this many epochs without a lower validation `monitor` metric.
        device: Torch device.
        seed: Seed for initialisation and shuffling.
        monitor: Validation metric (key of `evaluate`) for early stopping and the best epoch; lower is better.
    """

    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    device: str = "cpu"
    seed: int = 42
    monitor: str = "kl"


# Called after every epoch with (epoch, history row); may raise to stop training (e.g. optuna.TrialPruned).
EpochCallback = Callable[[int, dict[str, float]], None]


def build_policy_model(obs_space: gym.spaces.Space, n_actions: int, model_config: dict) -> TorchModelV2:
    """Build the RL_POLICY network as RLlib's PPO policy does.

    Args:
        obs_space: The RL agent's observation space (the converter's).
        n_actions: Discrete policy actions incl. do-nothing.
        model_config: The `model` dict of the RLlib config, with the PPO custom model name.

    Returns:
        The torch model; its `state_dict` matches the PPO policy's `model`.
    """
    # PPO's model defaults, not RLlib's MODEL_DEFAULTS: PPO sets vf_share_layers=False, which adds a
    # separate value network (`_value_branch_separate`) that the state_dict must contain.
    config = copy.deepcopy(PPOConfig().model)
    config.update(copy.deepcopy(model_config))
    return ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=gym.spaces.Discrete(n_actions),
        num_outputs=n_actions,
        model_config=config,
        framework="torch",
    )


def policy_logits(model: TorchModelV2, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Action logits [B, A] for a batch of converted observations."""
    logits, _ = model.forward({"obs": obs}, [], None)
    return logits


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy against target distributions (equals the usual CE for one-hot targets)."""
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def target_entropy(target: torch.Tensor) -> torch.Tensor:
    """Mean entropy (nats) of target distributions [B, A], with 0 * log 0 = 0; the lower bound of the cross-entropy."""
    return -torch.special.xlogy(target, target).sum(dim=-1).mean()


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (_to_device(v, device) if isinstance(v, dict) else v.to(device)) for k, v in batch.items()}


def valid_rho(max_rho: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Simulated max rho [B, A] with inf for every action that is masked out or whose simulation failed (NaN)."""
    return torch.where(valid & torch.isfinite(max_rho), max_rho, torch.full_like(max_rho, float("inf")))


def regret_top_k(logits: torch.Tensor, rho: torch.Tensor, k: int) -> torch.Tensor:
    """Per-state regret of top-k simulation: the best valid action among the k highest logits.

    Mirrors the agent's top-k simulation without its do-nothing margin. If none of the k actions is
    valid, the agent falls back to do-nothing. If do-nothing is invalid too (the episode would end),
    the state counts as its worst valid action.

    Args:
        logits: Action logits [B, A].
        rho: Simulated max rho [B, A], inf where invalid (see `valid_rho`).
        k: Number of top-ranked actions.

    Returns:
        Regret [B]: max rho of the chosen action minus that of the best valid action. Meaningless
        for states without any valid action (best rho inf); callers must leave them out.
    """
    top = logits.topk(min(k, logits.shape[-1]), dim=-1).indices
    chosen = rho.gather(1, top).min(dim=-1).values
    chosen = torch.where(torch.isfinite(chosen), chosen, rho[:, DO_NOTHING_INDEX])
    worst = torch.where(torch.isfinite(rho), rho, torch.full_like(rho, -float("inf"))).max(dim=-1).values
    chosen = torch.where(torch.isfinite(chosen), chosen, worst)
    return chosen - rho.min(dim=-1).values


@torch.no_grad()
def evaluate(model: TorchModelV2, dataset: Dataset, batch_size: int, device: torch.device) -> dict[str, float]:
    """Validation metrics.

    Always: loss (cross-entropy), entropy (of the targets), kl (loss - entropy), accuracy (argmax =
    argmax of target), top<k> for k in TOP_K (target argmax among the k highest logits),
    do_nothing_rate (share of argmax = do-nothing). If the dataset has simulated max rho:
    good_action_rate (argmax within GOOD_ACTION_TOLERANCE of the best valid action), regret (mean
    rho excess of the argmax over the best valid action; invalid choices count as missing),
    invalid_rate (argmax not valid), regret_top<k> for k in TOP_K (mean `regret_top_k` over the
    states with a valid action).
    """
    model.eval()
    sums: dict[str, float] = {}
    n_regret = 0
    n_with_valid = 0
    n = 0
    for batch in DataLoader(dataset, batch_size=batch_size):
        batch = _to_device(batch, device)
        logits = policy_logits(model, batch["obs"])
        target = batch["target"]
        pred = logits.argmax(dim=-1)
        best_target = target.argmax(dim=-1)
        hits = logits.topk(min(max(TOP_K), logits.shape[-1]), dim=-1).indices == best_target.unsqueeze(-1)
        b = len(pred)
        stats = {
            "loss": soft_cross_entropy(logits, target).item() * b,
            "entropy": target_entropy(target).item() * b,
            "accuracy": (pred == best_target).sum().item(),
            **{f"top{k}": hits[:, :k].any(dim=-1).sum().item() for k in TOP_K},
            "do_nothing_rate": (pred == DO_NOTHING_INDEX).sum().item(),
        }
        if "max_rho" in batch:
            rho = valid_rho(batch["max_rho"], batch["valid"])
            best = rho.min(dim=-1).values
            chosen = rho.gather(1, pred.unsqueeze(-1)).squeeze(-1)
            has_best = torch.isfinite(best)
            chosen_valid = torch.isfinite(chosen) & has_best
            stats["good_action_rate"] = (chosen_valid & (chosen - best <= GOOD_ACTION_TOLERANCE)).sum().item()
            stats["invalid_rate"] = (~torch.isfinite(chosen) & has_best).sum().item()
            stats["regret"] = (chosen - best)[chosen_valid].sum().item()
            for k in TOP_K:
                stats[f"regret_top{k}"] = regret_top_k(logits, rho, k)[has_best].sum().item()
            n_regret += int(chosen_valid.sum().item())
            n_with_valid += int(has_best.sum().item())
        for key, value in stats.items():
            sums[key] = sums.get(key, 0.0) + value
        n += b
    regret_counts = {"regret": n_regret, **{f"regret_top{k}": n_with_valid for k in TOP_K}}
    metrics = {k: v / max(regret_counts.get(k, n), 1) for k, v in sums.items()}
    metrics["kl"] = metrics["loss"] - metrics["entropy"]
    return metrics


@torch.no_grad()
def predict(model: TorchModelV2, dataset: Dataset, batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    """Per-state predictions of a trained model, for analyses beyond the means of `evaluate`.

    Returns:
        Arrays of length n:
            tutor_action: argmax of the target (the Tutor's executed action for greedy-Tutor data).
            pred: argmax of the logits.
            tutor_rank: rank of tutor_action among the logits (1 = highest), for top-k accuracies.
            p_tutor, p_pred: policy probability of tutor_action and of pred.
        If the dataset has simulated max rho (invalid actions are inf):
            rho_pred, rho_tutor, rho_best: simulated max rho of pred, of tutor_action and of the best
                valid action.
            good_rank: highest rank of an action within GOOD_ACTION_TOLERANCE of the best valid action;
                inf if no action is valid.
    """
    model.eval()
    out: dict[str, list[torch.Tensor]] = {}
    for batch in DataLoader(dataset, batch_size=batch_size):
        batch = _to_device(batch, device)
        logits = policy_logits(model, batch["obs"])
        probs = torch.softmax(logits, dim=-1)
        tutor = batch["target"].argmax(dim=-1, keepdim=True)
        pred = logits.argmax(dim=-1, keepdim=True)
        # rank[i, a] = 1 + number of actions with a higher logit than a (ties share the better rank)
        rank = (logits.unsqueeze(-1) < logits.unsqueeze(-2)).sum(dim=-1) + 1
        rows = {
            "tutor_action": tutor,
            "pred": pred,
            "tutor_rank": rank.gather(1, tutor),
            "p_tutor": probs.gather(1, tutor),
            "p_pred": probs.gather(1, pred),
        }
        if "max_rho" in batch:
            rho = torch.where(batch["valid"], batch["max_rho"], torch.full_like(batch["max_rho"], float("inf")))
            best = rho.min(dim=-1, keepdim=True).values
            good = torch.isfinite(rho) & (rho - best <= GOOD_ACTION_TOLERANCE)
            rows.update({
                "rho_pred": rho.gather(1, pred),
                "rho_tutor": rho.gather(1, tutor),
                "rho_best": best,
                "good_rank": torch.where(good, rank.float(), torch.full_like(rho, float("inf"))).min(dim=-1, keepdim=True).values,
            })
        for key, value in rows.items():
            out.setdefault(key, []).append(value.squeeze(-1).cpu())
    return {key: torch.cat(values).numpy() for key, values in out.items()}


def train_epoch(model: TorchModelV2, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """One pass over the training data.

    Returns:
        Mean training cross-entropy of the epoch.
    """
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        loss = soft_cross_entropy(policy_logits(model, batch["obs"]), batch["target"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch["target"])
        count += len(batch["target"])
    return total / count


def train(
    model: TorchModelV2,
    train_set: Dataset,
    val_set: Dataset,
    config: PretrainConfig,
    epoch_callback: EpochCallback | None = None,
) -> tuple[dict, list[dict]]:
    """Train with early stopping on the validation metric `config.monitor`.

    Args:
        model: Policy network, as from `build_policy_model`.
        train_set: Samples as from `ImitationTorchDataset`.
        val_set: Validation samples, for early stopping and the best epoch.
        config: Optimisation settings.
        epoch_callback: Called after every epoch with (epoch, history row), e.g. to report to Optuna.

    Returns:
        `(best_state_dict, history)`; history has one dict per epoch with train_loss and `val_*` metrics.
        The model is left with the best state loaded.
    """
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, generator=generator)

    best_value, best_state, epochs_without_gain = np.inf, copy.deepcopy(model.state_dict()), 0
    history = []
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, device)
        val = evaluate(model, val_set, config.batch_size, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        logger.info(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in history[-1].items()))
        if epoch_callback is not None:
            epoch_callback(epoch, history[-1])

        if val[config.monitor] < best_value:
            best_value, best_state, epochs_without_gain = val[config.monitor], copy.deepcopy(model.state_dict()), 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= config.patience:
                logger.info(f"Early stop after epoch {epoch}: no lower validation {config.monitor} for {config.patience} epochs")
                break
    model.load_state_dict(best_state)
    return best_state, history
