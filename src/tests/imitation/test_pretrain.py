"""
Unit tests for imitation/pretrain.py, imitation/torch_dataset.py and the critic warm-up of
core/callback.py.

Tests are organised by class:
  - TestSoftCrossEntropy
  - TestTargetEntropy
  - TestRegretTopK
  - TestEvaluate
  - TestPredict
  - TestTrain
  - TestTrimEdges
  - TestCriticWarmup
  - TestPPOCompatibility
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy

from core.rllib_config import register_custom_models
from grid2op_env.observation_converter import EDGE_INDEX, EDGE_MASK, GLOBAL, NODE_MASK, NODES
from imitation.pretrain import (
    PretrainConfig,
    build_policy_model,
    evaluate,
    policy_logits,
    predict,
    regret_top_k,
    soft_cross_entropy,
    target_entropy,
    train,
    valid_rho,
)
from imitation.torch_dataset import _trim_edges
from core.callback import is_value_parameter, set_policy_trainable

N_NODES, N_EDGES, N_ACTIONS = 6, 10, 4


def graph_obs_space() -> gym.spaces.Dict:
    return gym.spaces.Dict({
        NODES: gym.spaces.Box(-np.inf, np.inf, shape=(N_NODES, 3), dtype=np.float32),
        EDGE_INDEX: gym.spaces.Box(0, N_NODES - 1, shape=(2, N_EDGES), dtype=np.int64),
        EDGE_MASK: gym.spaces.Box(0, 1, shape=(N_EDGES,), dtype=np.bool_),
        NODE_MASK: gym.spaces.Box(0, 1, shape=(N_NODES,), dtype=np.bool_),
        GLOBAL: gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
    })


def graph_batch(batch_size: int) -> dict[str, torch.Tensor]:
    edge_index = torch.randint(0, N_NODES, (batch_size, 2, N_EDGES))
    edge_mask = torch.zeros(batch_size, N_EDGES, dtype=torch.bool)
    edge_mask[:, :7] = True
    return {
        NODES: torch.randn(batch_size, N_NODES, 3),
        EDGE_INDEX: edge_index,
        EDGE_MASK: edge_mask,
        NODE_MASK: torch.ones(batch_size, N_NODES, dtype=torch.bool),
        GLOBAL: torch.randn(batch_size, 6),
    }


GNN_MODEL_CONFIG = {
    "custom_model": "gnn_model",
    "fcnet_hiddens": [16],
    "fcnet_activation": "relu",
    "post_fcnet_hiddens": [],
    "custom_model_config": {"gnn": {"hidden_dim": 8, "out_dim": 8, "num_layers": 2, "conv_type": "gcn"}},
}


def gnn_model():
    register_custom_models()
    return build_policy_model(graph_obs_space(), N_ACTIONS, GNN_MODEL_CONFIG)


class TestSoftCrossEntropy(unittest.TestCase):
    def test_equals_cross_entropy_for_one_hot_targets(self):
        logits = torch.randn(5, N_ACTIONS)
        labels = torch.tensor([0, 1, 2, 3, 1])
        expected = F.cross_entropy(logits, labels)
        actual = soft_cross_entropy(logits, F.one_hot(labels, N_ACTIONS).float())
        self.assertAlmostEqual(actual.item(), expected.item(), places=5)


class TestTargetEntropy(unittest.TestCase):
    def test_one_hot_targets_have_zero_entropy(self):
        self.assertEqual(target_entropy(F.one_hot(torch.tensor([0, 3]), N_ACTIONS).float()).item(), 0.0)

    def test_uniform_target_over_k_actions_has_entropy_log_k(self):
        target = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        self.assertAlmostEqual(target_entropy(target).item(), np.log(2), places=6)

    def test_cross_entropy_minus_entropy_is_zero_for_matching_logits(self):
        target = torch.tensor([[0.7, 0.2, 0.1, 0.0]])
        logits = torch.log(target.clamp_min(1e-12))
        kl = soft_cross_entropy(logits, target) - target_entropy(target)
        self.assertAlmostEqual(kl.item(), 0.0, places=5)


class _FixedLogits(torch.nn.Module):
    """Stands in for a policy model: logits are passed in as the observation."""

    def forward(self, input_dict, state, seq_lens):
        return input_dict["obs"]["logits"], state


class TestRegretTopK(unittest.TestCase):
    def test_best_valid_action_among_top_k_with_fallbacks(self):
        inf = float("inf")
        # Logits rank the actions 4 > 3 > 2 > 1 > 0 in every state, so k = 2 simulates actions 4 and 3.
        logits = torch.arange(5, dtype=torch.float32).repeat(4, 1)
        rho = torch.tensor([
            [0.9, 0.95, 0.8, 0.7, 0.75],  # best valid action (3) among the top 2 -> 0
            [0.9, 0.6, 0.8, 0.85, 0.95],  # top 2 valid, best is action 1 -> 0.85 - 0.6
            [0.9, 0.6, 0.8, inf, inf],    # top 2 invalid -> do-nothing: 0.9 - 0.6
            [inf, 0.6, 0.8, inf, inf],    # top 2 and do-nothing invalid -> worst valid action: 0.8 - 0.6
        ])
        torch.testing.assert_close(regret_top_k(logits, rho, k=2), torch.tensor([0.0, 0.25, 0.3, 0.2]))

    def test_valid_rho_marks_masked_and_failed_simulations_as_inf(self):
        max_rho = torch.tensor([[0.9, float("nan"), 0.8]])
        valid = torch.tensor([[True, True, False]])
        torch.testing.assert_close(valid_rho(max_rho, valid), torch.tensor([[0.9, float("inf"), float("inf")]]))


class TestEvaluate(unittest.TestCase):
    def test_regret_top_k_skips_states_without_valid_action(self):
        # Logits rank action a at position a + 1. State 0: rho falls with the action index, best is
        # action 24 (0.76); top 5 reach action 4 (0.96), top 20 action 19 (0.81). State 1: nothing valid.
        n = 25
        logits = torch.arange(n, dtype=torch.float32).flip(0).repeat(2, 1)
        target = torch.zeros(2, n)
        target[:, 0] = 1.0
        max_rho = torch.stack([1.0 - 0.01 * torch.arange(n, dtype=torch.float32), torch.full((n,), float("nan"))])
        valid = torch.stack([torch.ones(n, dtype=torch.bool), torch.arange(n) == 0])
        dataset = [{"obs": {"logits": logits[i]}, "target": target[i], "max_rho": max_rho[i], "valid": valid[i]}
                   for i in range(2)]
        metrics = evaluate(_FixedLogits(), dataset, batch_size=2, device=torch.device("cpu"))
        self.assertAlmostEqual(metrics["regret_top5"], 0.2, places=5)
        self.assertAlmostEqual(metrics["regret_top20"], 0.05, places=5)

    def test_regret_and_good_action_rate(self):
        # Sample 0 picks action 1 (best). Sample 1 picks action 2: valid, 0.05 worse than best.
        # Sample 2 picks invalid action 3.
        logits = torch.tensor([[0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 5.0]])
        target = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        max_rho = torch.tensor([[0.9, 0.8, 0.85, 0.9], [0.9, 0.80, 0.85, 0.9], [0.9, 0.9, 0.9, float("nan")]])
        valid = torch.tensor([[True] * 4, [True] * 4, [True, True, True, False]])
        dataset = [{"obs": {"logits": logits[i]}, "target": target[i], "max_rho": max_rho[i], "valid": valid[i]}
                   for i in range(3)]
        metrics = evaluate(_FixedLogits(), dataset, batch_size=2, device=torch.device("cpu"))
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["good_action_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["invalid_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["regret"], 0.025, places=5)
        self.assertEqual(metrics["do_nothing_rate"], 0.0)

    def test_kl_is_loss_minus_target_entropy(self):
        logits = torch.randn(3, N_ACTIONS)
        target = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
        dataset = [{"obs": {"logits": logits[i]}, "target": target[i]} for i in range(3)]
        metrics = evaluate(_FixedLogits(), dataset, batch_size=2, device=torch.device("cpu"))
        self.assertAlmostEqual(metrics["entropy"], target_entropy(target).item(), places=5)
        self.assertAlmostEqual(metrics["kl"], metrics["loss"] - metrics["entropy"], places=6)
        self.assertGreaterEqual(metrics["kl"], 0.0)

    def test_top_k_counts_target_argmax_among_k_highest_logits(self):
        # Target argmax is action 0; it ranks 1st, 3rd, 6th and 21st (25 actions).
        logits = torch.arange(25, dtype=torch.float32).flip(0).repeat(4, 1)
        target = torch.zeros(4, 25)
        for i, rank in enumerate([1, 3, 6, 21]):
            target[i, rank - 1] = 1.0
        dataset = [{"obs": {"logits": logits[i]}, "target": target[i]} for i in range(4)]
        metrics = evaluate(_FixedLogits(), dataset, batch_size=4, device=torch.device("cpu"))
        self.assertAlmostEqual(metrics["accuracy"], 1 / 4)
        self.assertAlmostEqual(metrics["top5"], 2 / 4)
        self.assertAlmostEqual(metrics["top20"], 3 / 4)


class TestPredict(unittest.TestCase):
    def setUp(self):
        # Sample 0: Tutor action 1 has the highest logit. Sample 1: Tutor action 1 ranks 2nd behind
        # action 2, which is 0.05 worse; action 3 (rank 3) is within tolerance of the best.
        # Sample 2: no valid action.
        self.logits = torch.tensor([[0.0, 3.0, 1.0, 2.0], [0.0, 2.0, 3.0, 1.0], [3.0, 2.0, 1.0, 0.0]])
        target = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        max_rho = torch.tensor([[0.9, 0.8, 0.85, 0.9], [0.9, 0.80, 0.85, 0.805], [0.9, 0.9, 0.9, 0.9]])
        valid = torch.tensor([[True] * 4, [True] * 4, [False] * 4])
        self.dataset = [{"obs": {"logits": self.logits[i]}, "target": target[i], "max_rho": max_rho[i],
                         "valid": valid[i]} for i in range(3)]
        self.out = predict(_FixedLogits(), self.dataset, batch_size=2, device=torch.device("cpu"))

    def test_ranks_and_probabilities(self):
        np.testing.assert_array_equal(self.out["tutor_action"], [1, 1, 0])
        np.testing.assert_array_equal(self.out["pred"], [1, 2, 0])
        np.testing.assert_array_equal(self.out["tutor_rank"], [1, 2, 1])
        probs = torch.softmax(self.logits, dim=-1).numpy()
        np.testing.assert_allclose(self.out["p_tutor"], probs[[0, 1, 2], [1, 1, 0]], rtol=1e-6)
        np.testing.assert_allclose(self.out["p_pred"], probs.max(axis=-1), rtol=1e-6)

    def test_simulated_outcomes(self):
        np.testing.assert_allclose(self.out["rho_pred"][:2], [0.8, 0.85], rtol=1e-6)
        np.testing.assert_allclose(self.out["rho_tutor"][:2], [0.8, 0.8], rtol=1e-6)
        np.testing.assert_allclose(self.out["rho_best"][:2], [0.8, 0.8], rtol=1e-6)
        np.testing.assert_array_equal(self.out["good_rank"], [1, 2, np.inf])
        self.assertTrue(np.isinf(self.out["rho_best"][2]))

    def test_without_max_rho_has_only_policy_fields(self):
        dataset = [{k: v for k, v in item.items() if k in ("obs", "target")} for item in self.dataset]
        out = predict(_FixedLogits(), dataset, batch_size=2, device=torch.device("cpu"))
        self.assertEqual(set(out), {"tutor_action", "pred", "tutor_rank", "p_tutor", "p_pred"})


def graph_dataset(n: int) -> list[dict]:
    """Samples in the format of ImitationTorchDataset, with random graph observations and soft targets."""
    obs = graph_batch(n)
    target = torch.softmax(torch.randn(n, N_ACTIONS), dim=-1)
    return [{"obs": {k: v[i] for k, v in obs.items()}, "target": target[i]} for i in range(n)]


class TestTrain(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.train_set, self.val_set = graph_dataset(16), graph_dataset(8)

    def test_callback_gets_every_epoch_and_history_has_kl(self):
        seen = []
        config = PretrainConfig(epochs=3, batch_size=8, patience=10)
        _, history = train(gnn_model(), self.train_set, self.val_set, config, lambda epoch, row: seen.append(epoch))
        self.assertEqual(seen, [1, 2, 3])
        self.assertEqual(len(history), 3)
        self.assertIn("val_kl", history[0])

    def test_callback_exception_stops_training(self):
        def stop(epoch, row):
            if epoch == 2:
                raise StopIteration

        with self.assertRaises(StopIteration):
            train(gnn_model(), self.train_set, self.val_set, PretrainConfig(epochs=5, batch_size=8), stop)

    def test_keeps_weights_of_best_monitored_epoch(self):
        model = gnn_model()
        config = PretrainConfig(epochs=4, batch_size=8, patience=10, lr=1e-2)
        best_state, history = train(model, self.train_set, self.val_set, config)
        best_kl = min(row["val_kl"] for row in history)
        final = evaluate(model, self.val_set, batch_size=8, device=torch.device("cpu"))
        self.assertAlmostEqual(final["kl"], best_kl, places=5)


class TestTrimEdges(unittest.TestCase):
    def test_trims_edge_axes_to_largest_edge_count(self):
        mask = np.zeros((2, 6), dtype=bool)
        mask[0, :3] = True
        mask[1, :2] = True
        arrays = {EDGE_MASK: mask, EDGE_INDEX: np.zeros((2, 2, 6), dtype=np.int32), NODES: np.zeros((2, 6, 3))}
        trimmed = _trim_edges(arrays)
        self.assertEqual(trimmed[EDGE_MASK].shape, (2, 3))
        self.assertEqual(trimmed[EDGE_INDEX].shape, (2, 2, 3))
        self.assertEqual(trimmed[NODES].shape, (2, 6, 3))


class TestCriticWarmup(unittest.TestCase):
    """During the warm-up only the value branch may change."""

    def test_frozen_policy_keeps_logits_while_value_learns(self):
        torch.manual_seed(0)
        model = gnn_model()
        self.assertTrue(any(is_value_parameter(n) for n, _ in model.named_parameters()))
        set_policy_trainable(SimpleNamespace(model=model), False)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        obs = graph_batch(8)
        logits = policy_logits(model, obs)
        loss = model.value_function().pow(2).mean() + logits.sum() * 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for name, param in model.named_parameters():
            changed = not torch.equal(before[name], param.detach())
            self.assertEqual(changed, is_value_parameter(name), name)

        set_policy_trainable(SimpleNamespace(model=model), True)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_frozen_policy_trains_value_layer_of_rllibs_default_model(self):
        """RLlib's ComplexInputNetwork (flat MLP on a Dict observation) has its value head in `value_layer`."""
        torch.manual_seed(0)
        obs_space = gym.spaces.Dict({"a": gym.spaces.Box(-1, 1, (6,)), "b": gym.spaces.Box(-1, 1, (4,))})
        model = build_policy_model(obs_space, 5, {"custom_model": None, "fcnet_hiddens": [16, 16]})
        set_policy_trainable(SimpleNamespace(model=model), False)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}

        obs = {key: torch.rand(8, *space.shape) for key, space in obs_space.spaces.items()}
        logits_before = policy_logits(model, obs).detach()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        loss = model.value_function().pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        changed = {n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())}
        self.assertTrue(changed, "the value head did not train")
        self.assertTrue(all(n.startswith("value_layer") for n in changed), changed)
        self.assertTrue(torch.equal(policy_logits(model, obs).detach(), logits_before))

    def test_frozen_batch_norm_matches_sampling_in_train_mode(self):
        """While frozen, a train-mode forward (RLlib's updates) equals the eval-mode one (sampling)
        and leaves the running statistics unchanged; unfreezing restores normal BatchNorm training."""
        torch.manual_seed(0)
        model = gnn_model()
        batch_norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        self.assertTrue(batch_norms)
        obs = graph_batch(8)
        model.eval()
        sampled = policy_logits(model, obs).detach()
        stats = [m.running_mean.clone() for m in batch_norms]

        set_policy_trainable(SimpleNamespace(model=model), False)
        model.train()
        torch.testing.assert_close(policy_logits(model, obs).detach(), sampled)
        for before, m in zip(stats, batch_norms):
            torch.testing.assert_close(m.running_mean, before)

        set_policy_trainable(SimpleNamespace(model=model), True, freeze_batch_norm=True)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        model.train()
        torch.testing.assert_close(policy_logits(model, obs).detach(), sampled)

        set_policy_trainable(SimpleNamespace(model=model), True)
        model.train()
        policy_logits(model, obs)
        self.assertTrue(any(not torch.equal(m.running_mean, s) for m, s in zip(batch_norms, stats)))


class TestPPOCompatibility(unittest.TestCase):
    """Pretrained weights must load strictly into the model PPO builds (ImitationInitCallback)."""

    def test_state_dict_loads_into_ppo_policy_model(self):
        pretrained = gnn_model()
        config = PPOConfig().framework("torch").training(model=GNN_MODEL_CONFIG).to_dict()
        config["_disable_preprocessor_api"] = True  # as in experiments/train.py
        policy = PPOTorchPolicy(graph_obs_space(), gym.spaces.Discrete(N_ACTIONS), config)
        policy.model.load_state_dict(pretrained.state_dict())


if __name__ == "__main__":
    unittest.main()
