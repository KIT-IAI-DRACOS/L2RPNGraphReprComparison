# Configuration Overview

This folder contains all configuration files used by **[Hydra](https://hydra.cc/)**, the configuration framework that drives the training pipeline. Hydra composes configurations at runtime from multiple YAML files, enabling modular overrides without touching source code.

## Folder Structure

```
configs/
└── rllib/               # RL training configurations (composed by experiments/train.py)
    ├── config.yaml      # Root config — declares defaults for all groups below
    ├── experiment/
    ├── training/
    ├── model/
    ├── obs_space/
    ├── env/
    ├── reward/
    ├── rollouts/
    ├── callbacks/
    ├── evaluation/
    ├── optimization/
    └── opponent/
```

## `rllib/` — RL Training Configs

The `rllib/` subfolder is loaded by `experiments/train.py`. The root `config.yaml` sets default selections for every group; individual groups can be swapped via CLI:

```bash
python experiments/train.py training=ppo model=gnn env=case14
```

Each subfolder configures one modular piece of the pipeline:

---

### `experiment/`
Trial-level settings: total training timesteps, max episode length, checkpoint frequency, and whether to run post-training evaluation.

| File | Purpose |
|---|---|
| `default.yaml` | Standard run (200k timesteps) |
| `long.yaml` | Extended run (120M timesteps) with less frequent checkpoints |
| `test_minimal.yaml` | Smoke test (10 timesteps, local Ray mode) — use this to verify the setup |
| `hpc_test.yaml` | Short HPC smoke test (6k timesteps) without post-training evaluation |

---

### `training/`
Algorithm hyperparameters for the RLlib trainer. Pick one algorithm; shared settings live in `base.yaml`.

| File | Purpose |
|---|---|
| `base.yaml` | Shared settings (batch mode, step counting, policy mapping) |
| `ppo.yaml` | PPO (lr=1e-4, clip=0.3, 5 SGD iterations per batch) |
| `sac.yaml` | SAC (entropy regularisation, twin Q-networks, 1M replay buffer) |
| `dqn.yaml` | Rainbow DQN (dueling, double-Q, n-step=3, C51, noisy nets, PER) |
| `ppo_test.yaml` | PPO with reduced batch sizes for quick testing |
| `sac_test.yaml` | SAC with reduced batch sizes for quick testing |
| `dqn_test.yaml` | DQN with reduced batch sizes for quick testing |

---

### `model/`
Neural network architecture for the RL policy. Controls whether a graph encoder and GNN are used.

| File | Purpose |
|---|---|
| `gnn.yaml` | GNN policy — message-passing over the power grid graph. |
| `mlp.yaml` | Plain MLP baseline — flat observation, three hidden layers of 256 units, no graph structure. |

---

### `obs_space/`
Observation format fed to the policy network.

| File | Purpose |
|---|---|
| `graph.yaml` | Graph-structured observation: node feature matrix `[B, N, F]`, edge index, edge mask. Required for GNN models. |
| `flat.yaml` | Flattened Box observation. Required for the MLP model. |

---

### `env/`
Grid2Op environment definition: which power grid case to use, which action space file, and the line-overload threshold that triggers episode termination.

| File | Purpose                                                                |
|---|------------------------------------------------------------------------|
| `case14.yaml` | IEEE 14-bus test case, ρ threshold 0.95, requires `medha` action space |
| `case36.yaml` | IEEE 36-bus WCCI 2020 case, requires `rl2grid_bus36` action space      |
| `case118.yaml` | IEEE 118-bus WCCI 2022 case, requires `rl2grid_bus118` action space    |

---

### `reward/`
Reward function passed to the Grid2Op environment.

| File | Purpose |
|---|---|
| `rho.yaml` | Reward based on maximum line loading (ρ metric) |
| `scaled_l2rpn.yaml` | Scaled version of the official L2RPN challenge reward |
| `constant.yaml` | Constant reward — useful for isolating structural learning from reward shaping |
| `alpha_zero.yaml` | AlphaZero-style sparse reward |

---

### `rollouts/`
Ray worker topology: number of rollout workers, learner workers, and GPU allocation.

| File | Purpose                                                                                   |
|---|-------------------------------------------------------------------------------------------|
| `default.yaml` | Production cluster setup: 48 rollout workers, 4 learner workers, 8 eval workers           |
| `hpc_test.yaml` | Lightweight HPC test: 4 rollout workers, 1 learner worker (1 GPU)                         |
| `test_minimal.yaml` | Single CPU worker for local testing                                                       |

---

### `callbacks/`
Registers custom RLlib callbacks that run during training.

| File | Purpose |
|---|---|
| `default.yaml` | Enables `CustomMetricsCallback`, `TimerCallback`, and `TuneCallback` (logs custom metrics for Ray Tune) |

---

### `evaluation/`
In-training evaluation settings.

| File | Purpose |
|---|---|
| `default.yaml` | Evaluate every 10 episodes using 8 dedicated eval workers |
| `test_minimal.yaml` | Evaluate every episode using 1 worker (for quick tests) |

---

### `optimization/`
Hyperparameter search via Optuna (optional).

| File | Purpose |
|---|---|
| `optuna.yaml` | Run 50 Optuna trials sweeping over model and training hyperparameters |
| `disabled.yaml` | No sweep — use the fixed configuration as-is |

---

### `opponent/`
Adversarial line-attack opponent in the environment (optional).

| File | Purpose |
|---|---|
| `disabled.yaml` | No opponent — standard single-agent setting |
| `random_line.yaml` | `RandomLineOpponent` that randomly disconnects one of 5 specified transmission lines, subject to a cooldown and attack budget |
