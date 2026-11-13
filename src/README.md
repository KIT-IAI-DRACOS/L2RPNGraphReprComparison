# Source Package Overview

All Python source code lives here. Add this directory to `PYTHONPATH` when running scripts
from the project root:

```bash
PYTHONPATH=$(pwd)/src python experiments/train.py
```

---

## Data Flow

```
Grid2Op environment
      │  raw observation (BaseObservation)
      ▼
grid2op_env/observation_converter.py
      │  graph dict: {NODES [N,F], EDGE_INDEX, EDGE_MASK}
      ▼
rl/gnn_ppo_model                 RLlib policy wrapper
      │  logits / Q-values / action distribution
      ▼
grid2op_env/action_converters.py
      │  Grid2Op action
      ▼
Grid2Op environment  (step)
```

---

## Packages

### `grid2op_env/` — Environment wrappers

Bridges Grid2Op and RLlib's `MultiAgentEnv` interface.

| Module | Description |
|---|---|
| `env.py` | `CustomizedGrid2OpEnvironment` — the RLlib `MultiAgentEnv`; routes steps to three agents |
| `observation_converter.py` | Converters of all representations (Default/Complete, Heterogeneous, Substation, Elements, PTDF, LODF, Zbus, hybrids, no edges, flat); they normalise online with a running mean/variance |
| `action_converters.py` | Maps discrete RLlib actions to Grid2Op topology actions |
| `rewards.py` | Custom reward functions (`rho`, `scaled_l2rpn`, `constant`, `alpha_zero`) |
| `multi_agent_policies/do_nothing_policy.py` | DO_NOTHING agent — always returns the no-op action |
| `multi_agent_policies/select_agent_policy.py` | HIGH_LEVEL agent — decides whether the RL agent or do-nothing agent acts based on ρ threshold |

---

### `algorithms/` — Custom RLlib algorithm classes

Thin subclasses of RLlib's `PPO` and `Optuna`. These subclasses exist for three
reasons: (1) fix an RLlib bug where `load_checkpoint` ignores `policy_ids` and crashes
when only a subset of policies were checkpointed; (2) store `my_log_level` and curriculum
training state from the config; (3) `CustomPPO` additionally overrides
`training_step` with a corrected `custom_synchronous_parallel_sample` that counts steps
per trainable policy only, giving an accurate `train_batch_size` in the multi-agent setup.

| Module | Description |
|---|---|
| `custom_ppo.py` | `CustomPPO` — accurate batch sizing + checkpoint bugfix |
| `optuna_search.py` | Optuna integration for hyperparameter search via Ray Tune |

---

### `core/` — Training infrastructure

| Module | Description |
|---|---|
| `constants.py` | Global output paths (`LOGS_PATH`, `MODELS_PATH`, `EVAL_PATH`), agent name strings, policy name constants |
| `rllib_config.py` | Hydra config → RLlib env and model config, model registration (shared by RL training and imitation) |
| `callback.py` | RLlib callbacks: custom metrics, timing, `ImitationInitCallback` (Senior start from Junior weights) |
| `train.py` | `run_training` — builds the Ray Tune `Tuner`, runs it, prints checkpoint summary, triggers post-training evaluation |
| `evaluate.py` | `evaluate_rllib_checkpoint` — loads a checkpoint and runs evaluation episodes |
| `loading.py` | Checkpoint loading helpers (resolves paths, restores RLlib algorithm state) |
| `utils.py` | Miscellaneous utilities |
| `heuristic_actions.py` | Heuristic action selection logic used by some baseline agents |
| `graph.py` | Graph utility functions |

---

### `agents/` — Agent wrappers

High-level agent classes used by the multiagent setup in this repo.

| Module | Description |
|---|---|
| `rllib_agent.py` | `RllibAgent` — wraps a loaded RLlib policy for step-by-step interaction with a Grid2Op environment |
| `rho_greedy_agent.py` | Greedy agent that always picks the action minimising maximum ρ |
| `heuristic_agent.py` | Rule-based agent for baseline comparison |

---

### `analysis/` — Post-training analysis

| Package                  | Description                                                                                                                                                   |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `analyzers/`             | Analyze agent behavior / properties after training (survival, actions, congestion)                                                                            |
| `episodes.py`            | Shared per-episode tables and survival metrics over seeds (36-bus notebook)                                                                                    |
| `training_curves.py`     | Training/evaluation curves of a Tune trial from TensorBoard, cached as CSV                                                                                     |

---

### `rl/` — Policy networks

| Module | Description |
|---|---|
| `gnn.py` | GCN/GAT backbone with edge types and edge weights |
| `readout.py` | Mean and per-substation readouts |
| `gnn_ppo_model.py` | `GNNModel`, the RLlib TorchModelV2 of the GNN policy |

---

### `action_space_generation/` — N-1 Teacher

Wrapper around the N-1 Teacher of Fraunhofer's `curriculumagent` (separate conda env,
`environment_curriculum.yaml`): line selection, sharded Teacher runs, aggregation of the experience
into an action-space JSON (`experiments/build_action_space.py`).

---

### `imitation/` — Tutor, Junior and Senior

| Module | Description |
|---|---|
| `tutor.py` | Greedy Tutor: simulates every policy action in critical states and records the outcomes |
| `dataset.py` | Shared imitation dataset format, hard/soft targets, chronic-level split |
| `torch_dataset.py` | Converts the stored `obs.to_vect()` into the converter's observations (cached) |
| `pretrain.py`, `pretrain_run.py` | Junior pretraining (losses, metrics, early stopping, run directory) |
| `hpo.py` | Optuna study over pretraining settings |
| `rollout.py` | Greedy / top-k rollouts of a pretrained or exported policy |
| `senior.py` | Hydra overrides that rebuild a Junior's network for RL, best-val checkpoint selection |

---

### `visualization/` — Plotting utilities

Standalone scripts for visualizing training results. Not part of
the main training loop; run interactively or via Jupyter.

---

### `tests/`

Unit and integration tests, run with `pytest`:

```bash
python -m pytest src/tests/                         # all tests
```

| Subfolder        | Coverage |
|------------------|---|
| `grid2op_env/`   | Environment, observation converters, rewards, chronic order |
| `imitation/`     | Tutor, datasets, pretraining, HPO, rollouts, Senior helpers |
| `action_space_generation/` | N-1 Teacher wrapper (partly needs `curriculumagent`) |
| `analysis/`      | Episode tables, training curves |
| `visualization/` | Visualisation utilities |
