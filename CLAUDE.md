# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Code of the paper *A Comparative Study of Graph Representations for GNN-Based Power Grid Control in L2RPN*: 12 observation representations (graphs with the same node features, a graph without edges, a flat MLP baseline) under one pipeline. 14 bus: PPO from scratch. 36 bus: N-1 Teacher (action space) → greedy Tutor (imitation data) → Junior (imitation learning, Optuna HPO) → Senior (PPO fine-tuning). See README.md for the reproduction steps.

## Commands

### Environment Setup
```bash
conda env create -f environment.yaml && conda activate L2RPN
python setup_envs.py  # Download the grids and create the paper's train/val/test envs (data/splits/)
```

### Training
```bash
# GCN (per-substation readout) + PPO on the Complete graph (from project root)
PYTHONPATH=$(pwd)/src python experiments/train.py

# Other representations / the MLP baseline
PYTHONPATH=$(pwd)/src python experiments/train.py obs_space=graph_comparison/substation
PYTHONPATH=$(pwd)/src python experiments/train.py model=mlp obs_space=graph_comparison/flat
PYTHONPATH=$(pwd)/src python experiments/train.py --config-name=minimal  # Quick smoke test

# CLI overrides
python experiments/train.py experiment.nb_timesteps=500000 training.lr=3e-4
```

### Tests
```bash
cd src && python -m pytest tests
```

## Architecture

The training pipeline is configured entirely via **Hydra** (configs/rllib/) and orchestrated through `experiments/train.py`.

### Config Groups (configs/rllib/)
- **experiment/**: trial-level settings (timesteps, seed, checkpoint freq)
- **training/**: PPO params — `ppo.yaml`, `ppo_test.yaml`
- **model/**: `gcn_substation_readout.yaml` (GNN policy), `mlp.yaml` (RLlib default MLP, flat baseline)
- **obs_space/graph_comparison/**: the 12 representations, all with `node_features: complete`
- **env/**: Grid2Op case definition (case14, case36)
- **reward/**: reward function class (the paper: `constant` with `normalize_by_episode_length`)
- **imitation/**: Junior pretraining settings and the Senior's start from its weights
- `pretrain_hpo_graph36*.yaml`: primary configs of the Junior HPO

### Data Flow
```
Hydra Config
    │
    ├─► CustomizedGrid2OpEnvironment (MultiAgentEnv)
    │       Three agents: RL_AGENT, HIGH_LEVEL_AGENT, DO_NOTHING_AGENT
    │       ObservationConverter: raw Grid2Op obs → {NODES, EDGE_INDEX, EDGE_MASK}
    │
    └─► GNNModel or MLP (TorchModelV2, src/rl/)
            GNN  → message-passing over power grid graph
            FCNet → policy logits + value head
```

### Key Source Modules
- **src/rl/**: Pure PyTorch — `gnn.py` (GNN backbone), `readout.py` (per-substation readout), `gnn_ppo_model.py` (RLlib TorchModelV2 wrapper), `mlp.py`
- **src/grid2op_env/**: Grid2Op wrappers — multi-agent env, observation/action converters, multi-agent policy implementations
- **src/core/**: `constants.py` (global paths/names), `rllib_config.py` (Hydra → RLlib env/model config, shared by RL and imitation), `train.py` (training setup helpers), `callback.py`, `loading.py` (checkpoint loading), `evaluate.py`
- **src/algorithms/**: `CustomPPO`, Optuna integration
- **src/analysis/**: Post-training analyzers (survival/actions/congestion), `episodes.py` (shared episode tables and survival metrics), `training_curves.py`
- **src/action_space_generation/**: N-1 Teacher (needs `curriculumagent`, env `environment_curriculum.yaml`)
- **src/imitation/**: greedy Tutor, imitation datasets, Junior pretraining and HPO, rollouts, Senior helpers

### Constants (src/core/constants.py)
Defines global output paths (`LOGS_PATH`, `MODELS_PATH`, `EVAL_PATH`), agent name strings (`RL_AGENT`, `HIGH_LEVEL_AGENT`, `DO_NOTHING_AGENT`), and policy names used consistently across the codebase.

### Callbacks
- `CustomMetricsCallback`: collects episode metrics (survival, interact counts, curriculum)
- `TimerCallback`: logs per-step wall-clock timings to TensorBoard
- `ImitationInitCallback`: loads Junior weights into the RL policy (strictly), critic warm-up, frozen BatchNorm

### Multi-Agent Setup
RLlib's MultiAgentEnv with three policies: `DO_NOTHING_POLICY`, `RL_POLICY` (trainable GNN/MLP), `HIGH_LEVEL_POLICY` (selector that decides whether the RL agent acts or does nothing). Policy mapping is handled in `src/grid2op_env/multi_agent_policies/`.

## Dependencies
Python 3.10–3.12. Key packages: `torch`, `torch-geometric`, `ray[rllib,tune]`, `grid2op`, `lightsim2grid`, `gymnasium`, `hydra-core`, `optuna`. See `environment.yaml` for the full conda spec.
