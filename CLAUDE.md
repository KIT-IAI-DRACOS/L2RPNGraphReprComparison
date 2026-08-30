# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project compares different graph observation space representations for RL agents managing congestion in simulated power grids (L2RPN). Agents are trained with Ray RLlib using GNN or MLP policies on various graph encodings of the Grid2Op observation.

## Commands

### Environment Setup
```bash
conda env create -f environment.yaml && conda activate L2RPN
python setup_envs.py  # Download and split Grid2Op scenario data
```

### Training
```bash
# Default GNN + PPO (from project root, PYTHONPATH must include .)
PYTHONPATH=$(pwd) python experiments/train.py

# Key config group overrides
python experiments/train.py training=ppo model=gnn obs_space=graph
python experiments/train.py training=sac model=mlp obs_space=flat
python experiments/train.py experiment=test_minimal  # Quick smoke test

# CLI overrides
python experiments/train.py experiment.nb_timesteps=500000 training.lr=3e-4
```

### Tests
```bash
python -m pytest src/tests/
```

## Architecture

The training pipeline is configured entirely via **Hydra** (configs/rllib/) and orchestrated through `experiments/train.py`.

### Config Groups (configs/rllib/)
- **experiment/**: trial-level settings (timesteps, seed, checkpoint freq)
- **training/**: algorithm params — `ppo.yaml`, `sac.yaml`, `dqn.yaml`, and `*_test.yaml` variants
- **model/**: network architecture — `gnn.yaml` (GNN policy), `mlp.yaml` (baseline)
- **obs_space/**: graph observation space variant (e.g. `graph.yaml`, `lodf.yaml`, `ptdf.yaml`, …) or `flat.yaml`
- **env/**: Grid2Op case definition (case14, case36, case118)
- **reward/**: reward function class (`rho`, `scaled_l2rpn`, `constant`, `alpha_zero`)

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
- **src/rl/**: Pure PyTorch — `gnn.py` (GNN backbone), `gnn_ppo_model.py` (RLlib TorchModelV2 wrapper), `mlp.py`
- **src/grid2op_env/**: Grid2Op wrappers — multi-agent env, observation/action converters, multi-agent policy implementations
- **src/core/**: `constants.py` (global paths/names), `train.py` (training setup helpers), `loading.py` (checkpoint loading), `evaluate.py`
- **src/algorithms/**: `CustomPPO`, `CustomSAC`, Optuna integration
- **src/analysis/**: Post-training analysis — MetricAnalyzer, survival/action/congestion analyzers

### Constants (src/core/constants.py)
Defines global output paths (`LOGS_PATH`, `MODELS_PATH`, `EVAL_PATH`), agent name strings (`RL_AGENT`, `HIGH_LEVEL_AGENT`, `DO_NOTHING_AGENT`), and policy names used consistently across the codebase.

### Callbacks
- `CustomMetricsCallback`: collects episode metrics (survival, interact counts, curriculum)
- `TimerCallback`: logs per-step wall-clock timings to TensorBoard
- `TuneCallback`: logs custom metrics used by Ray Tune stoppers/schedulers

### Multi-Agent Setup
RLlib's MultiAgentEnv with three policies: `DO_NOTHING_POLICY`, `RL_POLICY` (trainable GNN/MLP), `HIGH_LEVEL_POLICY` (selector that decides whether the RL agent acts or does nothing). Policy mapping is handled in `src/grid2op_env/multi_agent_policies/`.

## Dependencies
Python 3.10–3.12. Key packages: `torch`, `torch-geometric`, `ray[rllib,tune]`, `grid2op`, `gymnasium`, `hydra-core`, `optuna`, `stable-baselines3`. See `environment.yaml` for the full conda spec.
