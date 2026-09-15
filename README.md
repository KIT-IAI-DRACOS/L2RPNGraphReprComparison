# L2RPN Graph Representation Comparison

Compares different graph observation space representations for RL agents in power grid congestion management (L2RPN). Agents are trained with **Ray RLlibs PPO** using a **GNN** or **MLP** policy on various graph encodings of the Grid2Op observation.

# Setup
I used `conda 24.9.1` and `python 3.12.10`

## Dependencies
### Step 0: (Optional) If you are on bwUniCluster:
```commandline
module load devel/miniforge
```
### Step 1: Clone Repo & Install dependencies
Clone the repository and `cd` into the cloned folder.

**If you use conda**
```commandline
conda env create -f environment.yaml
conda activate L2RPN
```

## Download and split scenarios
Download the required data. This may take a while.
```commandline
python setup_envs.py
```

## Train a model
```commandline
# GNN + PPO (default)
PYTHONPATH=$(pwd) python experiments/train.py

# MLP baseline
PYTHONPATH=$(pwd) python experiments/train.py model=mlp obs_space=flat

# Quick smoke test
PYTHONPATH=$(pwd) python experiments/train.py --config-name=minimal training=ppo_test
```

To adapt parameters explore the `configs`-folder.

# Project structure
- **configs**: Hydra configs containing hyperparameters
- **data**: necessary data to instantiate action spaces and observation converters
- **experiments**: entry points in the code
- **results**: folder to which results are written
- **src**: main source-code

The `src`-package contains the following packages:
- **agents**: Several agent wrappers for the G2OP ecosystem
- **algorithms**: Custom DQN, SAC, PPO and optuna extensions
- **analysis**: Code to analyze agents/models once they are trained
- **core**: Common code shared across several packages
- **grid2op_env**: RLlib environment implementation
- **rl**: GNN and MLP model implementations for RLlib
- **tests**: Unittests
- **visualization**: Utilities to create figures

# Citation
If you use this framework in your research, please consider citing our paper 📝 and giving the repository a star ⭐:
```commandline
@article{degenkolb2026comparative,
  title={A Comparative Study of Graph Representations for GNN-Based Power Grid Control in L2RPN},
  author={Degenkolb, Adrian and Huang, Qiong and Sch{\"a}fer, Benjamin},
  journal={arXiv preprint arXiv:2609.02538},
  year={2026}
}
```

