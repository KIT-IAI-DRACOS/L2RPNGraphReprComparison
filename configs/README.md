# Configuration

[Hydra](https://hydra.cc/) configs of `experiments/train.py` (RL), `experiments/pretrain_policy.py`
(Junior) and `experiments/pretrain_hpo.py` (Junior HPO). `rllib/config.yaml` selects one file per group;
any group or value can be overridden on the command line:

```bash
PYTHONPATH=$(pwd)/src python experiments/train.py env=case36 obs_space=graph_comparison/lodf training.lr=3e-4
```

| Group | Files | Purpose |
|---|---|---|
| `experiment/` | `default`, `minimal` | Timesteps, seed, checkpoint frequency, results folder (`experiment.name`), post-training evaluation |
| `training/` | `ppo`, `ppo_test` (+ `base`) | PPO hyperparameters. The paper runs set `training.kl_coeff=0.0` |
| `model/` | `gcn_substation_readout`, `mlp` | GCN with per-substation readout (all graph representations); RLlib's default MLP (flat baseline) |
| `obs_space/graph_comparison/` | 12 representations | `complete` (Default graph), `heterogeneous`, `substation`, `elements`, `ptdf`, `lodf`, `zbus`, `substation_ptdf`, `substation_zbus`, `elements_lodf`, `no_edges`, `flat`; all with `node_features: complete` |
| `env/` | `case14`, `case36` | Grid2Op env (`_train`/`_val`/`_test` appended), action space, rho threshold, chronic order |
| `reward/` | `constant`, ... | Reward class; the paper runs use `constant` with `reward.normalize_by_episode_length=true` |
| `imitation/` | `disabled`, `default` | Junior pretraining settings and the Senior's start from its weights (`imitation.weights_path`) |
| `evaluation/` | `default`, `minimal` | Evaluation on the val env during training (every evaluation plays all val chronics) |
| `rollouts/` | `default`, `minimal` | Rollout workers and GPUs (`rollouts.num_gpus` is the only GPU switch) |
| `callbacks/` | `default` | RLlib callbacks: custom metrics, timing, imitation initialisation |
| `optimization/` | `disabled`, `optuna` | Optional Ray Tune + Optuna search over RL hyperparameters (not used in the paper) |
| `opponent/` | `disabled`, `random_line` | Optional line-attack opponent (not used in the paper) |

Primary configs besides `config.yaml`:

- `minimal.yaml`: smoke test (`--config-name=minimal`).
- `pretrain_hpo_graph36.yaml`, `pretrain_hpo_graph36_mlp.yaml`: Junior HPO on IEEE-36 (search space and
  pruner under `hpo`), for the graph representations and the MLP baseline.

The Senior runs (`slurm/unicluster/graph_comparison_36bus_senior.sh`) use `config.yaml` with the
Junior's architecture added as overrides (`experiments/architecture_overrides.py`).
