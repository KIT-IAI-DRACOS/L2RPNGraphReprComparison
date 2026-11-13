# Experiments

Entry points and result notebooks. Run everything from the **project root** with `src/` on the Python path:

```bash
PYTHONPATH=$(pwd)/src python experiments/<script>.py
```

| File | Purpose |
|---|---|
| `train.py` | RL training (PPO); 14-bus runs and 36-bus Seniors |
| `build_action_space.py` | N-1 Teacher stages (env `curriculum`): line selection, sharded runs, aggregation |
| `imitation_data.py` | Greedy Tutor runs and the imitation dataset |
| `pretrain_policy.py` | Junior: imitation pretraining of the policy network |
| `pretrain_hpo.py`, `summarize_hpo.py` | Junior HPO (Optuna) and its summary / best trial |
| `architecture_overrides.py` | Hydra overrides that give a Senior its Junior's network |
| `export_policy_weights.py` | Exports a Senior checkpoint in the Junior's format (best-val selection) |
| `imitation_rollout.py` | Greedy rollouts of a Junior or exported Senior on the Grid2Op envs |
| `graph_obs_space_visualization.ipynb` | Fig. 1 and 2 |
| `compare_survival.ipynb` | Table I and Fig. S-3 (14 bus) |
| `graph_comparison_36bus_results.ipynb` | Tables II and III (36 bus) |

The paper's runs were submitted with the SLURM scripts in `slurm/unicluster/` (see README.md).
