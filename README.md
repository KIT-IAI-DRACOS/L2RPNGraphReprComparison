# L2RPN Graph Representation Comparison

Code and results of the paper *A Comparative Study of Graph Representations for GNN-Based Power Grid
Control in L2RPN*. Twelve observation representations (graphs built from the physical topology, from
electrical sensitivities (PTDF, LODF, Zbus) and hybrids of both, a graph without edges, and a flat MLP
baseline) are compared under one fixed pipeline:

- **14 bus** (`l2rpn_case14_sandbox`): PPO from scratch (Ray RLlib), Table I, Fig. S-1 to S-3.
- **36 bus** (`l2rpn_wcci_2020`): Teacher → Tutor → Junior (imitation learning) → Senior (PPO
  fine-tuning), Tables II and III.

Every representation observes the same node features (those of the Default graph), so only the graph
structure differs; all GNNs share one GCN with a per-substation readout.

# Setup

Tested with `conda 24.9.1` and Python 3.10. On bwUniCluster, run `module load devel/miniforge` first.

```bash
conda env create -f environment.yaml && conda activate L2RPN
python setup_envs.py   # download both grids and create the paper's train/val/test envs
```

`setup_envs.py` builds the val and test envs from the chronic lists in `data/splits/`, so evaluation
uses exactly the paper's episodes. Grid2Op's random split would draw different ones.

Only the N-1 Teacher (the first step of the 36-bus pipeline) needs the Fraunhofer
[`curriculumagent`](https://github.com/FraunhoferIEE/CurriculumAgent) package, in its own environment:

```bash
conda env create -f environment_curriculum.yaml   # env "curriculum"
```

The 36-bus Junior needs the Tutor dataset (667 MB), see [`data/README.md`](data/README.md).

# Reproducing the paper

All commands run from the repository root with `PYTHONPATH=$(pwd)/src`. The SLURM scripts in
`slurm/unicluster/` are the ones of the paper runs (bwUniCluster 3.0; adapt partitions and modules to your
cluster). Each script documents its options in its header; `DRY_RUN=1` prints the jobs instead of
submitting them.

The small result files the tables need are included in `results/`, so all tables and Fig. S-3 can be
regenerated without training:

| Paper | Notebook | Results |
|---|---|---|
| Fig. 1, 2 | `experiments/graph_obs_space_visualization.ipynb` | (no training) |
| Table I, Fig. S-3 | `experiments/compare_survival.ipynb` | `results/2026_10_28_compare_graph_obs_spaces_IEEE14/` |
| Tables II, III | `experiments/graph_comparison_36bus_results.ipynb` | `results/senior/2026_11_11_IEEE36_graph_comparison_senior_v2/` |

## 14 bus (Table I)

```bash
slurm/unicluster/compare_graph_obs_spaces_multiseed.sh   # 12 representations x 5 seeds, 100k steps each
```

Each run trains PPO (`experiments/train.py`) and then evaluates the latest checkpoint on the 50 test
episodes (`analysis/survival/`), including the action and congestion profiles of Fig. S-1 and S-2
(`analysis/actions/`, `analysis/congestion/`). A single run, e.g. the Substation graph:

```bash
PYTHONPATH=$(pwd)/src python experiments/train.py obs_space=graph_comparison/substation \
    reward.normalize_by_episode_length=true training.kl_coeff=0.0 experiment.nb_timesteps=100000 \
    experiment.post_training_evaluation.enabled=true experiment.name=my_run/substation
```
For local tests we recommend the minimal setup:
`PYTHONPATH=$(pwd)/src python experiments/train.py --config-name=minimal`.

## 36 bus (Tables II, III)

1. **Teacher** (env `curriculum`, several thousand CPU hours): the N-1 Teacher on the train chronics,
   aggregated to the 208 most frequent actions. The result,
   `data/action_spaces/l2rpn_wcci_2020/teacher_n1_k208.json`, is included; the commands of the paper
   run are in `data/README.md` and `slurm/unicluster/teacher_n1_array.sh`.
2. **Tutor** (env `L2RPN`): the greedy Tutor simulates every action of the action space in each critical
   state and labels the best one. The resulting dataset `data/imitation/case36_k208_greedy_hard.npz` is
   available for download (`data/README.md`); to rebuild it, run `slurm/unicluster/imitation_tutor_array.sh`
   and `experiments/imitation_data.py build-dataset`.
3. **Junior HPO**: one Optuna study per representation (40 trials, objective: val regret of top-20
   simulation). The best trial of every study is included (`results/junior/graph36_<obs>/best/config.yaml`,
   all trials in `trials.csv`), so this step can be skipped.
   ```bash
   slurm/unicluster/graph_comparison_36bus_il.sh
   ```
4. **Junior and Senior**: every best trial is retrained with seeds 0-4 (Junior), fine-tuned with PPO for
   200k agent steps (Senior; critic warm-up, the checkpoint with the highest val survival is kept), and
   both are rolled out greedily on the val and test envs (144 chronics each).
   ```bash
   slurm/unicluster/graph_comparison_36bus_senior.sh
   ```
   The rollout tables (`<obs>/junior_s<seed>/rollout/`, `<obs>/eval/senior_s<seed>/rollout/`) are what
   `experiments/graph_comparison_36bus_results.ipynb` reads.

# Project structure

- **configs/rllib**: Hydra configs; `obs_space/graph_comparison/` holds the 12 representations,
  `pretrain_hpo_graph36*.yaml` the Junior HPO.
- **data**: action spaces, env splits and imitation data ([`data/README.md`](data/README.md)).
- **experiments**: entry points and the result notebooks.
- **results**: result files of the paper runs needed by the notebooks.
- **slurm/unicluster**: job scripts of the paper runs.
- **src**:
  - **grid2op_env**: RLlib environment and the observation converters of all representations
  - **rl**: GNN (GCN, per-substation readout) and its RLlib model
  - **core**: training setup, config translation, callbacks, checkpoint loading
  - **action_space_generation**: N-1 Teacher (curriculumagent wrapper)
  - **imitation**: greedy Tutor, imitation datasets, Junior pretraining and HPO, rollouts, Senior helpers
  - **analysis**: post-training analysis (survival, actions, congestion), episode tables, training curves
  - **agents**, **algorithms**, **visualization**, **tests**

Tests: `cd src && python -m pytest tests`.

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
