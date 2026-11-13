# Data

Inputs of the experiments that are not produced by training.

```
data/
├── splits/                  # val/test chronics of the paper (setup_envs.py builds the Grid2Op envs from them)
├── action_spaces/
│   ├── l2rpn_case14_sandbox/medha.json        # 14 bus: topology actions of Subramanian et al.
│   └── l2rpn_wcci_2020/teacher_n1_k208.json   # 36 bus: N-1 Teacher, 208 most frequent actions
└── imitation/
    ├── case36_split.json                      # chronic-level train/val/test split of the imitation data
    └── case36_k208_greedy_hard.npz            # Tutor dataset (download, see below)
```

## `splits/`

`<env>.json` lists the chronics of the val and test envs used in the paper (14 bus: 50 / 50, 36 bus:
144 / 144). `setup_envs.py` creates `<env>_train`, `<env>_val` and `<env>_test` from them; every
chronic that is in neither list goes to the train env.

## `action_spaces/`

JSON lists of Grid2Op action dicts, selected with `env.action_space` (`configs/rllib/env/`). The RL
agent's discrete actions are do-nothing (index 0) followed by the file's actions in file order.

`teacher_n1_k208.json` was built with the N-1 Teacher of the Fraunhofer
[`curriculumagent`](https://github.com/FraunhoferIEE/CurriculumAgent) package on the 2592 chronics of
`l2rpn_wcci_2020_train` (`slurm/unicluster/teacher_n1_array.sh`, lines to attack
`20 13 39 33 35 32 11 34 23 22`), then aggregated:

```bash
PYTHONPATH=$(pwd)/src python experiments/build_action_space.py aggregate --grid case36 \
    --experience "results/teacher/2026_09_17_teacher_n1_case36/shard_*.csv" \
    --top-k 208 --out data/action_spaces/l2rpn_wcci_2020/teacher_n1_k208.json
```

## `imitation/`

`case36_k208_greedy_hard.npz` (667 MB) is the imitation dataset of the Junior: the greedy Tutor
(`src/imitation/tutor.py`) on `teacher_n1_k208` over the 2592 train chronics, 94,742 states with hard
labels (format: `src/imitation/dataset.py`). It is too large for git; download it into this folder:

- Download: **TODO: add the Zenodo link**
- SHA-256: `9f21c9bc15076d7a879de2563340b0bf40e4d90ed05d2bc914b53f05fb474498`

To rebuild it instead (see README.md):

```bash
slurm/unicluster/imitation_tutor_array.sh case36 <tutor_experiment> \
    data/action_spaces/l2rpn_wcci_2020/teacher_n1_k208.json 100 100
PYTHONPATH=$(pwd)/src python experiments/imitation_data.py build-dataset \
    --experience-dir results/tutor/<tutor_experiment>/experience --target hard \
    --out data/imitation/case36_k208_greedy_hard.npz
```

`case36_split.json` assigns the train chronics to the train/val/test splits of imitation learning
(stratified by month); every Junior uses it.
