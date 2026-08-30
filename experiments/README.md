# Experiments

All scripts must be launched from the **project root**. Two directories must be on the
Python path — `src/` (library code) and the project root itself (`experiments/` is
imported as a package by some analysis modules):

```bash
cd /path/to/L2RPNGraphReprComparison
PYTHONPATH=$(pwd)/src:$(pwd) python experiments/<script>.py
```

It is recommended to run the experiments via slurm on a HPC (see `slurm folder`)