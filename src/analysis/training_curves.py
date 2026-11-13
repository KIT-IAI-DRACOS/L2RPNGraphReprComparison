"""
Training curves of an RLlib/Tune run: selected TensorBoard scalars of a trial as one table with a row
per training iteration, e.g. for the 14-bus training curves (experiments/compare_survival.ipynb).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import SCALARS, EventAccumulator

LEARNER = "ray/tune/info/learner/reinforcement_learning_policy/learner_stats/"
# Label -> tag of the scalars the Senior notebook plots.
SENIOR_TAGS = {
    "vf_loss": LEARNER + "vf_loss",
    "vf_explained_var": LEARNER + "vf_explained_var",
    "kl": LEARNER + "kl",
    "entropy": LEARNER + "entropy",
    "policy_loss": LEARNER + "policy_loss",
    # Mean last grid2op step of the episodes: training (sampled actions) and val env (greedy, every few iterations).
    "train_survival": "ray/tune/custom_metrics/grid2op_end_mean",
    "eval_survival": "ray/tune/evaluation/custom_metrics/grid2op_end_mean",
}


# Label -> tag of the survival curves of the 14-bus training curves (experiments/compare_survival.ipynb).
SURVIVAL_TAGS = {key: SENIOR_TAGS[key] for key in ("train_survival", "eval_survival")}
SURVIVAL_CURVE_FILE = "analysis/training_curve.csv"


def load_survival_curve(trial_dir: Path) -> pd.DataFrame:
    """Training and evaluation survival of a Tune trial, one row per training iteration.

    Reads `<trial_dir>/analysis/training_curve.csv` if it exists; otherwise reads the TensorBoard
    event files (`load_scalars` with `SURVIVAL_TAGS`) and writes that CSV, so a results folder can be
    shipped without its event files.

    Returns:
        Columns `iteration`, `timesteps`, `train_survival` and `eval_survival` (mean last step of the
        episodes; NaN in iterations without evaluation).
    """
    path = trial_dir / SURVIVAL_CURVE_FILE
    if path.exists():
        return pd.read_csv(path)
    table = load_scalars(trial_dir, SURVIVAL_TAGS)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def load_scalars(trial_dir: Path, tags: dict[str, str]) -> pd.DataFrame:
    """Scalars of every event file in a Tune trial directory, one row per logged step.

    Args:
        trial_dir: Tune trial directory (contains `events.out.tfevents.*`).
        tags: Column label -> TensorBoard tag; tags the run did not log are left out.

    Returns:
        Columns `iteration` (1-based training iteration), `timesteps` (Tune's step: agent steps
        sampled so far) and one per found tag; NaN where a tag was not logged in an iteration
        (e.g. evaluation, which runs every few iterations). Later event files win on equal steps.
    """
    columns: dict[str, pd.Series] = {}
    for path in sorted(trial_dir.glob("events.out.tfevents.*")):
        events = EventAccumulator(str(path), size_guidance={SCALARS: 0})
        events.Reload()
        found = set(events.Tags()[SCALARS])
        for label, tag in tags.items():
            if tag not in found:
                continue
            series = pd.Series({e.step: e.value for e in events.Scalars(tag)}, dtype=float)
            columns[label] = series if label not in columns else series.combine_first(columns[label])
    table = pd.DataFrame(columns).sort_index()
    table.index.name = "timesteps"
    table = table.reset_index()
    table.insert(0, "iteration", range(1, len(table) + 1))
    return table
