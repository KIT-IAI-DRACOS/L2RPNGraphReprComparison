from __future__ import annotations
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.interfaces import EpisodeAnalyzer, StepContext


class SurvivalAnalyzer(EpisodeAnalyzer):
    def __init__(self) -> None:
        self._records: List[dict] = []

    def on_step(self, ctx: StepContext) -> None:
        pass

    def on_episode_end(self, steps_survived: int, max_steps: int) -> None:
        self._records.append({
            "steps_survived": steps_survived,
            "max_steps": max_steps,
            "completed": steps_survived >= max_steps,
            "survival_pct": steps_survived / max_steps if max_steps > 0 else 0.0,
        })

    def save(self, out_dir: Path) -> None:
        d = out_dir / "survival"
        d.mkdir(parents=True, exist_ok=True)

        if not self._records:
            return

        survived = np.array([r["steps_survived"] for r in self._records])
        max_s = np.array([r["max_steps"] for r in self._records])
        pcts = survived / np.maximum(max_s, 1)
        completed = np.array([r["completed"] for r in self._records])

        summary = {
            "n_episodes": len(self._records),
            "completed_episodes_pct": float(completed.mean() * 100),
            "survived_steps_pct": float(pcts.mean() * 100),
            "survived_steps_mean": float(survived.mean()),
            "survived_steps_std": float(survived.std()),
        }
        with open(d / "summary.json", "w") as f:
            json.dump(summary, f, indent=4)
        np.save(d / "survived_steps.npy", survived)
        np.save(d / "max_steps.npy", max_s)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(pcts * 100, bins=20, range=(0, 100), edgecolor="white")
        ax.axvline(float(pcts.mean() * 100), color="tab:red", linestyle="--",
                   label=f"Mean {pcts.mean()*100:.1f}%")
        ax.set_xlabel("Survival (%)")
        ax.set_ylabel("Episodes")
        ax.set_title("Survival duration distribution")
        ax.legend()
        plt.tight_layout()
        fig.savefig(d / "survival_histogram.png", dpi=150, bbox_inches="tight")
        fig.savefig(d / "survival_histogram.svg", bbox_inches="tight")
        plt.close(fig)
