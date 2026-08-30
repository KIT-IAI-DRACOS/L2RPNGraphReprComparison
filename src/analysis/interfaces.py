from __future__ import annotations
import abc
from dataclasses import dataclass
from pathlib import Path

from grid2op.Action import BaseAction
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation


@dataclass
class StepContext:
    obs: BaseObservation
    action: BaseAction
    obs_next: BaseObservation
    reward: float
    done: bool
    info: dict
    is_rl_step: bool
    episode_id: str
    step_idx: int
    env: Environment


class EpisodeAnalyzer(abc.ABC):
    """Single-pass post-training analysis module."""

    def on_episode_start(self, episode_id: str, max_steps: int) -> None:
        pass

    @abc.abstractmethod
    def on_step(self, ctx: StepContext) -> None:
        pass

    def on_episode_end(self, steps_survived: int, max_steps: int) -> None:
        pass

    @abc.abstractmethod
    def save(self, out_dir: Path) -> None:
        """Persist arrays and plots under out_dir/<subdir>/."""
