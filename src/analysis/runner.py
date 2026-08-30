from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from grid2op.Environment import Environment
from tqdm import tqdm

from agents import RllibAgent
from analysis.interfaces import EpisodeAnalyzer, StepContext

logger = logging.getLogger(__name__)


class PostTrainingRunner:
    def __init__(
        self,
        agent: RllibAgent,
        env: Environment,
        analyzers: List[EpisodeAnalyzer],
    ) -> None:
        self.agent = agent
        self.env = env
        self.analyzers = analyzers
        logger.info(
            "PostTrainingRunner: analyzers=%s",
            [type(a).__name__ for a in analyzers],
        )

    def run(self, num_episodes: Optional[int] = None) -> None:
        n_available = len(self.env.chronics_handler.available_chronics())
        n_episodes = n_available if num_episodes is None else min(num_episodes, n_available)
        logger.info("Running %d episodes.", n_episodes)

        for ep_idx in range(n_episodes):
            obs = self.env.reset()
            episode_id = self.env.chronics_handler.get_name()
            max_steps = self.env.max_episode_duration()

            for a in self.analyzers:
                a.on_episode_start(episode_id, max_steps)

            done = False
            step_idx = 0
            reward = 0.0

            pbar = tqdm(
                total=max_steps,
                desc=f"[{ep_idx+1}/{n_episodes}] {episode_id}",
                unit="step",
                leave=False,
            )
            while not done:
                is_rl_step = self.agent.activate_agent(obs)
                action = self.agent.act(obs, reward, done)

                obs_next, reward, done, info = self.env.step(action)

                ctx = StepContext(
                    obs=obs,
                    action=action,
                    obs_next=obs_next,
                    reward=reward,
                    done=done,
                    info=info,
                    is_rl_step=is_rl_step,
                    episode_id=episode_id,
                    step_idx=step_idx,
                    env=self.env,
                )
                for a in self.analyzers:
                    a.on_step(ctx)

                obs = obs_next
                step_idx += 1
                pbar.update(1)

            pbar.close()
            steps_survived = self.env.nb_time_step
            for a in self.analyzers:
                a.on_episode_end(steps_survived, max_steps)

            status = "OK" if steps_survived >= max_steps else f"FAIL@{steps_survived}"
            logger.info("Episode %s: %s/%s (%s)", episode_id, steps_survived, max_steps, status)

    def save_all(self, out_dir: Path) -> None:
        for a in self.analyzers:
            a.save(out_dir)
