"""
Chooses whatever action that minimizes the maximum load on the grid at the next time step.
"""

import random
from typing import Any, Optional

import numpy as np
from grid2op.Action import ActionSpace, BaseAction
from grid2op.Observation import BaseObservation
from grid2op.Reward import BaseReward
from grid2op.dtypes import dt_float

from .heuristic_agent import HeuristicsAgent


class RhoGreedyAgent(HeuristicsAgent):
    """
    Defines the behavior of a Greedy Agent that can perform topology changes based
    on a provided set of possible actions.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        env_config: dict[str, Any],
        possible_actions: list[BaseAction],
    ):
        HeuristicsAgent.__init__(self, action_space, env_config["rules"])
        self.tested_action: list[BaseAction] = []
        self.action_space = action_space
        self.possible_actions = possible_actions
        self.simulate = True

        random.seed(env_config["seed"])

        self.timesteps_saved = 0

    def act(
        self,
        observation: BaseObservation,
        reward: Optional[BaseReward],
        done: Optional[bool] = False,
    ) -> BaseAction:
        """
        By definition, all "greedy" agents are acting the same way. The only thing that can differentiate multiple
        agents is the actions that are tested.

        These actions are defined in the method :func:`._get_tested_action`. This :func:`.act` method implements the
        greedy logic: take the actions that maximizes the instantaneous reward on the simulated action.

        Parameters
        ----------
        observation: :class:`grid2op.Observation.Observation`
            The current observation of the :class:`grid2op.Environment.Environment`

        reward: ``float``
            The current reward. This is the reward obtained by the previous action

        done: ``bool``
            Whether the episode has ended or not. Used to maintain gym compatibility

        Returns
        -------
        res: :class:`grid2op.Action.Action`
            The action chosen by the bot / controller / agent.

        """
        # First do rule based part of the agent, line reconnections, disconnections and revert topo if needed.
        rb_action = HeuristicsAgent.act(self, observation, reward, done)
        # if the threshold is exceeded, act
        if HeuristicsAgent.activate_agent(self, observation):
            # get all possible actions to be tested
            self.tested_action = self._get_tested_action(observation)
            # simulate all possible actions and choose the best
            if len(self.tested_action) > 1:
                resulting_rho_observations = np.full(
                    shape=len(self.tested_action), fill_value=np.NaN, dtype=dt_float
                )

                for i, action in enumerate(self.tested_action):
                    (
                        simul_observation,
                        _,
                        _,
                        simul_info,
                    ) = observation.simulate(action + rb_action)
                    resulting_rho_observations[i] = np.max(
                        simul_observation.to_dict()["rho"]
                    )
                    # Include extra safeguard to prevent exception actions with converging power flow
                    if simul_info["exception"]:
                        resulting_rho_observations[i] = 999999
                # Get action with the lowest simulated rho_max(t+1)
                rho_idx = int(np.argmin(resulting_rho_observations))
                topo_action = self.tested_action[rho_idx]

                # Combine Greedy topology action with rb_action
                action = HeuristicsAgent.simulate_combinations(self, observation, topo_action, rb_action)
            else:
                action = rb_action
        # if the threshold is not exceeded, do nothing / no topology action.
        else:
            action = rb_action
        return action

    def _get_tested_action(self, _: BaseObservation) -> list[BaseAction]:
        """
        Adds all possible actions to be tested.
        """
        if not self.tested_action:
            # add the do nothing
            res = [self.action_space({})]
            # add all possible actions
            res += self.possible_actions
            self.tested_action = res
        return self.tested_action
