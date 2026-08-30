"""
An agent that runs a RLlib model in the Grid2Op environment on top of a heuristic agent.
"""

import os
from typing import Any, Optional

from grid2op.Action import ActionSpace, BaseAction
from grid2op.Observation import BaseObservation
from ray.rllib import Policy

from core.constants import RL_POLICY
from grid2op_env import CustomizedGrid2OpEnvironment
from .heuristic_agent import HeuristicsAgent


class RllibAgent(HeuristicsAgent):
    """
    Class that runs a RLlib model in the Grid2Op environment.

    """

    def __init__(
        self,
        action_space: ActionSpace,
        env_config: dict[str, Any],
        file_path: str,
        policy_name: str,
        checkpoint_name: str,
        gym_wrapper: CustomizedGrid2OpEnvironment,
        conv_type: Optional[str] = None,
    ):
        rules = {
            "activation_threshold": env_config["rho_threshold"],
            "line_reco": env_config["line_reco"],
            "line_disc": env_config["line_disc"],
            "reset_topo": env_config["reset_topo"],
            "simulate": True,
        }
        HeuristicsAgent.__init__(self, action_space, rules)

        # load neural network of (eg) PPO agent.
        # Handle two cases:
        # 1. file_path is experiment dir, checkpoint_name is subdirectory name (e.g., "checkpoint_000000")
        # 2. file_path already points to checkpoint dir, checkpoint_name is empty
        if checkpoint_name:
            checkpoint_path = os.path.join(file_path, checkpoint_name, "policies", policy_name)
        else:
            # file_path already points to checkpoint directory
            checkpoint_path = os.path.join(file_path, "policies", policy_name)

        if conv_type is not None:
            from core.loading import _policy_from_checkpoint_with_conv_type
            self._rllib_agent = _policy_from_checkpoint_with_conv_type(checkpoint_path, conv_type)
        else:
            self._rllib_agent = Policy.from_checkpoint(checkpoint_path)
        self.obs_keys_order = [key for key in self._rllib_agent.observation_space.spaces.keys()]
        self.gym_wrapper = gym_wrapper

    def reset(self, observation: BaseObservation) -> None:
        """Reset per-episode state, including the observation converter (e.g. ablation permutations)."""
        self.gym_wrapper.reset_metrics()

    def act(
        self, observation: BaseObservation, reward: float, done: bool = False
    ) -> BaseAction:
        """
        Returns a grid2op action based on a RLlib observation.
        """

        # Grid2Op to RLlib observation
        self.gym_wrapper.update_obs(observation)
        # First do rule based part of the agent, line reconnections, disconnections and revert topo if needed.
        rb_action = HeuristicsAgent.act(self, observation, reward, done)

        if HeuristicsAgent.activate_agent(self, observation):
            # Get action from trained RL-agent when in danger.
            # compute_single_action returns:
            #   - Tuple consisting of the action,
            #   - the list of RNN state outputs (if any), and
            #   - a dictionary of extra features (if any).
            gym_action, state_out, info = self._rllib_agent.compute_single_action(
                self.gym_wrapper.cur_gym_obs,
                policy_id=RL_POLICY
            )
            # convert Rllib action to grid2op
            topo_action = self.gym_wrapper.env_gym.action_space.from_gym(gym_action)
            action = HeuristicsAgent.simulate_combinations(self, observation, topo_action, rb_action)
        else:
            action = rb_action

        return action
