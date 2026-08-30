"""
Class that defines the custom grid2op to gym environment with the set observation and action spaces.
"""
import os
import time
from typing import Any, Dict, Optional, Tuple

import grid2op
import gymnasium as gym
import numpy as np
from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation
from grid2op.gym_compat import GymEnv
from ray.rllib import RolloutWorker
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env

from core.constants import DO_NOTHING_AGENT, RL_AGENT, HIGH_LEVEL_AGENT, DO_NOTHING_POLICY, RL_POLICY, \
    HIGH_LEVEL_POLICY
from core.heuristic_actions import reconnection_rule, revert_to_reference_topo, disconnection_rule
from .action_converters import CustomDiscreteActions, setup_converter, load_actions
from .observation_converter import make_observation_converter, ObservationConverter
from .utils import make_g2op_env

# Environment configuration per curriculum level:
ENV_CUR_MAP = [
    {
        # Easiest level
        "NO_OVERFLOW_DISCONNECTION": True,
        "SOFT_OVERFLOW_THRESHOLD": 99,
        "HARD_OVERFLOW_THRESHOLD": 999,
        "NB_TIMESTEP_OVERFLOW_ALLOWED": 99,
    },
    {
        # Mid-level
        "NO_OVERFLOW_DISCONNECTION": False,
        "SOFT_OVERFLOW_THRESHOLD": 2,
        "HARD_OVERFLOW_THRESHOLD": 99,
        "NB_TIMESTEP_OVERFLOW_ALLOWED": 15,
    },
    # None means no adjustments > using default values
    None
]


class CustomizedGrid2OpEnvironment(MultiAgentEnv):
    """Encapsulate Grid2Op environment and set action/observation space."""

    def __init__(self, env_config: dict[str, Any]):
        super().__init__()
        self._skip_env_checking = True

        lib_dir = env_config["lib_dir"]
        self._agent_ids = [HIGH_LEVEL_AGENT, RL_AGENT, DO_NOTHING_AGENT]

        # 1. create the grid2op environment
        self.env_g2op = make_g2op_env(env_config)
        if "env_name" not in env_config:
            raise RuntimeError("The configuration for RLLIB should provide the env name")

        # 2. create the gym environment
        self.env_gym = GymEnv(self.env_g2op, with_forecast=True)
        self.env_gym.reset()

        # 3. Setting up custom action space
        path = os.path.join(lib_dir, f"data/action_spaces/{self.env_g2op.env_name}/{env_config['action_space']}.json")  # path for action space
        self.possible_substation_actions = load_actions(path, self.env_g2op)                                            # load JSON file to get the list of possible actions for the RL agent
        do_nothing_action = self.env_g2op.action_space({})                                                              # add the do-nothing action at index 0
        self.possible_substation_actions.insert(0, do_nothing_action)
        converter = setup_converter(self.env_g2op, self.possible_substation_actions)                                    # enumerate the possible actions and create a converter to convert from gym action index to g2op action
        self.env_gym.action_space = CustomDiscreteActions(converter)                                                    # set the gym action space to be discrete with the number of possible actions for the RL agent
        self.action_space = self.define_action_space()                                                                  # set up the multi-agent action space with the high-level agent and the do-nothing agent

        # 4. customize observation space
        self._obs_space_in_preferred_format = True
        self.observation_converter: ObservationConverter = make_observation_converter(self.env_gym, env_config)
        self.observation_space = gym.spaces.Dict({
            HIGH_LEVEL_AGENT: gym.spaces.Discrete(2),
            RL_AGENT: self.observation_converter.observation_space,
            DO_NOTHING_AGENT: gym.spaces.Discrete(1),
        })
        self.cur_gym_obs = None
        self.cur_g2op_obs = None

        # 5. environment settings
        self.line_reco         = env_config.get("line_reco", True)                                           # reconnect lines when disconnected
        self.line_disc         = env_config.get("line_disc", False)                                          # disconnect lines when overloaded
        self.reset_topo        = env_config.get("reset_topo", 0)                                             # reset topo option
        self.penalty_game_over = env_config.get("penalty_game_over", 0)                                      # penalty for game over
        self.reward_finish     = env_config.get("reward_finish", 0)                                          # reward for finishing complete episode

        # 6. initialize metrics
        self.step_surv        = 0
        self.interact_count   = 0
        self.activated        = False
        self.active_dn_count  = 0
        self.reconnect_count  = 0
        self.disconnect_count = 0
        self.reset_count      = 0
        self._timing_sum: dict[str, float] = {}
        self._timing_count: int = 0

        # initialize curriculum level:
        if env_config.get("curriculum_training", False):
            self.set_curriculum(level=0)

    def reset_metrics(self):
        # different metrics to keep track of episode performance
        self.interact_count = 0
        self.activated = False
        self.active_dn_count = 0
        self.reconnect_count = 0
        self.disconnect_count = 0
        self.reset_count = 0
        self._timing_sum = {}
        self._timing_count = 0
        self.observation_converter.reset_obs()

    def define_action_space(self) -> gym.Space:
        # Defines Single Agent action space
        self._action_space_in_preferred_format = True
        return gym.spaces.Dict(
            {
                HIGH_LEVEL_AGENT: gym.spaces.Discrete(2),
                RL_AGENT: gym.spaces.Discrete(len(self.possible_substation_actions)),
                DO_NOTHING_AGENT: gym.spaces.Discrete(1),
            }
        )

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        # reset episode metrics
        self.reset_metrics()
        g2op_obs = self.env_g2op.reset()
        self.update_obs(g2op_obs)
        # Start with activation of the high level agent > decide to act or not to act.
        observations = {HIGH_LEVEL_AGENT: g2op_obs.rho.max().flatten()}
        chron_id = self.env_g2op.chronics_handler.get_name()
        infos = {"time series id": chron_id}

        return observations, infos

    def update_obs(self, g2op_obs):
        self.cur_g2op_obs = g2op_obs
        self.cur_gym_obs = self.observation_converter.to_gym(g2op_obs)

    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """
        This function performs a single step in the environment.
        """

        # Build termination dict
        terminateds = {
            "__all__": False,
        }
        truncateds = {
            "__all__": False,
        }

        rewards: Dict[str, Any] = {}
        infos: Dict[str, Any] = {}
        observations = {}

        if HIGH_LEVEL_AGENT in action_dict.keys():                                                                      # high level agent decides which agent acts next
            action = action_dict[HIGH_LEVEL_AGENT]                                                                      # get action
            if action == 0: observations = {RL_AGENT: self.cur_gym_obs}                                                 # if action == 0 we query the RL agent with the current observation
            elif action == 1: observations = {DO_NOTHING_AGENT: 0}                                                      # if action == 1 we query the do-nothing agent
            else: raise ValueError(f"An invalid action ({action}) is selected by the high_level_agent.")
            return observations, rewards, terminateds, truncateds, infos                                                # observation dict contains a key for either the RL agent or the do-nothing agent

        elif DO_NOTHING_AGENT in action_dict.keys():                                                                    # if the do-nothing agent selected the last action, we
            action = action_dict[DO_NOTHING_AGENT]                                                                      # get the action
            self.activated = False                                                                                      # mark the current step as idle
        elif RL_AGENT in action_dict.keys():                                                                            # if the RL agent selected the last action, we
            action = action_dict[RL_AGENT]                                                                              # get the action
            self.activated = True                                                                                       # mark the current step as non-idle
            self.interact_count += 1                                                                                    # increase the interaction count metric for the episode
        elif not bool(action_dict):
            return observations, rewards, terminateds, truncateds, infos # TODO do we need this
        else:
            raise ValueError("No agent found in action dictionary in step().")

        g2op_obs, reward, terminated, infos = self.gym_act_in_g2op(action)                                              # Execute action given by DN or RL agent (enriched by heuristic actions):
        rewards = {RL_AGENT: reward}                                                                                    # Give reward to RL agent
        observations = {HIGH_LEVEL_AGENT: g2op_obs.rho.max().flatten()}                                                 # Let high-level agent decide to act or not
        terminateds = {"__all__": terminated}
        truncateds = {"__all__": g2op_obs.current_step == g2op_obs.max_step}
        infos = {}
        return observations, rewards, terminateds, truncateds, infos

    def gym_act_in_g2op(self, action: int) -> Tuple[BaseObservation, float, bool, dict]:
        g2op_act = self.env_gym.action_space.from_gym(action)                                                           # get grid2op action from gym action index using the converter
        if self.activated:
            act_config = g2op_act.set_bus
            if np.all(self.env_g2op.current_obs.topo_vect[act_config!=0] == act_config[act_config!=0]):                 # If the rl-agent picks the do nothing action
                self.active_dn_count += 1                                                                               # we register this in active_dn_count

        t0 = time.perf_counter()
        if self.line_reco: g2op_act = self.reconnect_lines(g2op_act)                                                    # reconnect lines if needed.
        if self.line_disc: g2op_act = self.disconnect_lines(g2op_act)                                                   # disconnect lines if needed.
        if self.reset_topo: g2op_act = self.reset_ref_topo(g2op_act)                                                    # reset topo if needed.
        heuristic_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        g2op_obs, reward, terminated, infos, = self.env_g2op.step(g2op_act)                                             # execute action
        g2op_step_ms = (time.perf_counter() - t0) * 1000

        self.update_obs(g2op_obs)                                                                                       # memorize current observation

        # Accumulate timings for this step
        step_timings = {"g2op_step_ms": g2op_step_ms, "heuristic_ms": heuristic_ms}
        if hasattr(self.observation_converter, "_timings"):
            step_timings.update(self.observation_converter._timings)
        for key, val in step_timings.items():
            self._timing_sum[key] = self._timing_sum.get(key, 0.0) + val
        self._timing_count += 1

        if self.penalty_game_over and terminated: reward = self.penalty_game_over                                       # Adjust reward when episode terminates early
        if self.reward_finish and g2op_obs.current_step == g2op_obs.max_step: reward = self.reward_finish               # Adjust reward when episode finished

        return g2op_obs, reward, terminated, infos

    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        if not isinstance(x, dict):
            return False
        return all(self.observation_space.contains(val) for val in x.values())

    def reconnect_lines(self, g2op_action: BaseAction) -> BaseAction:
        """Enriches given action by line reconnections where simulation indicates improvement."""
        result = reconnection_rule(self.cur_g2op_obs, g2op_action, self.env_g2op.action_space)
        if result is not g2op_action:
            self.reconnect_count += 1
        return result

    def disconnect_lines(self, g2op_action: BaseAction) -> BaseAction:
        """Manually disconnects an overloaded line when simulation indicates improvement."""
        result = disconnection_rule(self.cur_g2op_obs, g2op_action, self.env_g2op.action_space)
        if result is not g2op_action:
            self.disconnect_count += 1
        return result

    def reset_ref_topo(self, g2op_action: BaseAction) -> BaseAction:
        """Reverts substations to reference topology in safe states when simulation indicates improvement."""
        result = revert_to_reference_topo(self.cur_g2op_obs, g2op_action, self.env_g2op.action_space, self.reset_topo)
        if result is not g2op_action:
            self.reset_count += 1
        return result

    def set_curriculum(self, level: int):
        print("Change curriculum to level: ", level)
        new_params = ENV_CUR_MAP[level]
        if new_params is not None:
            p = self.env_g2op.parameters
            p.init_from_dict(new_params)
            self.env_g2op.change_parameters(p)
            _ = self.env_g2op.reset()
            print("Parameters used: \n", self.env_g2op.parameters.to_dict())
        else:
            # use default parameters
            print("Using default parameters.")
            env_default = grid2op.make(self.env_g2op.env_name)
            default_par = env_default.parameters
            self.env_g2op.change_parameters(default_par)


def policy_mapping_fn(
    agent_id: str,
    episode: Optional[EpisodeV2] = None,
    worker: Optional[RolloutWorker] = None,
) -> str:
    """Maps each agent to a policy."""
    if agent_id.startswith(RL_AGENT):
        return RL_POLICY
    if agent_id.startswith(HIGH_LEVEL_AGENT):
        return HIGH_LEVEL_POLICY
    if agent_id.startswith(DO_NOTHING_AGENT):
        return DO_NOTHING_POLICY
    raise NotImplementedError


register_env("CustomizedGrid2OpEnvironment", CustomizedGrid2OpEnvironment)
