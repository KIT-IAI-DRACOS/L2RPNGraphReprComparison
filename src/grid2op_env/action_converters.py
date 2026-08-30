import json

import grid2op
import gymnasium
import numpy as np
from grid2op.Action import BaseAction
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv
from grid2op.gym_compat import GymEnv, DiscreteActSpace


class CustomIdToAct(IdToAct):
    """
    Defines also to_gym from actions.
    """

    def revert_act(self, action: BaseAction) -> int:
        """
        Do the opposite of convert_act. Given an action, return the id of this action in the list of all actions.
        """
        return int(np.where(self.all_actions == action)[0][0])


class CustomDiscreteActions(gymnasium.spaces.Discrete):
    """
    Class that customizes the action space.

    Example usage:

    import grid2op
    from grid2op.Converter import IdToAct

    env = grid2op.make("rte_case14_realistic")

    all_actions = # a list of of desired actions
    converter = IdToAct(env.action_space)
    converter.init_converter(all_actions=all_actions)

    env.action_space = ChooseDiscreteActions(converter=converter)
    """

    def __init__(self, converter: CustomIdToAct):
        self.converter = converter
        super().__init__(n=converter.n)

    # # NOTE: Implementation before fixing single agent
    # def from_gym(self, gym_action: dict[str, Any]) -> BaseAction:
    #     """
    #     Function that converts a gym action into a grid2op action.
    #     """
    #     return self.converter.convert_act(gym_action)

    def from_gym(self, gym_action: int) -> BaseAction:
        """
        Function that converts a gym action into a grid2op action.
        """
        return self.converter.convert_act(gym_action)

    def to_gym(self, action: BaseAction) -> int:
        return int(np.where(self.converter.all_actions == action)[0][0])

    def close(self) -> None:
        """Not implemented."""


def setup_converter(env: BaseEnv, possible_substation_actions: list[BaseAction]) -> CustomIdToAct:
    """
    Function that initializes and returns converter for gym to grid2op actions.
    """
    converter = CustomIdToAct(env.action_space)
    converter.init_converter(all_actions=possible_substation_actions)
    return converter


def load_actions(path: str, env: BaseEnv) -> list[BaseAction]:
    """
    Loads the .json with specified topology actions.
    """
    with open(path, "rt", encoding="utf-8") as action_set_file:
        return list(
            (
                env.action_space(action_dict)
                for action_dict in json.load(action_set_file)
            )
        )


if __name__ == "__main__":
    max_difficulty=10
    difficulty=1
    g2op_env = grid2op.make("l2rpn_case14_sandbox")
    gym_env = GymEnv(g2op_env)
    loaded_action_space = np.load("/home/adrian/Dev/RL2Grid/env/action_spaces/bus14_action_space.npy")
    n_actions = np.geomspace(50, len(loaded_action_space), num=max_difficulty).astype(int)
    gym_env.action_space = DiscreteActSpace(
        g2op_env.action_space,
        action_list=loaded_action_space[:n_actions[difficulty]]
    )

    for action in range(gym_env.action_space.n):
        print(gym_env.action_space.from_gym(action).to_json())
