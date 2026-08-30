"""
This package contains two simple policies that are used by the environment to implement a true semi-MDP.
The do_nothing policy simply returns the no-op action, while the high_level policy returns a single action that encodes whether the next action should be selected by the do_nothing policy or the RL policy.
"""
from .do_nothing_policy import DoNothingPolicy
from .select_agent_policy import SelectAgentPolicy

__all__ = ["DoNothingPolicy", "SelectAgentPolicy"]
