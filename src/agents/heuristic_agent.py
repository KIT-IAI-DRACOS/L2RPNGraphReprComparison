"""
A heuristic agent that executes heuristic rules based on the rule_config given.
"""

from grid2op.Action import ActionSpace, BaseAction
from grid2op.Agent import BaseAgent
from grid2op.Observation import BaseObservation

from core.heuristic_actions import reconnection_rule, revert_to_reference_topo, disconnection_rule


class HeuristicsAgent(BaseAgent):
    """
    This agent executes heuristic rules based on the rule_config given.

    rule_config can contain:
    -   rho_threshold: float value. Activation threshold of the lower level agent.
    -   line_reco: Boolean value. If True: attempt to reconnect all disconnected power lines if
        this is beneficial for the max rho value.
    -   line_disc: Boolean value. If True: manually disconnect a line during sustained periods of
        overflow in order to avoid permanent damage. Reconnect the line back soon after the
        cooldown period ends.
    -   reset_topo: float value. Revert Threshold. If the max load rho < reset_topo, the agent
        will execute actions to revert to the reference topology.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        rule_config: dict,
    ):
        BaseAgent.__init__(self, action_space)
        self.activation_thresh = rule_config.get("activation_threshold", 0.95)
        self.line_reco = rule_config.get("line_reco", False)
        self.line_disc = rule_config.get("line_disc", False)
        self.reset_topo = rule_config.get("reset_topo", 0)
        self.simulate = rule_config.get("simulate", False)
        self.rho_max = 0

    def activate_agent(self, _: BaseObservation):
        return self.rho_max > self.activation_thresh

    def act(self, observation: BaseObservation, reward: float, done : bool=False) -> BaseAction:
        current_action = self.action_space({})
        self.rho_max = (observation.rho.max() if observation.rho.max() > 0 else 2)
        if self.line_reco:
            current_action = reconnection_rule(observation, current_action, self.action_space)
        if self.reset_topo:
            current_action = revert_to_reference_topo(observation, current_action, self.action_space, self.reset_topo)
        if self.line_disc:
            current_action = disconnection_rule(observation, current_action, self.action_space)
        return current_action

    def simulate_combinations(self,
                              observation: BaseObservation,
                              topo_action: BaseAction,
                              rb_action: BaseAction) -> BaseAction:
        comb_action = rb_action + topo_action
        if self.simulate:
            # Test if the proposed topo_action improves the result.
            sim_obs, _, _, _ = observation.simulate(rb_action)
            cur_max_rho = (sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2)
            sim_obs, _, _, _ = observation.simulate(comb_action)
            if cur_max_rho > (sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2):
                # combined with the rule based action.
                action = comb_action
            else:
                # or excl the rule based action.
                sim_obs, _, _, _ = observation.simulate(topo_action)
                if cur_max_rho > (sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2):
                    action = topo_action
                else:
                    # Proposed topo_action is not better than rb_action only -> Take rule based action.
                    action = rb_action
        else:
            action = comb_action
        return action
