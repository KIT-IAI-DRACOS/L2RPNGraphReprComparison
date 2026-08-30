"""
Shared heuristic action implementations used by both the environment-side heuristics and agents.

Functions here operate on Grid2Op observations and action spaces to modify actions according to rules:
- reconnection_rule
- revert_to_reference_topo
- disconnection_rule

Each function is pure with respect to input arguments and returns a possibly updated action.
"""
import numpy as np
from grid2op.Action import BaseAction, ActionSpace
from grid2op.Observation import BaseObservation


def reconnection_rule(observation: BaseObservation, current_action: BaseAction, action_space: ActionSpace) -> BaseAction:
    """
    Reconnect disconnected lines when simulation indicates improvement.

    :param observation: The current observation.
    :param current_action: The action (so far).
    :param action_space: The action space.
    :return: The updated action including line reconnections.
    """
    line_stat_s = observation.line_status
    cooldown = observation.time_before_cooldown_line
    can_be_reco = ~line_stat_s & (cooldown == 0)
    if can_be_reco.any():
        sim_obs = observation.simulate(current_action)[0]
        cur_max_rho = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
        for id_ in can_be_reco.nonzero()[0]:
            action = current_action + action_space({"set_line_status": [(int(id_), +1)]})
            sim_obs = observation.simulate(action)[0]
            if cur_max_rho > (sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2):
                current_action = action
    return current_action


def revert_to_reference_topo(observation: BaseObservation, current_action: BaseAction, action_space: ActionSpace, reset_topo: float) -> BaseAction:
    """
    Revert each changed substation to reference topology when below a rho threshold and simulation indicates improvement.
    Applies each beneficial reset greedily in sequence.

    :param observation: The current observation.
    :param current_action: The action (so far).
    :param action_space: The action space.
    :param reset_topo: The threshold below which topology resets are considered (compared to max_rho).
    :return: The updated action including topology resets.
    """
    if (observation.rho.max() < reset_topo) and (observation.current_step < observation.max_step - 1):
        subs_changed = np.unique(observation._topo_vect_to_sub[observation.topo_vect != 1])
        if len(subs_changed) > 0:
            sim_obs = observation.simulate(current_action)[0]
            cur_max_rho = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
            for sub_id in subs_changed:
                action = current_action + action_space({
                    "set_bus": {
                        "substations_id": [(int(sub_id), np.ones(observation.sub_info[int(sub_id)], dtype=int))]
                    }
                })
                sim_obs = observation.simulate(action)[0]
                if cur_max_rho > (sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2):
                    current_action = action
    return current_action


def disconnection_rule(observation: BaseObservation, current_action: BaseAction, action_space: ActionSpace) -> BaseAction:
    """Manually disconnect a line during sustained overflow if simulation indicates improvement."""
    if np.any(observation.timestep_overflow > 1):
        sim_obs = observation.simulate(current_action)[0]
        cur_max_rho = sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2
        id_ = int(observation.timestep_overflow.argmax())
        action = current_action + action_space({"set_line_status": [(id_, -1)]})
        sim_obs = observation.simulate(action)[0]
        if cur_max_rho > (sim_obs.rho.max() if sim_obs.rho.max() > 0 else 2):
            current_action = action
    return current_action

