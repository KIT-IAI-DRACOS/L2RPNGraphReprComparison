"""
Imitation-learning data for the RL policy: a greedy Tutor that labels states in the policy's own
action space (do-nothing at index 0 + the reduced topology action set), recording the simulated
max rho of every action so hard labels, soft labels and regret can all be derived offline.

Runs in the main `L2RPN` environment — unlike `action_space_generation`, it does not depend on
`curriculumagent`.
"""
