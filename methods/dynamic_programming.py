import numpy as np
from mdp.markov_decision_process import MDP

# TODO: remove policy attribute from MDP class


def get_optimal_state_value(mdp: MDP) -> np.ndarray:
    # Initialize state-values matrix
    v = np.zeros(len(mdp.env.states), dtype=float)

    # Solution convergence threshold
    epsilon = 1e-10

    delta = float('inf')
    while delta > epsilon:
        delta = 0
        for state_id, state in enumerate(mdp.env.states):
            v_old = v[state_id]
            v_new = __compute_state_value(mdp, state, v)
            v[state_id] = v_new
            delta = max(delta, abs(v_new - v_old))

    return v


def get_optimal_action_value(mdp: MDP) -> np.ndarray:
    # Initialize action-values matrix
    q = np.zeros((len(mdp.env.states), len(mdp.env.actions)))

    # Solution convergence threshold
    epsilon = 1e-10

    delta = float('inf')
    while delta > epsilon:
        delta = 0
        for state_id, state in enumerate(mdp.env.states):
            for action_id, action in enumerate(mdp.env.actions):
                q_old = q[state_id][action_id]
                q_new = __compute_action_value(mdp, state, action, q)
                q[state_id][action_id] = q_new
                delta = max(delta, abs(q_new - q_old))

    return q


def policy_iteration(mdp: MDP) -> (np.ndarray, np.ndarray):
    while True:
        v = policy_evaluation(mdp)
        policy_stable = __policy_improvement(mdp, v)
        if policy_stable:
            break

    return v, mdp.policy


def value_iteration(mdp: MDP) -> (np.ndarray, np.ndarray):
    # Initialize state-values matrix
    v = np.zeros(len(mdp.env.states), dtype=float)

    # Solution convergence threshold
    epsilon = 1e-10

    delta = float('inf')
    while delta > epsilon:
        delta = 0
        for state_id, state in enumerate(mdp.env.states):
            v_old = v[state_id]
            v_new = __compute_state_value(mdp, state, v)
            max_action = __improve_policy(mdp, state, v)
            v[state_id] = v_new
            mdp.policy[state_id] = np.zeros_like(mdp.policy[state_id])
            mdp.policy[state_id][mdp.env.actions.index(max_action)] = 1
            delta = max(delta, abs(v_new - v_old))

    return v, mdp.policy


def policy_evaluation(mdp: MDP) -> np.ndarray:
    # Initialize state-values matrix
    v = np.zeros(len(mdp.env.states), dtype=float)

    # Solution convergence threshold
    epsilon = 1e-10

    delta = float('inf')
    while delta > epsilon:
        delta = 0
        for state_id, state in enumerate(mdp.env.states):
            v_old = v[state_id]
            v_new = __evaluate_state(mdp, state, v)
            v[state_id] = v_new
            delta = max(delta, abs(v_new - v_old))

    return v


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

def __compute_state_value(mdp: MDP, state, state_values) -> float:
    if state in mdp.env.terminal_states:
        return 0

    if state in mdp.env.obstacle_states:
        return 0

    max_value = float('-inf')

    for action in mdp.env.actions:
        v = 0
        probs, next_states = mdp.env.get_next_transitions(state, action)
        for prob, next_state in zip(probs, next_states):
            reward = mdp.rw(state, action, next_state)
            v += prob * (reward + mdp.gamma * state_values[mdp.env.states.index(next_state)])
            if v > max_value:
                max_value = v

    return max_value



def __compute_action_value(mdp: MDP, state, action, q_table) -> float:
    if state in mdp.env.terminal_states:
        return 0

    if state in mdp.env.obstacle_states:
        return 0

    q = 0
    probs, next_states = mdp.env.get_next_transitions(state, action)
    for prob, next_state in zip(probs, next_states):
        reward = mdp.rw(state, action, next_state)
        q = prob * (reward + mdp.gamma*max(q_table[mdp.env.states.index(next_state)]))

    return q


def __evaluate_state(mdp: MDP, state, state_values) -> float:
    if state in mdp.env.terminal_states:
        return 0

    if state in mdp.env.obstacle_states:
        return 0

    v = 0
    for action_id, action in enumerate(mdp.env.actions):
        probs, next_states = mdp.env.get_next_transitions(state, action)
        for prob, next_state in zip(probs, next_states):
            pi_a = mdp.policy[mdp.env.states.index(state)][action_id]
            reward = mdp.rw(state, action, next_state)
            v += pi_a * ( prob * (reward + mdp.gamma * state_values[mdp.env.states.index(next_state)]))

    return v


def __improve_policy(mdp: MDP, state, state_values) -> str:
    max_value = float('-inf')
    max_action = None
    for action in mdp.env.actions:
        v = 0
        probs, next_states = mdp.env.get_next_transitions(state, action)
        for prob, next_state in zip(probs, next_states):
            reward = mdp.rw(state, action, next_state)
            v += prob * (reward + mdp.gamma * state_values[mdp.env.states.index(next_state)])
            if v > max_value:
                max_value = v
                max_action = action

    return max_action


def __policy_improvement(mdp: MDP, v: np.ndarray) -> bool:
    policy_stable = True
    for state_id, state in enumerate(mdp.env.states):
        action_old = np.random.choice(mdp.env.actions, p=mdp.policy[state])
        action_new = __improve_policy(mdp, state, v)

        mdp.policy[state_id] = np.zeros_like(mdp.policy[state])
        mdp.policy[state_id][mdp.env.actions.index(action_new)] = 1

        if action_old != action_new:
            policy_stable = False

    return policy_stable


