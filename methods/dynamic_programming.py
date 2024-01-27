import numpy as np
from mdp.mdp import MDP

# TODO: remove policy attribute from MDP class

def get_optimal_state_value(mdp: MDP) -> np.ndarray:
    # Initialize state-values matrix
    v = np.zeros_like(mdp.states, dtype=float)

    # Solution convergence threshold
    epsilon = 1e-10

    delta = float('inf')
    while delta > epsilon:
        delta = 0
        for state in mdp.states:
            v_old = v[state]
            v_new = __compute_state_value(mdp, state, v)
            v[state] = v_new
            delta = max(delta, abs(v_new - v_old))

    return v


def get_optimal_action_value(mdp: MDP) -> np.ndarray:
    # Initialize action-values tensor
    q = np.zeros((len(mdp.states), len(mdp.actions)))

    # Solution convergence threshold
    epsilon = 1e-10

    delta = float('inf')
    while delta > epsilon:
        delta = 0
        for state in mdp.states:
            for action in mdp.actions:
                q_old = q[state][action]
                q_new = __compute_action_value(mdp, state, action, q)
                q[state][action] = q_new
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
    v = np.zeros_like(mdp.states, dtype=float)

    # Solution convergence threshold
    epsilon = 1e-10

    delta = float('inf')
    while delta > epsilon:
        delta = 0
        for index, state in enumerate(mdp.states):
            v_old = v[state]
            v_new = __compute_state_value(mdp, state, v)
            max_action = __improve_policy(mdp, state, v)
            v[state] = v_new
            mdp.policy[index] = np.zeros_like(mdp.policy[index])
            mdp.policy[index][max_action] = 1
            delta = max(delta, abs(v_new - v_old))

    return v, mdp.policy


def policy_evaluation(mdp: MDP) -> np.ndarray:
    # Initialize state-values matrix
    v = np.zeros_like(mdp.states, dtype=float)

    # Solution convergence threshold
    epsilon = 1e-10

    delta = float('inf')
    while delta > epsilon:
        delta = 0
        for state in mdp.states:
            v_old = v[state]
            v_new = __evaluate_state(mdp, state, v)
            v[state] = v_new
            delta = max(delta, abs(v_new - v_old))

    return v


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

def __compute_state_value(mdp: MDP, state, state_values) -> float:
    if state in mdp.terminal_states:
        return 0

    if state in mdp.obstacle_states:
        return 0

    max_value = float('-inf')
    for action in mdp.actions:
        next_state = mdp.get_next_state(state, action)
        reward = mdp.reward_function(mdp.environment.states[state], mdp.environment.actions[action], mdp.environment.states[next_state])
        state_value = reward + mdp.gamma*state_values[next_state]
        if state_value > max_value:
            max_value = state_value

    return max_value


def __compute_action_value(mdp: MDP, state, action, q_table) -> float:
    if state in mdp.terminal_states:
        return 0

    if state in mdp.obstacle_states:
        return 0

    next_state = mdp.get_next_state(state, action)
    reward = mdp.reward_function(mdp.environment.states[state], mdp.environment.actions[action], mdp.environment.states[next_state])
    q = reward + mdp.gamma*max(q_table[next_state])

    return q


def __evaluate_state(mdp: MDP, state, state_values) -> float:
    if state in mdp.terminal_states:
        return 0

    if state in mdp.obstacle_states:
        return 0

    v = 0
    for action in mdp.actions:
        pi_a = mdp.policy[state][action]
        next_state = mdp.get_next_state(state, action)
        reward = mdp.reward_function(mdp.environment.states[state], mdp.environment.actions[action], mdp.environment.states[next_state])
        v += pi_a * (reward + mdp.gamma * state_values[next_state])

    return v


def __improve_policy(mdp: MDP, state, state_values) -> str:
    max_value = float('-inf')
    max_action = None
    for action in mdp.actions:
        next_state = mdp.get_next_state(state, action)
        reward = mdp.reward_function(mdp.environment.states[state], mdp.environment.actions[action], mdp.environment.states[next_state])
        state_value = reward + mdp.gamma*state_values[next_state]
        if state_value > max_value:
            max_value = state_value
            max_action = action

    return max_action


def __policy_improvement(mdp: MDP, v: np.ndarray) -> bool:
    policy_stable = True
    for state in mdp.states:
        action_old = np.argmax(mdp.policy[state])
        action_new = __improve_policy(mdp, state, v)

        mdp.policy[state] = np.zeros_like(mdp.policy[state])
        mdp.policy[state][action_new] = 1

        if action_old != action_new:
            policy_stable = False

    return policy_stable


