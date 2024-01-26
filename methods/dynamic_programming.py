import numpy as np

import mdp as MDP


def get_optimal_state_value(mdp: MDP) -> np.ndarray:
    # Initialize state-values matrix
    v = np.zeros_like(mdp.environment.grid)

    # Solution convergence threshold
    epsilon = 1e-10

    delta = float('inf')
    while delta > epsilon:
        delta = 0
        for i in range(mdp.environment.width):
            for j in range(mdp.environment.height):
                v_old = v[i, j]
                v_new = __compute_state_value(mdp,(i, j), v)
                v[i, j] = v_new
                delta = max(delta, abs(v_new - v_old))

    return v


def get_optimal_action_value(mdp: MDP) -> np.ndarray:
    # Initialize action-values tensor
    q = np.zeros((mdp.environment.width, mdp.environment.height, len(mdp.actions)))

    # Solution convergence threshold
    epsilon = 1e-10

    delta = float('inf')
    while delta > epsilon:
        delta = 0
        for i in range(mdp.environment.width):
            for j in range(mdp.environment.height):
                for action in mdp.actions:
                    q_old = q[i, j][action]
                    q_new = __compute_action_value(mdp, (i, j), action, q)
                    q[i, j][action] = q_new
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
    v = np.zeros_like(mdp.environment.grid)

    # Solution convergence threshold
    epsilon = 1e-10

    delta = float('inf')
    while delta > epsilon:
        delta = 0
        for i in range(mdp.environment.width):
            for j in range(mdp.environment.height):
                v_old = v[i, j]
                v_new = __compute_state_value(mdp, (i, j), v)
                max_action = __improve_policy(mdp, (i, j), v)
                v[i, j] = v_new
                mdp.policy[(i, j)] = np.zeros_like(mdp.policy[(i, j)])
                mdp.policy[i, j][max_action] = 1
                delta = max(delta, abs(v_new - v_old))

    return v, mdp.policy


def policy_evaluation(mdp) -> np.ndarray:
    # Initialize state-values matrix
    v = np.zeros_like(mdp.environment.grid)

    # Solution convergence threshold
    epsilon = 1e-10

    delta = float('inf')
    while delta > epsilon:
        delta = 0
        for i in range(mdp.environment.width):
            for j in range(mdp.environment.height):
                v_old = v[i, j]
                v_new = __evaluate_state(mdp, (i, j), v)
                v[i, j] = v_new
                delta = max(delta, abs(v_new - v_old))

    return v


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

def __compute_state_value(mdp, state, state_values) -> float:
    if mdp.environment.grid[state] == mdp.environment.GridElements.OBSTACLE.value:
        return 0

    if mdp.environment.grid[state] == mdp.environment.GridElements.TERMINAL.value:
        return 0

    max_value = float('-inf')
    for action in mdp.actions:
        next_state = mdp.get_next_state(state, action)
        reward = mdp.reward_function(state, action, next_state)
        state_value = reward + mdp.gamma * state_values[next_state]
        if state_value > max_value:
            max_value = state_value

    return max_value


def __compute_action_value(mdp: MDP, state, action, q_table) -> float:
    if mdp.environment.grid[state] == mdp.environment.GridElements.OBSTACLE.value:
        return 0

    if mdp.environment.grid[state] == mdp.environment.GridElements.TERMINAL.value:
        return 0

    next_state = mdp.get_next_state(state, action)
    reward = mdp.reward_function(state, action, next_state)
    q = reward + mdp.gamma * max(q_table[next_state])

    return q


def __evaluate_state(mdp: MDP, state, state_values) -> float:
    if mdp.environment.grid[state] == mdp.environment.GridElements.OBSTACLE.value:
        return 0

    if mdp.environment.grid[state] == mdp.environment.GridElements.TERMINAL.value:
        return 0

    v = 0
    for pi_a, action in zip(mdp.policy[state], mdp.actions):
        next_state = mdp.get_next_state(state, action)
        reward = mdp.reward_function(state, action, next_state)
        v += pi_a * (reward + mdp.gamma * state_values[next_state])

    return v


def __improve_policy(mdp: MDP, state, state_values) -> str:
    max_value = float('-inf')
    max_action = None
    for action in mdp.actions:
        next_state = mdp.get_next_state(state, action)
        reward = mdp.reward_function(state, action, next_state)
        state_value = reward + mdp.gamma * state_values[next_state]
        if state_value > max_value:
            max_value = state_value
            max_action = action

    return max_action


def __policy_improvement(mdp: MDP, v: np.ndarray) -> bool:
    policy_stable = True
    for i in range(mdp.environment.width):
        for j in range(mdp.environment.height):
            action_old = np.argmax(mdp.policy[(i, j)])
            action_new = __improve_policy(mdp, [i, j], v)

            mdp.policy[(i, j)] = np.zeros_like(mdp.policy[(i, j)])
            mdp.policy[(i, j)][action_new] = 1

            if action_old != action_new:
                policy_stable = False

    return policy_stable


