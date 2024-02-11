import numpy as np

from mdp.markov_decision_process import MDP


def one_step_td(mdp: MDP, episodes, steps, step_size: float, policy=None):
    # If policy not given, assume equiprobable
    if policy is None:
        policy = mdp.equiprobable_policy()

    # Initialize state-values matrix
    v = np.zeros(len(mdp.env.states), dtype=float)

    for _ in range(episodes):
        step = 0

        state = mdp.env.initial_state
        while state not in mdp.env.terminal_states and step < steps:
            s = mdp.env.states.index(state)

            # Choose action given by the policy
            action = np.random.choice(mdp.env.actions, p=policy[s])

            # Get the next state
            next_state = mdp.get_next_state(state, action)
            sp = mdp.env.states.index(next_state)

            # Get the reward
            reward = mdp.rw(state, action, next_state)

            # Compute the state value
            v[s] = v[s] + step_size * (reward + mdp.gamma*v[sp] - v[s])

            # Assign the next state as current for next iteration
            state = next_state

            # Episode step increment
            step += 1

    return v


def sarsa(mdp: MDP, episodes, steps, step_size: float, epsilon: float):
    # Initialize action-values matrix
    q = np.zeros((len(mdp.env.states), len(mdp.env.actions)))

    # Define an epsilon-greedy action policy
    def take_action(state):
        if np.random.random() >= epsilon:
            return mdp.env.actions[np.argmax(q[state])]
        else:
            return np.random.choice(mdp.env.actions)

    # For each episode
    for _ in range(episodes):
        step = 0

        # Initialize state
        state = mdp.env.initial_state
        while state not in mdp.env.terminal_states and step < steps:
            s = mdp.env.states.index(state)

            # Choose action greedily derived from the q values
            action = take_action(s)
            a = mdp.env.actions.index(action)

            # Get the next state
            next_state = mdp.get_next_state(state, action)
            sp = mdp.env.states.index(next_state)

            # Get the reward
            reward = mdp.rw(state, action, next_state)

            # Get the next action
            next_action = take_action(sp)
            ap = mdp.env.actions.index(next_action)

            # Compute the state-action value
            q[s][a] = q[s][a] + step_size * (reward + mdp.gamma*q[sp][ap] - q[s][a])

            # Assign the next state as current for next iteration
            state = next_state

            # Episode step increment
            step += 1

    return q


def qlearning(mdp: MDP, episodes, steps, step_size: float, epsilon: float):
    # Initialize action-values matrix
    q = np.zeros((len(mdp.env.states), len(mdp.env.actions)))

    # Define an epsilon-greedy action policy
    def take_action(state):
        if np.random.random() >= epsilon:
            return mdp.env.actions[np.argmax(q[state])]
        else:
            return np.random.choice(mdp.env.actions)

    # For each episode
    for _ in range(episodes):
        step = 0

        # Initialize state
        state = mdp.env.initial_state
        while state not in mdp.env.terminal_states and step < steps:
            s = mdp.env.states.index(state)

            # Choose action greedily derived from the q values
            action = take_action(s)
            a = mdp.env.actions.index(action)

            # Get the next state
            next_state = mdp.get_next_state(state, action)
            sp = mdp.env.states.index(next_state)

            # Get the reward
            reward = mdp.rw(state, action, next_state)

            # Compute the state-action value
            q[s][a] = q[s][a] + step_size * (reward + mdp.gamma * np.max(q[sp]) - q[s][a])

            # Assign the next state as current for next iteration
            state = next_state

            # Episode step increment
            step += 1

    return q


def expected_sarsa(mdp: MDP, episodes, steps, step_size: float, epsilon: float, policy=None):
    # If policy not given, assume equiprobable
    if policy is None:
        policy = mdp.equiprobable_policy()

    # Initialize action-values matrix
    q = np.zeros((len(mdp.env.states), len(mdp.env.actions)))

    # Define an epsilon-greedy action policy
    def take_action(state):
        if np.random.random() >= epsilon:
            return mdp.env.actions[np.argmax(q[state])]
        else:
            return np.random.choice(mdp.env.actions)

    # For each episode
    for _ in range(episodes):
        step = 0

        # Initialize state
        state = mdp.env.initial_state
        while state not in mdp.env.terminal_states and step < steps:
            s = mdp.env.states.index(state)

            # Choose action greedily derived from the q values
            action = take_action(s)
            a = mdp.env.actions.index(action)

            # Get the next state
            next_state = mdp.get_next_state(state, action)
            sp = mdp.env.states.index(next_state)

            # Get the reward
            reward = mdp.rw(state, action, next_state)

            # Compute the state-action value
            q[s][a] = q[s][a] + step_size * (reward + mdp.gamma * np.sum([policy[sp][a]*q[sp][a] for a in range(len(q[sp]))]) - q[s][a])

            # Assign the next state as current for next iteration
            state = next_state

            # Episode step increment
            step += 1

    return q





