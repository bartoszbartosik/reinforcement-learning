import numpy as np

from mdp.markov_decision_process import MDP


def one_step_td(mdp: MDP, episodes, step_size: float, policy=None):
    # If policy not given, assume equiprobable
    if policy is None:
        policy = 1/len(mdp.env.actions)*np.ones((len(mdp.env.states), len(mdp.env.actions)))

    # Number of steps
    steps = 1/step_size

    # Initialize state-values matrix
    v = np.zeros(len(mdp.env.states), dtype=float)

    for _ in range(episodes):
        step = 0

        # Initialize state
        available_states = [state for state in mdp.env.states if
                            state not in mdp.env.terminal_states + mdp.env.obstacle_states]

        state = available_states[np.random.choice(len(available_states))]
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


def sarsa(mdp: MDP, episodes, step_size: float, epsilon: float):
    # Number of steps
    steps = 1/step_size

    # Initialize action-values matrix
    q = 1/len(mdp.env.actions)*np.ones((len(mdp.env.states), len(mdp.env.actions)))

    for _ in range(episodes):
        step = 0

        # Initialize state
        available_states = [state for state in mdp.env.states if
                            state not in mdp.env.terminal_states + mdp.env.obstacle_states]

        take_action = lambda s: mdp.env.actions[np.argmax(q[s])] if np.random.random() >= epsilon else np.random.choice(mdp.env.actions)

        state = available_states[np.random.choice(len(available_states))]
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



