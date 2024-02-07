import numpy as np

from mdp.markov_decision_process import MDP


def one_step_td(mdp: MDP, episodes, step_size: float, policy=None):
    # If policy not given, assume equiprobable
    if policy is None:
        policy = 1/len(mdp.env.actions)*np.ones((len(mdp.env.states), len(mdp.env.actions)))

    # Initialize state-values matrix
    v = np.zeros(len(mdp.env.states), dtype=float)

    for _ in range(episodes):
        # Initialize state
        available_states = [state for state in mdp.env.states if
                            state not in mdp.env.terminal_states + mdp.env.obstacle_states]

        state = available_states[np.random.choice(len(available_states))]
        while state not in mdp.env.terminal_states:
            state_id = mdp.env.states.index(state)
            action = np.random.choice(mdp.env.actions, p=policy[state_id])
            next_state = mdp.get_next_state(state, action)
            next_state_id = mdp.env.states.index(next_state)
            reward = mdp.rw(state, action, next_state)
            v[state_id] = v[state_id] + step_size * (reward + mdp.gamma*v[next_state_id] - v[state_id])
            state = next_state

    return v


