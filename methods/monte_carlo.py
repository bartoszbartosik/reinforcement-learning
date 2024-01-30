import numpy as np

from mdp.markov_decision_process import MDP


def first_visit_prediction(mdp: MDP, episodes, steps, policy=None):
    # If policy not given, assume equiprobable
    if policy is None:
        policy = 1/len(mdp.env.actions)*np.ones((len(mdp.env.states), len(mdp.env.actions)))

    # Initialize state-values matrix
    v = np.zeros(len(mdp.env.states), dtype=float)

    # Visits
    visits = np.zeros_like(v)

    for _ in range(episodes):
        # Generate MDP episode
        episode = mdp.generate_episode(steps=steps, policy=policy)

        # Initialize expected return
        G = 0

        # For each step (T-1, T-2, ..., 0) of an episode
        for state, _, reward in reversed(episode):
            # Increment expected return
            G = mdp.gamma * G + reward
            v[mdp.env.states.index(state)] += G
            visits[mdp.env.states.index(state)] += 1


    # Compute average expected return
    v /= visits

    return v


def __first_visit_state_value():
    pass
