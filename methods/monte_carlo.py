import numpy as np


def first_visit_prediction(mdp, episodes, steps, policy=None):
    # If policy not given, assume equiprobable
    if policy is None:
        policy = 1/len(mdp.actions)*np.ones((len(mdp.states), len(mdp.actions)))

    # Initialize state-values matrix
    v = np.zeros(len(mdp.states), dtype=float)

    # Visits
    visits = np.zeros_like(v)

    for _ in range(episodes):
        episode = mdp.generate_episode(steps=steps, policy=policy)
        G = 0
        for state, _, reward in reversed(episode):
            G = mdp.gamma * G + reward
            v[mdp.states.index(state)] += G
            visits[mdp.states.index(state)] += 1


    v /= visits

    return v


def __first_visit_state_value():
    pass
