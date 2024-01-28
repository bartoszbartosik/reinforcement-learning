import numpy as np


def first_visit_prediction(mdp, episodes, policy=None):
    # If policy not given, assume equiprobable
    if policy is None:
        policy = 1/len(mdp.actions)*np.ones((len(mdp.states), len(mdp.actions)))

    # Initialize state-values matrix
    v = np.zeros_like(mdp.states, dtype=float)

    # Returns
    G = np.zeros_like(mdp.states, dtype=float)

    # Visits
    visits = np.zeros_like(mdp.states)

    for _ in range(episodes):
        episode = mdp.generate_episode(steps=10, policy=policy)
        for state, _, reward in reversed(episode):
            G[state] = mdp.gamma*G[state] + reward
            visits[state] += 1

    v = G/visits


def __first_visit_state_value():
    pass
