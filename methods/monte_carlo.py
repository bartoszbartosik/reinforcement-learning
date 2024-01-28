import numpy as np


def first_visit_prediction(mdp, policy):
    # Initialize state-values matrix
    v = np.zeros_like(mdp.states, dtype=float)


