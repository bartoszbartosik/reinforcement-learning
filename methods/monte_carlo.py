import numpy as np


def first_visit_preditction(mdp, policy):
    # Initialize state-values matrix
    v = np.zeros_like(mdp.environment.grid)

    

    for i in range(mdp.environment.width):
        for j in range(mdp.environment.height):
            action = np.argmax(policy[(i, j)])
