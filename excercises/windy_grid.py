import numpy as np
from matplotlib import pyplot as plt

from envs.grid_windy import GridWindy
from mdp.markov_decision_process import MDP
import methods.temporal_difference as TD


def main():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # #   G A M B L E R ' S   P R O B L E M   # # # # # # # # # # # # # # # # # #
    grid = GridWindy()


    def reward_function(state, action, next_state):
        return -1

    mdp = MDP(grid, reward_function, 0.5)

    grid.set_agent((3, 1))
    mdp.action('RIGHT')
    mdp.action('RIGHT')
    mdp.action('RIGHT')
    mdp.action('DOWN')
    mdp.action('UP')
    mdp.action('RIGHT')

    epsilon = 0.1
    alpha = 0.5

    q = TD.sarsa(mdp, 10000, 1000000, step_size=alpha, epsilon=epsilon).reshape(grid.height, grid.width, -1)
    print(q)

    results = np.zeros_like(grid.grid, dtype=str)
    for i in range(len(q)):
        for j in range(len(q[0])):
            action = np.argmax(q[i][j])
            results[i, j] = grid.actions[action]
            if (i, j) == grid.initial_state:
                results[i, j] = 's'
            if (i, j) in grid.terminal_states:
                results[i, j] = 't'

    print(results)





if __name__ == '__main__':
    main()