import numpy as np

from envs.grid import Grid
from envs.grid_a import GridA
from envs.gambler_problem import GamblerProblem
from mdp import MDP
import methods.dynamic_programming as dp


def main():

    grid_a = GridA(5, 5)

    def gridworld_reward(state: tuple, action, next_state):
        if next_state == state and (state != (0, 1) and state != (0, 3)):
            return -1
        elif state == (0, 1):
            return 10
        elif state == (0, 3):
            return 5
        else:
            return 0

    mdp_grid_a = MDP(grid_a, gridworld_reward, 0.9)

    print('Optimal state-values:')
    print(np.round(dp.get_optimal_state_value(mdp_grid_a), 1))
    print('Optimal action-values:')
    print(dp.get_optimal_action_value(mdp_grid_a))

    print('Equiprobable policy evaluation:')
    print(dp.policy_evaluation(mdp_grid_a))
    print('Optimal state-values and policy using value iteration:')
    print(dp.value_iteration(mdp_grid_a))
    print('Optimal state-values and policy using policy iteration:')
    print(dp.policy_iteration(mdp_grid_a))


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    grid = Grid(4, 4)
    grid.set_terminal((0, 0))
    grid.set_terminal((3, 3))
    print(grid.grid)

    def rw(state, action, next_state):
        return -1


    mdp_grid = MDP(grid, rw, 1)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    print('Equiprobable policy evaluation:')
    print(dp.policy_evaluation(mdp_grid))
    print(dp.policy_iteration(mdp_grid))
    for i in range(4):
        for j in range(4):
            print('state[{}, {}], pi(state) = {}'.format(i, j, mdp_grid.policy[i, j]))
        print()
    print(mdp_grid.policy)

    print(dp.value_iteration(mdp_grid))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    gp = GamblerProblem(100, heads_probability = 0.4)
    print(gp.actions)
    print(gp.states)

    def gambler_reward(state, action, next_state):
        if next_state >= 100:
            return 1
        else:
            return 0

    mdp_gp = MDP(gp, gambler_reward, 1)
    print(dp.policy_iteration(mdp_gp))



if __name__ == '__main__':
    main()