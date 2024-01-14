import numpy as np

from envs.grid import Grid
from envs.grid_a import GridA
from envs.gambler_problem import GamblerProblem
from mdp import MDP
from dp import DynamicProgramming


def main():

    gridworld = GridA(5, 5)

    def gridworld_reward(state: tuple, action, next_state):
        if next_state == state and (state != (0, 1) and state != (0, 3)):
            return -1
        elif state == (0, 1):
            return 10
        elif state == (0, 3):
            return 5
        else:
            return 0

    mdp_gridworld = MDP(gridworld, gridworld_reward, 0.9)

    print(gridworld.grid)
    print('*action*')
    print('reward: {}'.format(mdp_gridworld.action('RIGHT')))
    print(gridworld.grid)
    print('*action*')
    print('reward: {}'.format(mdp_gridworld.action('DOWN')))
    print(gridworld.grid)
    print('*action*')
    print('reward: {}'.format(mdp_gridworld.action('UP')))
    print(gridworld.grid)
    print('*action*')
    print('reward: {}'.format(mdp_gridworld.action('RIGHT')))
    print(gridworld.grid)
    print('*action*')
    print('reward: {}'.format(mdp_gridworld.action('DOWN')))
    print(gridworld.grid)

    dp = DynamicProgramming(mdp_gridworld)

    print('Optimal state-values:')
    print(dp.get_optimal_state_value())
    print('Optimal action-values:')
    print(dp.get_optimal_action_value())

    print('Equiprobable policy evaluation:')
    print(dp.policy_evaluation())
    print('Optimal state-values and policy using value iteration:')
    print(dp.value_iteration())
    print('Optimal state-values and policy using policy iteration:')
    print(dp.policy_iteration())


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    grid = Grid(4, 4)
    grid.set_terminal((0, 0))
    grid.set_terminal((3, 3))
    print(grid.grid)

    def rw(state, action, next_state):
        return -1


    mdp_g = MDP(grid, rw, 1)
    dp_g = DynamicProgramming(mdp_g)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    print('Equiprobable policy evaluation:')
    print(dp_g.policy_evaluation())
    print(dp_g.policy_iteration())
    for i in range(4):
        for j in range(4):
            print('state[{}, {}], pi(state) = {}'.format(i, j, dp_g.pi[i, j]))
        print()
    print(dp_g.pi)

    print(dp_g.value_iteration())

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
    dp_gp = DynamicProgramming(mdp_gp)
    print(dp_gp.policy_iteration())



if __name__ == '__main__':
    main()