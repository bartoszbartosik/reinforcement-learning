import numpy as np

from envs.grid import Grid
from mdp import MDP
from envs.grid_a import GridA
from dp import DynamicProgramming
import pprint


def main():

    gridworld = GridA(5, 5)

    def gridworld_reward(state: list, action, next_state):
        if next_state == state and (state != [0, 1] and state != [0, 3]):
            return -1
        elif state == [0, 1] and action in GridA.AgentActions:
            return 10
        elif state == [0, 3] and action in GridA.AgentActions:
            return 5
        else:
            return 0

    mdp_gridworld = MDP(gridworld, gridworld_reward, 0.9)

    print(gridworld.grid)
    print('*action*')
    print('reward: {}'.format(mdp_gridworld.action(GridA.AgentActions.RIGHT)))
    print(gridworld.grid)
    print('*action*')
    print('reward: {}'.format(mdp_gridworld.action(GridA.AgentActions.DOWN)))
    print(gridworld.grid)
    print('*action*')
    print('reward: {}'.format(mdp_gridworld.action(GridA.AgentActions.UP)))
    print(gridworld.grid)
    print('*action*')
    print('reward: {}'.format(mdp_gridworld.action(GridA.AgentActions.RIGHT)))
    print(gridworld.grid)
    print('*action*')
    print('reward: {}'.format(mdp_gridworld.action(GridA.AgentActions.DOWN)))
    print(gridworld.grid)

    dp = DynamicProgramming(mdp_gridworld)

    print(dp.compute_state_values())

    pp = pprint.PrettyPrinter()
    pp.pprint(dp.compute_action_values())

    print(mdp_gridworld.policy)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    grid = Grid(4, 4)
    grid.set_terminal([0, 0])
    grid.set_terminal([3, 3])
    print(grid.grid)

    def rw(state, action, next_state):
        return -1


    mdp_g = MDP(grid, rw, 1)
    dp_g = DynamicProgramming(mdp_g)

    print(dp_g.policy_evaluation())


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    print(mdp_g.policy)

if __name__ == '__main__':
    main()