import numpy as np
from mdp import MDP
from grid_env import GridEnv


def main():

    gridworld = GridEnv(5, 5)

    def gridworld_reward(state: list, action, next_state):
        if state == [0, 1] and action in GridEnv.AgentActions:
            return 10
        elif state == [0, 3] and action in GridEnv.AgentActions:
            return 5
        elif next_state is None:
            return -1
        else:
            return 0

    mdp_gridworld = MDP(gridworld, gridworld_reward, 0.9)

    print(gridworld.grid)
    print('state value for state {}: {}'.format(mdp_gridworld.environment.state, mdp_gridworld.compute_state_value(mdp_gridworld.environment.state)))
    print('reward: {}'.format(mdp_gridworld.action(GridEnv.AgentActions.RIGHT)))
    print(gridworld.grid)
    print('state value for state {}: {}'.format(mdp_gridworld.environment.state, mdp_gridworld.compute_state_value(mdp_gridworld.environment.state)))
    print('reward: {}'.format(mdp_gridworld.action(GridEnv.AgentActions.DOWN)))
    print(gridworld.grid)
    print('state value for state {}: {}'.format(mdp_gridworld.environment.state, mdp_gridworld.compute_state_value(mdp_gridworld.environment.state)))
    print('reward: {}'.format(mdp_gridworld.action(GridEnv.AgentActions.UP)))
    print(gridworld.grid)
    print('state value for state {}: {}'.format(mdp_gridworld.environment.state, mdp_gridworld.compute_state_value(mdp_gridworld.environment.state)))
    print('reward: {}'.format(mdp_gridworld.action(GridEnv.AgentActions.RIGHT)))
    print(gridworld.grid)
    print('state value for state {}: {}'.format(mdp_gridworld.environment.state, mdp_gridworld.compute_state_value(mdp_gridworld.environment.state)))
    print('reward: {}'.format(mdp_gridworld.action(GridEnv.AgentActions.DOWN)))
    print(gridworld.grid)
    print('state value for state {}: {}'.format(mdp_gridworld.environment.state, mdp_gridworld.compute_state_value(mdp_gridworld.environment.state)))

    print(mdp_gridworld.state_values)
    mdp_gridworld.compute_state_values()
    print(mdp_gridworld.state_values)
    # print(mdp_gridworld.compute_state_values())
    # print(mdp_gridworld.state_values)

    # print(mdp_gridworld.state_value2([0,0]))

    # print(mdp_gridworld.policy)
    # print(mdp_gridworld.policy[GridEnv.AgentActions.UP])


if __name__ == '__main__':
    main()