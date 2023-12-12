import numpy as np
from mdp import MDP
from grid_env import GridEnv


def main():

    gridworld = GridEnv(5, 5)

    def gridworld_reward(env: GridEnv, action: GridEnv.AgentActions):
        if env.agent_pos == [0,1]:
            return 10
        elif env.agent_pos == [0,3]:
            return 5
        elif not env.validate_action(action):
            return -1
        else:
            return 0

    mdp_gridworld = MDP(gridworld, gridworld_reward)

    print(gridworld.grid)
    print(mdp_gridworld.action(GridEnv.AgentActions.RIGHT))
    print(mdp_gridworld.action(GridEnv.AgentActions.RIGHT))
    print(mdp_gridworld.action(GridEnv.AgentActions.UP))
    print(mdp_gridworld.action(GridEnv.AgentActions.RIGHT))
    print(mdp_gridworld.action(GridEnv.AgentActions.DOWN))
    print(gridworld.grid)
    print(mdp_gridworld.policy)
    print(mdp_gridworld.policy[GridEnv.AgentActions.UP])

    # print(mdp_gridworld.policy[])


if __name__ == '__main__':
    main()