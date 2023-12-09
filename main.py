import numpy as np
from mdp import MDP
from grid_env import GridEnv, GridElements, AgentActions


def main():

    def reward_function(env, valid):
        if valid:
            if env[0, 1] == 1:
                return 10
            return 0
        return -10

    environment = np.array([[0, 3, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [2, 2, 2, 0, 0],
                            [1, 0, 0, 0, 0]])

    grid = GridEnv(5, 4)
    grid.set_obstacle((2,0))
    grid.set_obstacle((2,1))
    grid.set_obstacle((2,2))
    grid.set_agent([3,0])
    grid.set_reward((0,1))

    print(grid.grid)
    print(grid.action(AgentActions.RIGHT))
    print(grid.grid)

    mdp = MDP(grid, reward_function)

    mdp.action(AgentActions.RIGHT)
    print(mdp.environment.grid)


    # print('Environment:\n{}'.format(mdp.environment))
    # print('Actions: {}'.format(mdp.actions))
    # print('Current agent position: {}'.format(mdp.environment[mdp.get_agent_position()]))
    #
    # print('Action reward: {}'.format(mdp.take_action('r')))
    # print('Environment:\n{}'.format(mdp.environment))
    # print('Action reward: {}'.format(mdp.take_action('r')))
    # print('Environment:\n{}'.format(mdp.environment))
    # print('Action reward: {}'.format(mdp.take_action('r')))
    # print('Environment:\n{}'.format(mdp.environment))
    # print('Action reward: {}'.format(mdp.take_action('u')))
    # print('Environment:\n{}'.format(mdp.environment))
    # print('Action reward: {}'.format(mdp.take_action('u')))
    # print('Environment:\n{}'.format(mdp.environment))
    # print('Action reward: {}'.format(mdp.take_action('u')))
    # print('Environment:\n{}'.format(mdp.environment))
    # print('Action reward: {}'.format(mdp.take_action('l')))
    # print('Environment:\n{}'.format(mdp.environment))
    # print('Action reward: {}'.format(mdp.take_action('l')))
    # print('Environment:\n{}'.format(mdp.environment))


if __name__ == '__main__':
    main()