import numpy as np
from mdp import MDP
from qlearning import QLearning

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


    actions = ('l', 'r', 'u', 'd')
    episodes = 5

    mdp = MDP(environment, actions, reward_function, agent_encoding=1, obstacle_encoding=2)


    print('Environment:\n{}'.format(mdp.environment))
    print('Actions: {}'.format(mdp.actions))
    print('Current agent position: {}'.format(mdp.environment[mdp.get_agent_position()]))

    print('Action reward: {}'.format(mdp.take_action('r')))
    print('Environment:\n{}'.format(mdp.environment))
    print('Action reward: {}'.format(mdp.take_action('r')))
    print('Environment:\n{}'.format(mdp.environment))
    print('Action reward: {}'.format(mdp.take_action('r')))
    print('Environment:\n{}'.format(mdp.environment))
    print('Action reward: {}'.format(mdp.take_action('u')))
    print('Environment:\n{}'.format(mdp.environment))
    print('Action reward: {}'.format(mdp.take_action('u')))
    print('Environment:\n{}'.format(mdp.environment))
    print('Action reward: {}'.format(mdp.take_action('u')))
    print('Environment:\n{}'.format(mdp.environment))
    print('Action reward: {}'.format(mdp.take_action('l')))
    print('Environment:\n{}'.format(mdp.environment))
    print('Action reward: {}'.format(mdp.take_action('l')))
    print('Environment:\n{}'.format(mdp.environment))


if __name__ == '__main__':
    main()