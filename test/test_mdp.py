import unittest

import numpy as np

from mdp import MDP

class TestMDP(unittest.TestCase):


    def setUp(self):

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

        self.mdp = MDP(environment, actions, reward_function, agent_encoding=1, obstacle_encoding=2)

    def test_rewards(self):
        reward = self.mdp.take_action('r')
        self.assertEqual(reward, 0)
        print('Action reward: {}'.format(reward))
        print('Environment:\n{}'.format(self.mdp.environment))



if __name__ == '__main__':
    unittest.main()