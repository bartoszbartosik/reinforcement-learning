import unittest

import numpy as np

from mdp import MDP
from qlearning import QLearning

class TestQLearning(unittest.TestCase):


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
        self.ql = QLearning(self.mdp, learning_rate=0.7)


    def test_qtable(self):
        print(self.ql.qtable)



if __name__ == '__main__':
    unittest.main()