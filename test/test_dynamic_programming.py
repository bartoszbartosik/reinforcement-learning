import unittest

import numpy as np

from envs.grid_a import GridA
from mdp import MDP
import methods.dynamic_programming as dp

class TestDP(unittest.TestCase):


    def setUp(self):

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

        self.mdp_grid_a = MDP(grid_a, gridworld_reward, 0.9)


    def test_optimal_state_value(self):
        optimal_state_values = dp.get_optimal_state_value(self.mdp_grid_a)
        expected_optimal_state_values = np.array([
                [22,    24.4,   22,     19.4,   17.5],
                [19.8,  22,     19.8,   17.8,   16],
                [17.8,  19.8,   17.8,   16,     14.4],
                [16,    17.8,   16,     14.4,   13],
                [14.4,  16,     14.4,   13,     11.7]
            ])

        self.assertEqual(
            np.round(optimal_state_values, 1),
            expected_optimal_state_values
        )



if __name__ == '__main__':
    unittest.main()