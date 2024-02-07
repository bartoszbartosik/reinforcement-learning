import unittest

import numpy as np

from envs.grid import Grid
from envs.grid_a import GridA
from envs.gambler import Gambler
from mdp.markov_decision_process import MDP
import methods.dynamic_programming as dp


class TestDP(unittest.TestCase):

    def setUp(self):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # #   G R I D   A   # # # # # # # # # # # # # # # # # # # # # # # #
        # Set the environment
        self.grid_a = GridA(5, 5)

        # Define the reward function
        def gridworld_reward(state: tuple, action, next_state):
            if next_state == state and (state != (0, 1) and state != (0, 3)):
                return -1
            elif state == (0, 1):
                return 10
            elif state == (0, 3):
                return 5
            else:
                return 0

        # Instantiate MDP for the environment
        self.mdp_grid_a = MDP(self.grid_a, gridworld_reward, 0.9)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # #   G R I D   4 x 4   # # # # # # # # # # # # # # # # # # # # # # #
        # Set the environment
        self.grid = Grid(4, 4, terminals=[(0, 0), (3, 3)])

        # Define the reward function
        def rw(state, action, next_state):
            return -1

        # Instantiate MDP for the environment
        self.mdp_grid = MDP(self.grid, rw, 1)

    def test_optimal_state_value(self):
        optimal_state_values = dp.get_optimal_state_value(self.mdp_grid_a).reshape(self.grid_a.grid.shape)
        expected_optimal_state_values = np.array([
            [22, 24.4, 22, 19.4, 17.5],
            [19.8, 22, 19.8, 17.8, 16],
            [17.8, 19.8, 17.8, 16, 14.4],
            [16, 17.8, 16, 14.4, 13],
            [14.4, 16, 14.4, 13, 11.7]
        ])

        np.testing.assert_allclose(
            np.round(optimal_state_values, 1),
            expected_optimal_state_values
        )

    def test_optimal_action_value(self):
        optimal_action_values = (dp.get_optimal_action_value(self.mdp_grid_a)
                                 .reshape((self.grid_a.height, self.grid_a.width, len(self.mdp_grid_a.env.actions))))
        expected_optimal_state_values = np.array([
            [[18.8, 17.8, 18.8, 22.],
             [24.4, 24.4, 24.4, 24.4],
             [18.8, 17.8, 22., 17.5],
             [19.4, 19.4, 19.4, 19.4],
             [14.7, 14.4, 17.5, 14.7]],

            [[19.8, 16., 16.8, 19.8],
             [22., 17.8, 17.8, 17.8],
             [19.8, 16., 19.8, 16.],
             [17.5, 14.4, 17.8, 14.4],
             [15.7, 13., 16., 13.4]],

            [[17.8, 14.4, 15., 17.8],
             [19.8, 16., 16., 16.],
             [17.8, 14.4, 17.8, 14.4],
             [16., 13., 16., 13.],
             [14.4, 11.7, 14.4, 12.]],

            [[16., 13., 13.4, 16.],
             [17.8, 14.4, 14.4, 14.4],
             [16., 13., 16., 13.],
             [14.4, 11.7, 14.4, 11.7],
             [13., 10.5, 13., 10.7]],

            [[14.4, 12., 12., 14.4],
             [16., 13.4, 13., 13.],
             [14.4, 12., 14.4, 11.7],
             [13., 10.7, 13., 10.5],
             [11.7, 9.5, 11.7, 9.5]]
        ])

        np.testing.assert_allclose(
            np.round(optimal_action_values, 1),
            expected_optimal_state_values
        )

    def test_equiprobable_policy(self):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # # #   G R I D   A   # # # # # # # # # # # # # # # # # # # # # # # #
        equiprobable_policy_values = dp.policy_evaluation(self.mdp_grid_a).reshape(self.grid_a.grid.shape)
        expected_equiprobable_policy_values = np.array([
            [3.3, 8.8, 4.4, 5.3, 1.5],
            [1.5, 3, 2.3, 1.9, 0.5],
            [0.1, 0.7, 0.7, 0.4, -0.4],
            [-1., -0.4, -0.4, -0.6, -1.2],
            [-1.9, -1.3, -1.2, -1.4, -2.]
        ])

        np.testing.assert_allclose(
            np.round(equiprobable_policy_values, 1),
            expected_equiprobable_policy_values
        )

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # # # # # #   G R I D   4 x 4   # # # # # # # # # # # # # # # # # # # # # # #
        equiprobable_policy_values = dp.policy_evaluation(self.mdp_grid).reshape(self.grid.grid.shape)
        expected_equiprobable_policy_values = np.array([
            [0, -14., -20., -22],
            [-14., -18., -20., -20],
            [-20., -20., -18., -14],
            [-22., -20., -14., 0]
        ])

        np.testing.assert_allclose(
            np.round(equiprobable_policy_values, 1),
            expected_equiprobable_policy_values
        )

    def test_value_iteration(self):
        value_iteration_values = dp.value_iteration(self.mdp_grid_a)[0].reshape(self.grid_a.grid.shape)
        expected_value_iteration_values = np.array([
            [22, 24.4, 22, 19.4, 17.5],
            [19.8, 22, 19.8, 17.8, 16],
            [17.8, 19.8, 17.8, 16, 14.4],
            [16, 17.8, 16, 14.4, 13],
            [14.4, 16, 14.4, 13, 11.7]
        ])

        np.testing.assert_allclose(
            np.round(value_iteration_values, 1),
            expected_value_iteration_values
        )

    def test_policy_iteration(self):
        policy_iteration_values = dp.value_iteration(self.mdp_grid_a)[0].reshape(self.grid_a.grid.shape)
        expected_policy_iteration_values = np.array([
            [22, 24.4, 22, 19.4, 17.5],
            [19.8, 22, 19.8, 17.8, 16],
            [17.8, 19.8, 17.8, 16, 14.4],
            [16, 17.8, 16, 14.4, 13],
            [14.4, 16, 14.4, 13, 11.7]
        ])

        np.testing.assert_allclose(
            np.round(policy_iteration_values, 1),
            expected_policy_iteration_values
        )


if __name__ == '__main__':
    unittest.main()
