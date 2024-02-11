import unittest

import numpy as np

import methods.temporal_difference as td
from envs.grid import Grid
from envs.grid_a import GridA
from mdp.markov_decision_process import MDP


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


    def test_one_step_td(self):
        v = td.one_step_td(self.mdp_grid, 1000, 100, 0.2).reshape(self.grid.grid.shape)
        print(v)


    def test_sarsa(self):
        q = td.sarsa(self.mdp_grid_a, 1000, 100, 0.01, 0.1)
        # print(q)


    def test_q_learning(self):
        q = td.qlearning(self.mdp_grid_a, 1000, 1000, 0.1, 0.1)
        # print(q)


    def test_expected_sarsa(self):
        q = td.expected_sarsa(self.mdp_grid_a, 1000, 1000, 0.1, 0.1)
        # print(q)


    def test_double_qlearning(self):
        q = td.double_qlearning(self.mdp_grid_a, 1000, 1000, 0.1, 0.1)
        print(q)
