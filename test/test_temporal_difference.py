import unittest

import numpy as np

import methods.temporal_difference as td
from envs.grid import Grid
from mdp.markov_decision_process import MDP


class TestDP(unittest.TestCase):

    def setUp(self):
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
        v = td.one_step_td(self.mdp_grid, 1000, 0.2).reshape(self.grid.grid.shape)
        print(v)
