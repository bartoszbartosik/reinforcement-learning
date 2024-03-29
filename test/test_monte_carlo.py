import unittest

import numpy as np

from envs.grid import Grid
from envs.grid_a import GridA
from mdp.markov_decision_process import MDP
import methods.monte_carlo as MC


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


    def test_first_visit_prediction(self):
        v = MC.first_visit_prediction(self.mdp_grid_a, 10, 1000).reshape(self.grid_a.grid.shape)
        print(v)


    def test_every_visit_prediction(self):
        v = MC.every_visit_prediction(self.mdp_grid_a, 10, 1000).reshape(self.grid_a.grid.shape)


    def test_exploring_starts(self):
        q = MC.exploring_starts(self.mdp_grid_a, 10, 1000)


    def test_on_policy_first_visit(self):
        q, pi = MC.on_policy_first_visit(self.mdp_grid_a, 1000, 1000, 0.1)
        print('on policy', q, pi)


    def test_off_policy_prediction(self):
        q = MC.off_policy_prediction(self.mdp_grid_a, 10, 1000)


    def test_off_policy_control(self):
        q, pi = MC.off_policy_control(self.mdp_grid_a, 1000, 1000)
        print('off policy', q, pi)

