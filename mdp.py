import random

import numpy as np

from grid_env import GridEnv


class MDP:

    def __init__(self, environment: GridEnv, reward_function):
        self.environment = environment
        self.actions = list(GridEnv.AgentActions)
        self.reward_function = reward_function

        self.policy = dict()
        for action in self.actions:
            self.policy[action] = 1/len(self.actions)*np.ones(shape=(len(self.environment.grid[0]), len(self.environment.grid[1])))


    def action(self, action):
        self.environment.action(action)
        return self.reward_function(self.environment, action)


    def reward(self, action):
        pass


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

