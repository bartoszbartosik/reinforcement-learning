import random

import numpy as np

from grid_env import GridEnv, GridElements, AgentActions


class MDP:

    def __init__(self, environment: GridEnv, reward_function):
        self.environment = environment
        self.actions = list(AgentActions)
        self.reward_function = reward_function


    def action(self, action):
        self.environment.action(action)
        return self.reward_function(self.environment, action)


    def reward(self, action):
        pass


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

