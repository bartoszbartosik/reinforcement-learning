import random

import numpy as np

from grid_env import GridEnv


class MDP:

    def __init__(self, environment: GridEnv, reward_function, discount_factor):
        self.environment = environment
        self.actions = list(GridEnv.AgentActions)
        self.reward_function = reward_function
        self.gamma = discount_factor

        self.policy = dict()
        for action in self.actions:
            self.policy[action] = 1/len(self.actions)*np.ones(shape=(len(self.environment.grid[0]), len(self.environment.grid[1])))

        self.state_values = np.zeros(shape=(len(self.environment.grid[0]), len(self.environment.grid[1])))


    def action(self, action):
        next_state = self.environment.get_next_state(action, self.environment.state.copy())
        reward = self.reward_function(self.environment.state, action, next_state)
        self.environment.action(action)
        return reward


    def compute_state_value(self, state):
        if self.environment.grid[tuple(state)] == self.environment.GridElements.OBSTACLE.value:
            return 0

        max_value = float('-inf')
        for action in self.actions:
            next_state = self.environment.get_next_state(action, state.copy())
            reward = self.reward_function(state, action, next_state)
            value = reward + self.gamma*self.state_values[tuple(next_state)]
            if value > max_value:
                max_value = value

        return max_value


    def compute_state_values(self):
        for _ in range(1000):
            for i in range(self.environment.width):
                for j in range(self.environment.height):
                    v_new = self.compute_state_value([i, j])
                    self.state_values[i, j] = v_new


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

