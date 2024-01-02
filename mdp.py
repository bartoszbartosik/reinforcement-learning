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

        self.v_values = np.zeros(shape=(len(self.environment.grid[0]), len(self.environment.grid[1])))
        self.q_values = dict()
        for i in range(self.environment.width):
            for j in range(self.environment.height):
                self.q_values[(i, j)] = np.zeros(shape=(len(self.environment.AgentActions)))


    def action(self, action):
        next_state = self.environment.get_next_state(self.environment.state.copy(), action)
        reward = self.reward_function(self.environment.state, action, next_state)
        self.environment.action(action)
        return reward


    def compute_state_value(self, state):
        if self.environment.grid[tuple(state)] == self.environment.GridElements.OBSTACLE.value:
            return 0

        max_value = float('-inf')
        for action in self.actions:
            next_state = self.environment.get_next_state(state.copy(), action)
            reward = self.reward_function(state, action, next_state)
            state_value = reward + self.gamma*self.v_values[tuple(next_state)]
            if state_value > max_value:
                max_value = state_value

        return max_value


    def compute_state_values(self):
        for _ in range(1000):
            for i in range(self.environment.width):
                for j in range(self.environment.height):
                    v = self.compute_state_value([i, j])
                    self.v_values[i, j] = v


    def compute_action_value(self, state, action):
        if self.environment.grid[tuple(state)] == self.environment.GridElements.OBSTACLE.value:
            return 0

        next_state = self.environment.get_next_state(state.copy(), action)
        reward = self.reward_function(state, action, next_state)
        q = reward + self.gamma*max(self.q_values[tuple(next_state)])

        return q


    def compute_action_values(self):
        for _ in range(1000):
            for i in range(self.environment.width):
                for j in range(self.environment.height):
                    for action in self.environment.AgentActions:
                        q = self.compute_action_value([i, j], action)
                        self.q_values[i, j][action.value] = q


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

