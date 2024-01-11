import numpy as np

from envs.grid import Grid


class MDP:

    def __init__(self, environment: Grid, reward_function, discount_factor):
        self.environment = environment
        self.actions = list(Grid.AgentActions)
        self.reward_function = reward_function
        self.gamma = discount_factor

        self.policy = len(self.environment.AgentActions)**(-1)*np.ones((self.environment.width, self.environment.height, len(self.actions)))


    def action(self, action):
        next_state = self.environment.get_next_state(self.environment.state.copy(), action)
        reward = self.reward_function(self.environment.state, action, next_state)
        self.environment.action(action)
        return reward


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

