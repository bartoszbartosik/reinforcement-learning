import numpy as np

from envs.grid import Grid


class MDP:

    def __init__(self, environment: Grid, reward_function, discount_factor):
        self.environment = environment
        self.actions = [i for i in range(len(environment.actions))]
        self.reward_function = reward_function
        self.gamma = discount_factor

        # Initialize as equiprobable random policy
        self.policy = len(self.actions)**(-1)*np.ones((self.environment.width, self.environment.height, len(self.actions)))


    def action(self, action):
        next_state = self.environment.get_next_state(self.environment.state, action)
        reward = self.reward_function(self.environment.state, action, next_state)
        self.environment.action(action)
        return reward

    def get_next_state(self, state, action):
        return self.environment.get_next_state(state, self.environment.actions[action])


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

