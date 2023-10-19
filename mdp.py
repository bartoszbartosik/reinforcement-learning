import random

import numpy as np


class MDP:

    def __init__(self, environment, actions, reward_function, agent_encoding, obstacle_encoding):
        self.environment = environment
        self.actions = actions
        self.reward_function = reward_function
        self.agent_encoding = agent_encoding
        self.obstacle_encoding = obstacle_encoding


    def get_agent_position(self):
        for position, i in np.ndenumerate(self.environment):
            if i == self.agent_encoding:
                return position


    def take_action(self, action):
        action_encoded = self.__encode_action(action)
        agent_position = self.get_agent_position()

        is_valid = self.__validate_action(action)

        if is_valid:
            self.environment[action_encoded], self.environment[agent_position] = (
                self.environment[agent_position], self.environment[action_encoded])

        return self.__get_reward(is_valid)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

    def __get_reward(self, valid):
        return self.reward_function(self.environment, valid)


    def __validate_action(self, action):
        x_new, y_new = action_encoded = self.__encode_action(action)

        if (x_new < 0 or x_new >= len(self.environment) or
                y_new < 0 or y_new >= len(self.environment[0]) or
                self.environment[action_encoded] == self.obstacle_encoding):
            return False

        return True


    def __encode_action(self, action):
        x, y = self.get_agent_position()

        match action:
            case 'l':
                return x, y - 1
            case 'r':
                return x, y + 1
            case 'u':
                return x - 1, y
            case 'd':
                return x + 1, y
            case _:
                return x, y

