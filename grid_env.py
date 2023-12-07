from enum import Enum

import numpy as np

class GridElements(Enum):
    FREE = 0
    AGENT = 1
    OBSTACLE = 2
    REWARD = 3

class AgentActions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class GridEnv:

    def __init__(self, width, height):
        self.width = width      # columns
        self.height = height    # rows

        self.grid = GridElements.FREE*np.ones(shape=(height, width))


    def set_agent(self, agent_position: tuple):
        self.grid[agent_position] = GridElements.AGENT

    def set_obstacle(self, obstacle_position: tuple):
        self.grid[obstacle_position] = GridElements.OBSTACLE

    def set_reward(self, reward_position: tuple):
        self.grid[reward_position] = GridElements.REWARD