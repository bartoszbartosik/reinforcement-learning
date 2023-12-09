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

        self.grid = GridElements.FREE.value*np.ones(shape=(height, width))

        self.agent_pos = [0, 0]


    def action(self, action: AgentActions):
        if self.validate_action(action):
            old_pos = self.agent_pos.copy()
            match action:
                case AgentActions.UP:
                    self.agent_pos[0] -= 1
                case AgentActions.DOWN:
                    self.agent_pos[0] += 1
                case AgentActions.LEFT:
                    self.agent_pos[1] -= 1
                case AgentActions.RIGHT:
                    self.agent_pos[1] += 1
            self.grid[tuple(self.agent_pos)], self.grid[tuple(old_pos)] = self.grid[tuple(old_pos)], self.grid[tuple(self.agent_pos)]
            return True
        return False


    def validate_action(self, action):

        # Check for walls
        if self.agent_pos[0] == 0 and action == AgentActions.UP:
            return False
        elif self.agent_pos[0] == len(self.grid)-1 and action == AgentActions.DOWN:
            return False
        elif self.agent_pos[1] == 0 and action == AgentActions.LEFT:
            return False
        elif self.agent_pos[1] == len(self.grid[0])-1 and action == AgentActions.RIGHT:
            return False
        # Check for obstacles
        elif action == AgentActions.UP and self.grid[self.agent_pos[0]-1, self.agent_pos[1]] == GridElements.OBSTACLE.value:
            return False
        elif action == AgentActions.DOWN and self.grid[self.agent_pos[0]+1, self.agent_pos[1]] == GridElements.OBSTACLE.value:
            return False
        elif action == AgentActions.LEFT and self.grid[self.agent_pos[0], self.agent_pos[1]-1] == GridElements.OBSTACLE.value:
            return False
        elif action == AgentActions.RIGHT and self.grid[self.agent_pos[0], self.agent_pos[1]+1] == GridElements.OBSTACLE.value:
            return False
        return True


    def set_agent(self, agent_position: list):
        self.agent_pos = agent_position
        self.grid[tuple(agent_position)] = GridElements.AGENT.value


    def set_obstacle(self, obstacle_position: tuple):
        self.grid[obstacle_position] = GridElements.OBSTACLE.value


    def set_reward(self, reward_position: tuple):
        self.grid[reward_position] = GridElements.REWARD.value