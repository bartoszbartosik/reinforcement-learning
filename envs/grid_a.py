from enum import Enum

from envs.environment import Environment

import numpy as np


class GridA(Environment):

    # Encode grid elements
    class GridElements(Enum):
        FREE = 0
        AGENT = 1
        OBSTACLE = 2
        TERMINAL = 4

    # Encode agent's actions
    class AgentActions(Environment.AgentActions):
        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3


    def __init__(self, width, height):
        # Set dimensions of the grid
        self.width = width      # columns
        self.height = height    # rows

        # Create a grid
        self.grid = GridA.GridElements.FREE.value * np.ones(shape=(height, width))

        # Initialize the agent position
        self.state = [0, 0]
        self.grid[tuple(self.state)] = GridA.GridElements.AGENT.value


    def action(self, action: AgentActions):
        # Verify action and get state transition
        next_state = self.get_next_state(self.state.copy(), action)

        # If state is valid, perform its transition
        if next_state != self.state:
            self.grid[tuple(next_state)], self.grid[tuple(self.state)] = self.grid[tuple(self.state)], self.grid[tuple(next_state)]
            self.state = next_state
            return True

        return False


    def get_next_state(self, state, action):
        # Check for walls
        if state == [0, 1]:
            state = [4, 1]
        elif state == [0, 3]:
            state = [2, 3]
        elif (state[0] == 0 and action == GridA.AgentActions.UP or
              state[0] == len(self.grid) - 1 and action == GridA.AgentActions.DOWN or
              state[1] == 0 and action == GridA.AgentActions.LEFT or
              state[1] == len(self.grid[0]) - 1 and action == GridA.AgentActions.RIGHT or
              action == GridA.AgentActions.UP and self.grid[state[0] - 1, state[1]] == GridA.GridElements.OBSTACLE.value or
              action == GridA.AgentActions.DOWN and self.grid[state[0] + 1, state[1]] == GridA.GridElements.OBSTACLE.value or
              action == GridA.AgentActions.LEFT and self.grid[state[0], state[1] - 1] == GridA.GridElements.OBSTACLE.value or
              action == GridA.AgentActions.RIGHT and self.grid[state[0], state[1] + 1] == GridA.GridElements.OBSTACLE.value):
            pass
        # Update the state
        else:
            match action:
                case GridA.AgentActions.UP:
                    state[0] -= 1
                case GridA.AgentActions.DOWN:
                    state[0] += 1
                case GridA.AgentActions.LEFT:
                    state[1] -= 1
                case GridA.AgentActions.RIGHT:
                    state[1] += 1
        return state


    def set_agent(self, agent_position: list):
        self.grid[tuple(self.state)], self.grid[tuple(agent_position)] = self.grid[tuple(agent_position)], self.grid[tuple(self.state)]
        self.state = agent_position


    def set_obstacle(self, obstacle_position: list):
        self.grid[tuple(obstacle_position)] = GridA.GridElements.OBSTACLE.value


    def set_terminal(self, terminal_position: list):
        self.grid[tuple(terminal_position)] = GridA.GridElements.TERMINAL.value
