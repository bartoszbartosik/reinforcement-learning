from enum import Enum

import numpy as np

from envs.environment import Environment


class Grid(Environment):

    class AgentActions(Environment.AgentActions):
        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3


    # Encode grid elements
    class GridElements(Enum):
        FREE = 0
        AGENT = 1
        OBSTACLE = 2
        TERMINAL = 4


    def __init__(self, width, height):
        # Set dimensions of the grid
        self.width = width      # columns
        self.height = height    # rows

        # Create a grid
        self.grid = Grid.GridElements.FREE.value * np.ones(shape=(height, width))

        # Initialize the agent position
        self.state = [0, 0]
        self.grid[tuple(self.state)] = Grid.GridElements.AGENT.value


    def action(self, action: AgentActions):
        # Get the next state
        next_state = self.get_next_state(self.state.copy(), action)

        # If the next state is valid, perform transition and return True
        if next_state != self.state:
            self.grid[tuple(next_state)], self.grid[tuple(self.state)] = self.grid[tuple(self.state)], self.grid[tuple(next_state)]
            self.state = next_state
            return True

        # If the next state is not valid, return False
        return False


    def get_next_state(self, state, action):
        # Check for obstacles
        if (state[0] == 0 and action == Grid.AgentActions.UP or
              state[0] == len(self.grid) - 1 and action == Grid.AgentActions.DOWN or
              state[1] == 0 and action == Grid.AgentActions.LEFT or
              state[1] == len(self.grid[0]) - 1 and action == Grid.AgentActions.RIGHT or
              action == Grid.AgentActions.UP and self.grid[
                  state[0] - 1, state[1]] == Grid.GridElements.OBSTACLE.value or
              action == Grid.AgentActions.DOWN and self.grid[
                  state[0] + 1, state[1]] == Grid.GridElements.OBSTACLE.value or
              action == Grid.AgentActions.LEFT and self.grid[
                  state[0], state[1] - 1] == Grid.GridElements.OBSTACLE.value or
              action == Grid.AgentActions.RIGHT and self.grid[
                  state[0], state[1] + 1] == Grid.GridElements.OBSTACLE.value):
            pass

        # Update the state
        else:
            match action:
                case Grid.AgentActions.UP:
                    state[0] -= 1
                case Grid.AgentActions.DOWN:
                    state[0] += 1
                case Grid.AgentActions.LEFT:
                    state[1] -= 1
                case Grid.AgentActions.RIGHT:
                    state[1] += 1

        return state


    def set_agent(self, agent_position: list):
        self.grid[tuple(self.state)], self.grid[tuple(agent_position)] = self.grid[tuple(agent_position)], self.grid[tuple(self.state)]
        self.state = agent_position


    def set_obstacle(self, obstacle_position: list):
        self.grid[tuple(obstacle_position)] = Grid.GridElements.OBSTACLE.value


    def set_terminal(self, terminal_position: list):
        self.grid[tuple(terminal_position)] = Grid.GridElements.TERMINAL.value