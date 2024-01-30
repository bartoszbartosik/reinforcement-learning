from enum import Enum

import numpy as np

from envs.environment import Environment


class Grid(Environment):

    # Encode grid elements
    class GridElements(Enum):
        FREE = 0
        AGENT = 1
        OBSTACLE = 2
        TERMINAL = 4


    def __init__(self, width, height, terminals=None, obstacles=None):
        super().__init__(actions=('UP', 'DOWN', 'LEFT', 'RIGHT'),
                         states=[(i, j) for i in range(height) for j in range(width)],
                         terminal_states=terminals,
                         obstacle_states=obstacles)

        # Set dimensions of the grid
        self.width = width      # columns
        self.height = height    # rows

        # Create a grid
        self.grid = Grid.GridElements.FREE.value * np.ones(shape=(height, width))

        # Initialize the agent position
        self.state = (0, 0)
        self.grid[self.state] = Grid.GridElements.AGENT.value

        # Apply terminal positions
        if terminals is not None:
            for terminal in terminals:
                self.grid[terminal] = Grid.GridElements.TERMINAL.value

        # Apply obstacle positions
        if obstacles is not None:
            for obstacle in obstacles:
                self.grid[obstacle] = Grid.GridElements.OBSTACLE.value


    def set_state(self, state):
        self.state = state


    def action(self, action: str):
        # Get the next state
        next_state = self.get_next_state(self.state, action)

        # If the next state is valid, perform transition and return True
        if next_state != self.state:
            self.grid[next_state], self.grid[self.state] = self.grid[self.state], self.grid[next_state]
            self.state = next_state
            return True

        # If the next state is not valid, return False
        return False


    def get_next_state(self, state: tuple, action):
        state = list(state)
        # Check for obstacles
        if (state[0] == 0 and action == 'UP' or
              state[0] == len(self.grid) - 1 and action == 'DOWN' or
              state[1] == 0 and action == 'LEFT' or
              state[1] == len(self.grid[0]) - 1 and action == 'RIGHT' or
              action == 'UP' and self.grid[
                  state[0] - 1, state[1]] == Grid.GridElements.OBSTACLE.value or
              action == 'DOWN' and self.grid[
                  state[0] + 1, state[1]] == Grid.GridElements.OBSTACLE.value or
              action == 'LEFT' and self.grid[
                  state[0], state[1] - 1] == Grid.GridElements.OBSTACLE.value or
              action == 'RIGHT' and self.grid[
                  state[0], state[1] + 1] == Grid.GridElements.OBSTACLE.value):
            pass

        # Update the state
        else:
            match action:
                case 'UP':
                    state[0] -= 1
                case 'DOWN':
                    state[0] += 1
                case 'LEFT':
                    state[1] -= 1
                case 'RIGHT':
                    state[1] += 1

        return tuple(state.copy())


    def set_agent(self, agent_position: tuple):
        self.grid[self.state], self.grid[agent_position] = self.grid[agent_position], self.grid[self.state]
        self.state = agent_position


    def set_obstacle(self, obstacle_position: tuple):
        self.grid[obstacle_position] = Grid.GridElements.OBSTACLE.value


    def set_terminal(self, terminal_position: tuple):
        self.grid[terminal_position] = Grid.GridElements.TERMINAL.value