from enum import Enum

import numpy as np




class GridEnv:

    # Encode grid elements
    class GridElements(Enum):
        FREE = 0
        AGENT = 1
        OBSTACLE = 2

    # Encode agent's actions
    class AgentActions(Enum):
        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3


    def __init__(self, width, height):
        # Set dimensions of the grid
        self.width = width      # columns
        self.height = height    # rows

        # Create a grid
        self.grid = GridEnv.GridElements.FREE.value*np.ones(shape=(height, width))

        # Initialize the agent position
        self.state = [0, 0]
        self.grid[tuple(self.state)] = GridEnv.GridElements.AGENT.value


    def action(self, action: AgentActions):
        # Verify action and get state transition
        next_state = self.validate_action(action, self.state.copy())

        # If state is valid, perform its transition
        if next_state is not None:
            self.grid[tuple(next_state)], self.grid[tuple(self.state)] = self.grid[tuple(self.state)], self.grid[tuple(next_state)]
            self.state = next_state
            return True
        return False


    def validate_action(self, action, state):
        # Check for walls
        if (state[0] == 0 and action == GridEnv.AgentActions.UP or
            state[0] == len(self.grid)-1 and action == GridEnv.AgentActions.DOWN or
            state[1] == 0 and action == GridEnv.AgentActions.LEFT or
            state[1] == len(self.grid[0])-1 and action == GridEnv.AgentActions.RIGHT or
            action == GridEnv.AgentActions.UP and self.grid[state[0]-1, state[1]] == GridEnv.GridElements.OBSTACLE.value or
            action == GridEnv.AgentActions.DOWN and self.grid[state[0]+1, state[1]] == GridEnv.GridElements.OBSTACLE.value or
            action == GridEnv.AgentActions.LEFT and self.grid[state[0], state[1]-1] == GridEnv.GridElements.OBSTACLE.value or
            action == GridEnv.AgentActions.RIGHT and self.grid[state[0], state[1]+1] == GridEnv.GridElements.OBSTACLE.value):
            return None
        # Update the state
        else:
            match action:
                case GridEnv.AgentActions.UP:
                    state[0] -= 1
                case GridEnv.AgentActions.DOWN:
                    state[0] += 1
                case GridEnv.AgentActions.LEFT:
                    state[1] -= 1
                case GridEnv.AgentActions.RIGHT:
                    state[1] += 1
            return state


    def set_agent(self, agent_position: list):
        self.grid[tuple(self.state)], self.grid[tuple(agent_position)] = self.grid[tuple(agent_position)], self.grid[tuple(self.state)]
        self.state = agent_position


    def set_obstacle(self, obstacle_position: list):
        self.grid[tuple(obstacle_position)] = GridEnv.GridElements.OBSTACLE.value

