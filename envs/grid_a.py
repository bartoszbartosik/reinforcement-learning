import numpy as np

from envs.grid import Grid


class GridA(Grid):


    def __init__(self, width, height):
        super().__init__(width, height)


    def action(self, action):
        super().action(action)


    def get_next_state(self, state, action):
        # Perform transition to a specific next state if in given state
        if state == (0, 1):
            return 4, 1
        elif state == (0, 3):
            return 2, 3
        # Else perform normal transition
        else:
            return super().get_next_state(state, action)


    def set_agent(self, agent_position: tuple):
        super().set_agent(agent_position)


    def set_obstacle(self, obstacle_position: tuple):
       super().set_obstacle(obstacle_position)


    def set_terminal(self, terminal_position: tuple):
        super().set_terminal(terminal_position)

