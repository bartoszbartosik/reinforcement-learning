from envs.grid import Grid


class GridWindy(Grid):

    def __init__(self):
        super().__init__(10, 7, [(3, 7)])
        self.wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        self.initial_state = (3, 0)


    def action(self, action: str):
        return super().action(action)


    def get_next_state(self, state: tuple, action):
        next_state = super().get_next_state(state, action)
        for _ in range(self.wind_strength[state[1]]):
            next_state = super().get_next_state(next_state, 'UP')
        return next_state


    def get_next_transitions(self, state, action):
        return super().get_next_transitions(state, action)


    def set_agent(self, agent_position: tuple):
        super().set_agent(agent_position)


    def set_obstacle(self, obstacle_position: tuple):
        super().set_obstacle(obstacle_position)


    def set_terminal(self, terminal_position: tuple):
        super().set_terminal(terminal_position)

