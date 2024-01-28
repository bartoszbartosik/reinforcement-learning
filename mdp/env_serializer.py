from envs.environment import Environment


class EnvSerializer:

    def __init__(self, environment: Environment):
        self.env = environment

    def serialize_states(self):
        return [i for i in range(len(self.env.states))]


    def serialize_actions(self):
        return [i for i in range(len(self.env.actions))]


    def serialize_terminals(self):
        return [self.env.states.index(i) for i in self.env.terminal_states]


    def serialize_obstacles(self):
        return [self.env.states.index(i) for i in self.env.obstacle_states]


    def serialize_state(self, state):
        return self.env.states.index(state)


    def deserialize_state(self, state: int):
        return self.env.states[state]


    def deserialize_action(self, action: int):
        return self.env.actions[action]