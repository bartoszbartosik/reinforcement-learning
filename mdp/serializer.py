from envs.environment import Environment


def serialize_states(environment: Environment):
    return [i for i in range(len(environment.states))]


def serialize_actions(environment: Environment):
    return [i for i in range(len(environment.actions))]


def serialize_terminals(environment: Environment):
    return [environment.states.index(i) for i in environment.terminal_states]


def serialize_obstacles(environment: Environment):
    return [environment.states.index(i) for i in environment.obstacle_states]