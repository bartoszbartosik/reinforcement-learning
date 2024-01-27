from envs.environment import Environment


def map_data(environment: Environment):
    return [i for i in range(len(environment))]

def map_terminals(environment: Environment):
    return [environment.states.index(i) for i in environment.terminal_states]