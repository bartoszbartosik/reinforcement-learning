import numpy as np

from envs.environment import Environment
import mapper


class MDP:

    def __init__(self, environment: Environment, reward_function, discount_factor):
        self.environment = environment
        self.actions = mapper.map_data(environment.actions)
        self.states = mapper.map_data(environment.states)
        self.terminal_states = mapper.map_terminals(environment.terminal_states)
        self.obstacle_states = [environment.states.index(i) for i in environment.obstacle_states]

        self.reward_function = reward_function
        self.gamma = discount_factor

        # Initialize as equiprobable random policy
        self.policy = 1/len(self.actions)*np.ones((len(self.environment.states), len(self.actions)))


    def follow_policy(self):
        pass


    def action(self, action):
        next_state = self.environment.get_next_state(self.environment.state, action)
        reward = self.reward_function(self.environment.state, action, next_state)
        self.environment.action(action)
        return reward


    def get_next_state(self, state: int, action: int):
        return self.environment.states.index(self.environment.get_next_state(self.environment.states[state], self.environment.actions[action]))


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #
