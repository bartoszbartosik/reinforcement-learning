import numpy as np

from envs.environment import Environment
from mdp import serializer


class MDP:

    def __init__(self, environment: Environment, reward_function, discount_factor):
        # Initialize environment
        self.environment = environment

        # Serialize MDP's sets
        self.actions = serializer.serialize_actions(environment)
        self.states = serializer.serialize_states(environment)
        self.terminal_states = serializer.serialize_terminals(environment)
        self.obstacle_states = serializer.serialize_obstacles(environment)

        # Initialize reward function
        self.rw = reward_function

        # Initialize discount factor
        self.gamma = discount_factor

        # Initialize as equiprobable random policy
        self.policy = 1/len(self.actions)*np.ones((len(self.environment.states), len(self.actions)))


    def follow_policy(self):
        pass


    def reward_function(self, state, action, next_state):
        return self.rw(self.environment.states[state], self.environment.actions[action], self.environment.states[next_state])


    def action(self, action):
        next_state = self.environment.get_next_state(self.environment.state, action)
        reward = self.rw(self.environment.state, action, next_state)
        self.environment.action(action)
        return reward


    def get_next_state(self, state: int, action: int):
        return self.environment.states.index(self.environment.get_next_state(self.environment.states[state], self.environment.actions[action]))


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

