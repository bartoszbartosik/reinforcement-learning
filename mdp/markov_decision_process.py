import numpy as np

from envs.environment import Environment


class MDP:

    def __init__(self, environment: Environment, reward_function, discount_factor):
        # Initialize environment
        self.env = environment

        # Serialize MDP's sets
        self.actions = self.env.actions
        self.states = self.env.states
        self.terminal_states = self.env.terminal_states
        self.obstacle_states = self.env.obstacle_states

        # Initialize reward function
        self.rw = reward_function

        # Initialize discount factor
        self.gamma = discount_factor

        # Initialize as equiprobable random policy
        self.policy = 1/len(self.actions)*np.ones((len(self.states), len(self.actions)))


    def generate_episode(self, steps, policy):
        episode = []

        # Randomly choose T_0 state which is not a terminal nor obstacle state
        state = np.random.choice([state for state in self.states if state not in self.terminal_states+self.obstacle_states])
        self.env.state = self.serializer.deserialize_state(state)
        action = np.random.choice(self.actions, p=policy[state])
        reward = self.action(self.serializer.deserialize_action(action))
        episode.append((state, action, reward))
        if self.serializer.serialize_state(self.env.state) not in self.terminal_states:
            for T in range(1, steps):
                state = self.serializer.serialize_state(self.env.state)
                action = np.random.choice(self.actions, p=policy[state])
                reward = self.action(self.serializer.deserialize_action(action))
                episode.append((state, action, reward))
                if self.serializer.serialize_state(self.env.state) in self.terminal_states:
                    break

        return episode


    def action(self, action):
        next_state = self.env.get_next_state(self.env.state, action)
        reward = self.rw(self.env.state, action, next_state)
        self.env.action(action)
        return reward


    def get_next_state(self, state, action):
        return self.env.get_next_state(state, action)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

