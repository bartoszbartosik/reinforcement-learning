import numpy as np

from envs.environment import Environment


class MDP:

    def __init__(self, environment: Environment, reward_function, discount_factor):
        # Initialize environment
        self.env = environment

        # Initialize reward function
        self.rw = reward_function

        # Initialize discount factor
        self.gamma = discount_factor


    def generate_episode(self, steps, policy, initial_state=None) -> list:
        # If initial state not given, choose random
        if initial_state is None:
            initial_state = self.env.available_states[np.random.choice(len(self.env.available_states))]
        else:
            initial_state = initial_state

        # Initialize episode tuples list
        episode: [tuple] = []

        for T in range(steps):
            if T == 0:
                # Randomly choose T_0 state which is not a terminal nor obstacle state
                self.env.state = initial_state

            # Update current state
            state = self.env.state

            # Choose an action for T state following given policy
            action = np.random.choice(self.env.actions, p=policy[self.env.states.index(state)])

            # Perform an action and receive reward
            reward = self.action(action)

            # Append record to the episode (s_0, a_0, r_1)
            episode.append((state, action, reward))

        return episode


    def action(self, action):
        next_state = self.env.get_next_state(self.env.state, action)
        reward = self.rw(self.env.state, action, next_state)
        self.env.action(action)
        return reward


    def get_next_state(self, state, action):
        return self.env.get_next_state(state, action)


    def equiprobable_policy(self):
        return 1 / len(self.env.actions) * np.ones((len(self.env.states), len(self.env.actions)))


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

