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

        # Initialize as equiprobable random policy
        self.policy = 1/len(self.env.actions)*np.ones((len(self.env.states), len(self.env.actions)))


    def generate_episode(self, steps, policy):
        # Initialize episode tuples list
        episode: [tuple] = []

        for T in range(steps):
            if T == 0:
                # Randomly choose T_0 state which is not a terminal nor obstacle state
                available_states = [state for state in self.env.states if state not in self.env.terminal_states + self.env.obstacle_states]
                state_id = np.random.choice(len(available_states))
                self.env.state = self.env.states[state_id]

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


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

