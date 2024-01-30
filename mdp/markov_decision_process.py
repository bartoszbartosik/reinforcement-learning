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
        available_states = [state for state in self.states if state not in self.terminal_states + self.obstacle_states]
        state_id = np.random.choice(len(available_states))
        state = self.states[state_id]
        self.env.state = state

        # Choose an action for T_0 state following given policy
        action = np.random.choice(self.actions, p=policy[self.states.index(state)])

        # Perform an action and receive reward
        reward = self.action(action)

        # Append record to the episode (s_0, a_0, r_1)
        episode.append((state, action, reward))
        if self.env.state not in self.terminal_states:
            for T in range(1, steps):
                state = self.env.state
                action = np.random.choice(self.actions, p=policy[self.states.index(state)])
                reward = self.action(action)
                episode.append((state, action, reward))
                if self.env.state in self.terminal_states:
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

