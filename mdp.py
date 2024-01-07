import numpy as np

from envs.grid import Grid


class MDP:

    def __init__(self, environment: Grid, reward_function, discount_factor):
        self.environment = environment
        self.actions = list(Grid.AgentActions)
        self.reward_function = reward_function
        self.gamma = discount_factor

        self.policy = dict()
        for i in range(self.environment.width):
            for j in range(self.environment.height):
                self.policy[(i, j)] = 1/len(self.environment.AgentActions) * np.ones_like(self.environment.AgentActions)

        self.v_values = np.zeros_like(self.environment.grid)
        self.q_values = dict()
        for i in range(self.environment.width):
            for j in range(self.environment.height):
                self.q_values[(i, j)] = np.zeros(shape=(len(self.environment.AgentActions)))

        self.evaluation_values = np.zeros_like(self.environment.grid)


    def action(self, action):
        next_state = self.environment.get_next_state(self.environment.state.copy(), action)
        reward = self.reward_function(self.environment.state, action, next_state)
        self.environment.action(action)
        return reward


    def compute_state_value(self, state):
        if self.environment.grid[tuple(state)] == self.environment.GridElements.OBSTACLE.value:
            return 0

        if self.environment.grid[tuple(state)] == self.environment.GridElements.TERMINAL.value:
            return 0

        max_value = float('-inf')
        for action in self.actions:
            next_state = self.environment.get_next_state(state.copy(), action)
            reward = self.reward_function(state, action, next_state)
            state_value = reward + self.gamma*self.v_values[tuple(next_state)]
            if state_value > max_value:
                max_value = state_value

        return max_value


    def compute_state_values(self):
        epsilon = 0.0001

        delta = float('inf')
        while delta > epsilon:
            delta = 0
            for i in range(self.environment.width):
                for j in range(self.environment.height):
                    v_old = self.v_values[i, j]
                    v_new = self.compute_state_value([i, j])
                    self.v_values[i, j] = v_new
                    delta = max(delta, abs(v_new - v_old))


    def compute_action_value(self, state, action):
        if self.environment.grid[tuple(state)] == self.environment.GridElements.OBSTACLE.value:
            return 0

        if self.environment.grid[tuple(state)] == self.environment.GridElements.TERMINAL.value:
            return 0

        next_state = self.environment.get_next_state(state.copy(), action)
        reward = self.reward_function(state, action, next_state)
        q = reward + self.gamma*max(self.q_values[tuple(next_state)])

        return q


    def compute_action_values(self):
        epsilon = 0.0001

        delta = float('inf')
        while delta > epsilon:
            delta = 0
            for i in range(self.environment.width):
                for j in range(self.environment.height):
                    for action in self.environment.AgentActions:
                        q_old = self.q_values[i, j][action.value]
                        q_new = self.compute_action_value([i, j], action)
                        self.q_values[i, j][action.value] = q_new
                        delta = max(delta, abs(q_new - q_old))


    def evaluate_state(self, state):
        if self.environment.grid[tuple(state)] == self.environment.GridElements.OBSTACLE.value:
            return 0

        if self.environment.grid[tuple(state)] == self.environment.GridElements.TERMINAL.value:
            return 0

        v = 0
        for pi_a, action in zip(self.policy[tuple(state)], self.environment.AgentActions):
            next_state = self.environment.get_next_state(state.copy(), action)
            reward = self.reward_function(state, action, next_state)
            v += pi_a * (reward + self.gamma * self.evaluation_values[tuple(next_state)])

        return v


    def policy_evaluation(self):
        epsilon = 1e-10

        delta = float('inf')
        while delta > epsilon:
            delta = 0
            evaluation_values_copy = self.evaluation_values.copy()
            for i in range(self.environment.width):
                for j in range(self.environment.height):
                    v_old = self.evaluation_values[i, j]
                    v_new = self.evaluate_state([i, j])
                    evaluation_values_copy[i, j] = v_new
                    delta = max(delta, abs(v_new - v_old))
            self.evaluation_values = evaluation_values_copy.copy()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

