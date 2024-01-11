import numpy as np


class DynamicProgramming:

    def __init__(self, mdp):
        self.env = mdp.environment
        self.actions = mdp.actions
        self.reward_function = mdp.reward_function
        self.gamma = mdp.gamma
        self.pi = mdp.policy


    def compute_state_value(self, state, state_values):
        if self.env.grid[tuple(state)] == self.env.GridElements.OBSTACLE.value:
            return 0

        if self.env.grid[tuple(state)] == self.env.GridElements.TERMINAL.value:
            return 0

        max_value = float('-inf')
        for action in self.actions:
            next_state = self.env.get_next_state(state.copy(), action)
            reward = self.reward_function(state, action, next_state)
            state_value = reward + self.gamma*state_values[tuple(next_state)]
            if state_value > max_value:
                max_value = state_value

        return max_value


    def get_optimal_state_value(self):
        # Initialize state-values matrix
        v = np.zeros_like(self.env.grid)

        # Solution convergence threshold
        epsilon = 0.0001

        delta = float('inf')
        while delta > epsilon:
            delta = 0
            for i in range(self.env.width):
                for j in range(self.env.height):
                    v_old = v[i, j]
                    v_new = self.compute_state_value([i, j], v)
                    v[i, j] = v_new
                    delta = max(delta, abs(v_new - v_old))

        return v


    def compute_action_value(self, state, action, q_table):
        if self.env.grid[tuple(state)] == self.env.GridElements.OBSTACLE.value:
            return 0

        if self.env.grid[tuple(state)] == self.env.GridElements.TERMINAL.value:
            return 0

        next_state = self.env.get_next_state(state.copy(), action)
        reward = self.reward_function(state, action, next_state)
        q = reward + self.gamma*max(q_table[tuple(next_state)])

        return q


    def get_optimal_action_value(self):
        # Initialize action-values tensor
        q = np.zeros((self.env.width, self.env.height, len(self.actions)))

        # Solution convergence threshold
        epsilon = 0.0001

        delta = float('inf')
        while delta > epsilon:
            delta = 0
            for i in range(self.env.width):
                for j in range(self.env.height):
                    for action in self.env.AgentActions:
                        q_old = q[i, j][action.value]
                        q_new = self.compute_action_value([i, j], action, q)
                        q[i, j][action.value] = q_new
                        delta = max(delta, abs(q_new - q_old))

        return q


    def evaluate_state(self, state, state_values):
        if self.env.grid[tuple(state)] == self.env.GridElements.OBSTACLE.value:
            return 0

        if self.env.grid[tuple(state)] == self.env.GridElements.TERMINAL.value:
            return 0

        v = 0
        for pi_a, action in zip(self.pi[tuple(state)], self.env.AgentActions):
            next_state = self.env.get_next_state(state.copy(), action)
            reward = self.reward_function(state, action, next_state)
            v += pi_a * (reward + self.gamma * state_values[tuple(next_state)])

        return v


    def policy_evaluation(self):
        # Initialize state-values matrix
        v = np.zeros_like(self.env.grid)

        # Solution convergence threshold
        epsilon = 1e-10

        delta = float('inf')
        while delta > epsilon:
            delta = 0
            for i in range(self.env.width):
                for j in range(self.env.height):
                    v_old = v[i, j]
                    v_new = self.evaluate_state([i, j], v)
                    v[i, j] = v_new
                    delta = max(delta, abs(v_new - v_old))

        return v


    def improve_policy(self, state, state_values):
        max_value = float('-inf')
        max_action = None
        for action in self.actions:
            next_state = self.env.get_next_state(state.copy(), action)
            reward = self.reward_function(state, action, next_state)
            state_value = reward + self.gamma*state_values[tuple(next_state)]
            if state_value > max_value:
                max_value = state_value
                max_action = action

        return max_action


    def policy_improvement(self, v):
        policy_stable = True
        for i in range(self.env.width):
            for j in range(self.env.height):
                action_old = np.argmax(self.pi[(i, j)])
                action_new = self.improve_policy([i, j], v).value

                self.pi[(i, j)] = np.zeros_like(self.pi[(i, j)])
                self.pi[(i, j)][action_new] = 1

                if action_old != action_new:
                    policy_stable = False

        return policy_stable


    def policy_iteration(self):
        # Initialize state-values matrix
        while True:
            v = self.policy_evaluation()
            policy_stable = self.policy_improvement(v)
            if policy_stable:
                break

        return self.policy_evaluation()
