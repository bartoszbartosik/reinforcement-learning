import numpy as np

import mdp as MDP

class DynamicProgramming:

    def __init__(self, mdp: MDP):
        self.env = mdp.environment
        self.actions = mdp.actions
        self.reward_function = mdp.reward_function
        self.gamma = mdp.gamma
        self.pi = mdp.policy


    def get_optimal_state_value(self) -> np.ndarray:
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
                    v_new = self.__compute_state_value((i, j), v)
                    v[i, j] = v_new
                    delta = max(delta, abs(v_new - v_old))

        return v


    def get_optimal_action_value(self) -> np.ndarray:
        # Initialize action-values tensor
        q = np.zeros((self.env.width, self.env.height, len(self.actions)))

        # Solution convergence threshold
        epsilon = 1e-10

        delta = float('inf')
        while delta > epsilon:
            delta = 0
            for i in range(self.env.width):
                for j in range(self.env.height):
                    for action in self.env.AgentActions:
                        q_old = q[i, j][action.value]
                        q_new = self.__compute_action_value((i, j), action, q)
                        q[i, j][action.value] = q_new
                        delta = max(delta, abs(q_new - q_old))

        return q


    def policy_iteration(self) -> (np.ndarray, np.ndarray):
        while True:
            v = self.policy_evaluation()
            policy_stable = self.__policy_improvement(v)
            if policy_stable:
                break

        return v, self.pi


    def value_iteration(self) -> (np.ndarray, np.ndarray):
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
                    v_new = self.__compute_state_value((i, j), v)
                    max_action = self.__improve_policy((i, j), v)
                    v[i, j] = v_new
                    self.pi[(i, j)] = np.zeros_like(self.pi[(i, j)])
                    self.pi[i, j][max_action.value] = 1
                    delta = max(delta, abs(v_new - v_old))

        return v, self.pi


    def policy_evaluation(self) -> np.ndarray:
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
                    v_new = self.__evaluate_state((i, j), v)
                    v[i, j] = v_new
                    delta = max(delta, abs(v_new - v_old))

        return v


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

    def __compute_state_value(self, state, state_values) -> (MDP.Grid.AgentActions, float):
        if self.env.grid[state] == self.env.GridElements.OBSTACLE.value:
            return 0

        if self.env.grid[state] == self.env.GridElements.TERMINAL.value:
            return 0

        max_value = float('-inf')
        for action in self.actions:
            next_state = self.env.get_next_state(state, action)
            reward = self.reward_function(state, action, next_state)
            state_value = reward + self.gamma*state_values[next_state]
            if state_value > max_value:
                max_value = state_value

        return max_value


    def __compute_action_value(self, state, action, q_table) -> float:
        if self.env.grid[state] == self.env.GridElements.OBSTACLE.value:
            return 0

        if self.env.grid[state] == self.env.GridElements.TERMINAL.value:
            return 0

        next_state = self.env.get_next_state(state, action)
        reward = self.reward_function(state, action, next_state)
        q = reward + self.gamma*max(q_table[next_state])

        return q


    def __evaluate_state(self, state, state_values) -> float:
        if self.env.grid[state] == self.env.GridElements.OBSTACLE.value:
            return 0

        if self.env.grid[state] == self.env.GridElements.TERMINAL.value:
            return 0

        v = 0
        for pi_a, action in zip(self.pi[state], self.env.AgentActions):
            next_state = self.env.get_next_state(state, action)
            reward = self.reward_function(state, action, next_state)
            v += pi_a * (reward + self.gamma * state_values[next_state])

        return v


    def __improve_policy(self, state, state_values) -> MDP.Grid.AgentActions:
        max_value = float('-inf')
        max_action = None
        for action in self.actions:
            next_state = self.env.get_next_state(state, action)
            reward = self.reward_function(state, action, next_state)
            state_value = reward + self.gamma*state_values[next_state]
            if state_value > max_value:
                max_value = state_value
                max_action = action

        return max_action


    def __policy_improvement(self, v: np.ndarray) -> bool:
        policy_stable = True
        for i in range(self.env.width):
            for j in range(self.env.height):
                action_old = np.argmax(self.pi[(i, j)])
                action_new = self.__improve_policy([i, j], v).value

                self.pi[(i, j)] = np.zeros_like(self.pi[(i, j)])
                self.pi[(i, j)][action_new] = 1

                if action_old != action_new:
                    policy_stable = False

        return policy_stable
