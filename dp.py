import numpy as np

import mdp as MDP

class DynamicProgramming:

    def __init__(self, mdp: MDP):
        self.mdp = mdp
        self.env = mdp.environment
        self.actions = mdp.actions
        self.reward_function = mdp.reward_function
        self.gamma = mdp.gamma
        self.pi = mdp.policy


    def get_optimal_state_value(self) -> np.ndarray:
        # Initialize state-values matrix
        v = np.zeros_like(self.mdp.states, dtype=float)

        # Solution convergence threshold
        epsilon = 1e-10

        delta = float('inf')
        while delta > epsilon:
            delta = 0
            for state in self.mdp.states:
                v_old = v[state]
                v_new = self.__compute_state_value(state, v)
                v[state] = v_new
                delta = max(delta, abs(v_new - v_old))

        return v


    def get_optimal_action_value(self) -> np.ndarray:
        # Initialize action-values tensor
        q = np.zeros((len(self.mdp.states), len(self.actions)))

        # Solution convergence threshold
        epsilon = 1e-10

        delta = float('inf')
        while delta > epsilon:
            delta = 0
            for state in self.mdp.states:
                for action in self.mdp.actions:
                    q_old = q[state][action]
                    q_new = self.__compute_action_value(state, action, q)
                    q[state][action] = q_new
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
        v = np.zeros_like(self.mdp.states, dtype=float)

        # Solution convergence threshold
        epsilon = 1e-10

        delta = float('inf')
        while delta > epsilon:
            delta = 0
            for index, state in enumerate(self.mdp.states):
                v_old = v[state]
                v_new = self.__compute_state_value(state, v)
                max_action = self.__improve_policy(state, v)
                v[state] = v_new
                self.pi[index] = np.zeros_like(self.pi[index])
                self.pi[index][max_action] = 1
                delta = max(delta, abs(v_new - v_old))

        return v, self.pi


    def policy_evaluation(self) -> np.ndarray:
        # Initialize state-values matrix
        v = np.zeros_like(self.mdp.states, dtype=float)

        # Solution convergence threshold
        epsilon = 1e-10

        delta = float('inf')
        while delta > epsilon:
            delta = 0
            for state in self.mdp.states:
                v_old = v[state]
                v_new = self.__evaluate_state(state, v)
                v[state] = v_new
                delta = max(delta, abs(v_new - v_old))

        return v


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # #   PRIVATE   FUNCTIONS   # # # # # # # # # # # # # # # # # # # # #

    def __compute_state_value(self, state, state_values) -> (float):
        if state in self.mdp.terminal_states:
            return 0

        if state in self.mdp.obstacle_states:
            return 0

        max_value = float('-inf')
        for action in self.actions:
            next_state = self.mdp.get_next_state(state, action)
            reward = self.reward_function(self.env.states[state], self.env.actions[action], self.env.states[next_state])
            state_value = reward + self.gamma*state_values[next_state]
            if state_value > max_value:
                max_value = state_value

        return max_value


    def __compute_action_value(self, state, action, q_table) -> float:
        if state in self.mdp.terminal_states:
            return 0

        if state in self.mdp.obstacle_states:
            return 0

        next_state = self.mdp.get_next_state(state, action)
        reward = self.reward_function(self.env.states[state], self.env.actions[action], self.env.states[next_state])
        q = reward + self.gamma*max(q_table[next_state])

        return q


    def __evaluate_state(self, state, state_values) -> float:
        if state in self.mdp.terminal_states:
            return 0

        if state in self.mdp.obstacle_states:
            return 0

        v = 0
        for action in self.mdp.actions:
            pi_a = self.pi[state][action]
            next_state = self.mdp.get_next_state(state, action)
            reward = self.reward_function(self.env.states[state], self.env.actions[action], self.env.states[next_state])
            v += pi_a * (reward + self.gamma * state_values[next_state])

        return v


    def __improve_policy(self, state, state_values) -> str:
        max_value = float('-inf')
        max_action = None
        for action in self.actions:
            next_state = self.mdp.get_next_state(state, action)
            reward = self.reward_function(self.env.states[state], self.env.actions[action], self.env.states[next_state])
            state_value = reward + self.gamma*state_values[next_state]
            if state_value > max_value:
                max_value = state_value
                max_action = action

        return max_action


    def __policy_improvement(self, v: np.ndarray) -> bool:
        policy_stable = True
        for state in self.mdp.states:
            action_old = np.argmax(self.pi[state])
            action_new = self.__improve_policy(state, v)

            self.pi[state] = np.zeros_like(self.pi[state])
            self.pi[state][action_new] = 1

            if action_old != action_new:
                policy_stable = False

        return policy_stable
