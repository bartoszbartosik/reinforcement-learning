import numpy as np

from envs.environment import Environment


class GamblerProblem(Environment):

    def __init__(self, goal_capital, heads_probability):
        self.ph = heads_probability
        state = int(goal_capital / 2)
        super().__init__(actions=list(np.arange(0, min(state, 100 - state) + 1, 1)),
                         states=list(np.arange(0, goal_capital+1, 1)),
                         terminal_states=[0, 100])
        self.state = state


    def action(self, action):
        next_state = self.get_next_state(super().state, action)
        self.actions = list(np.arange(0, min(next_state, 100 - next_state) + 1, 1))
        self.state = next_state


    def set_state(self, state):
        self.state = state
        self.actions = list(np.arange(0, min(state, 100 - state) + 1, 1))


    def get_next_state(self, state, action):
        stake = action

        coin_flip = np.random.choice([False, True], p=[1-self.ph, self.ph])

        if coin_flip:
            next_state = state + stake
        else:
            next_state = state - stake

        return next_state

