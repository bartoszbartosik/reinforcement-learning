import numpy as np

from envs.environment import Environment


class GamblerProblem(Environment):

    def __init__(self, capital, heads_probability):
        self.ph = heads_probability
        self.state = int(capital/2)

        states = np.arange(1, capital+1, 1)
        actions = np.arange(0, min(self.state, capital - self.state), 1)
        super().__init__(actions, states)

    def action(self, action):
        next_state = self.get_next_state(self.state, action)
        self.state = next_state

    def get_next_state(self, state, action):
        stake = action

        coin_flip = np.random.choice([False, True], p=[1-self.ph, self.ph])

        if coin_flip:
            return self.state + stake
        else:
            return self.state - stake

