import numpy as np

from envs.environment import Environment


class GamblerProblem(Environment):

    def __init__(self, capital, heads_probability):
        self.capital = capital
        self.ph = heads_probability
        self.state = int(capital/2)

        states = list(np.arange(1, capital+1, 1))
        actions = list(np.arange(0, min(self.state, capital - self.state), 1))
        super().__init__(actions, states)

    def action(self, action):
        next_state = self.get_next_state(self.state, action)
        self.state = next_state

    def get_next_state(self, state, action):
        stake = action

        coin_flip = np.random.choice([False, True], p=[1-self.ph, self.ph])

        if coin_flip:
            next_state = self.state + stake
        else:
            next_state = self.state - stake

        self.actions = list(np.arange(0, min(next_state, self.capital - next_state), 1))

        return next_state

