import numpy as np

from envs.environment import Environment


class GamblerProblem(Environment):

    def __init__(self, goal_capital, heads_probability):
        self.ph = heads_probability
        state = int(goal_capital / 1)
        super().__init__(actions=list(np.arange(0, goal_capital + 1, 1)),
                         # actions=list(np.arange(0, min(state, 100 - state) + 1, 1)),
                         states=list(np.arange(0, goal_capital+1, 1)),
                         terminal_states=[0])
        self.state = state

        for state_id, state in enumerate(self.states):
            for action_id, action in enumerate(self.actions):
                for next_state_id, next_state in enumerate(self.states):
                    if state + action == next_state:
                        self.p[next_state_id][state_id][action_id] = self.ph
                    elif state - action == next_state:
                        self.p[next_state_id][state_id][action_id] = 1-self.ph
                    else:
                        self.p[next_state_id][state_id][action_id] = 0

    def action(self, action):
        next_state = self.get_next_state(super().state, action)
        self.actions = list(np.arange(0, min(next_state, 100 - next_state) + 1, 1))
        self.state = next_state


    def get_next_state(self, state, action):
        action = min(state, 100 - state)

        stake = action

        coin_flip = np.random.choice([False, True], p=[1-self.ph, self.ph])

        if coin_flip:
            next_state = state + stake
        else:
            next_state = state - stake

        return next_state


    def get_next_transitions(self, state, action):
        tails_state = state - action
        heads_state = state + action
        p_tails = self.p[self.states.index(tails_state)][self.actions.index(action)][self.states.index(state)]
        p_heads = self.p[self.states.index(heads_state)][self.actions.index(action)][self.states.index(state)]

        return (p_tails, p_heads), (tails_state, heads_state)

