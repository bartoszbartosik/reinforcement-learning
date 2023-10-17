import numpy as np


class QLearning:

    def __init__(self, mdp, learning_rate):
        self.mdp = mdp
        self.learning_rate = learning_rate

        self.qtable = np.zeros(shape=(len(self.mdp.actions), len(self.mdp.environment.flatten())))


    def update_qtable(self, action):
        # pos_old = pos_new = self.agent_position

        # q_old = self.qtable[tuple(pos_old)]
        # q_new = self.qtable[tuple(pos_new)]
        # r_new = self.environment[(tuple(pos_new))]
        #
        # return self.qtable[self.agent_position]

        agent_position = self.mdp.get_position('agent')

        q_old = self.qtable[agent_position]
        r_new = self.mdp.environment[agent_position]
        self.qtable[agent_position] = ((1 - self.learning_rate) * self.qtable[agent_position] +
                                                   self.learning_rate*(self.qtable[agent_position]))
