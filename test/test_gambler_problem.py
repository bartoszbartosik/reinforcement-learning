import unittest

import numpy as np
from matplotlib import pyplot as plt

from envs.gambler_problem import GamblerProblem
from mdp.markov_decision_process import MDP
import methods.dynamic_programming as dp


class TestDP(unittest.TestCase):

    def test_gambler_problem(self):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # # # # # # # # # # # # # # # # # #   G A M B L E R ' S   P R O B L E M   # # # # # # # # # # # # # # # # # #
        self.gambler = GamblerProblem(100, 0.4)


        def gambler_reward(state, action, next_state):
            if next_state == 100:
                return 1
            else:
                return 0

        self.mdp_gambler = MDP(self.gambler, gambler_reward, 1)
        v, pi = dp.value_iteration(self.mdp_gambler)
        pi = [self.gambler.states[np.argmax(p)] for p in pi]

        plt.figure(1)
        plt.subplot(211)
        plt.xlabel('Capital')
        plt.ylabel('Value estimates')
        plt.plot(self.gambler.states, v)
        plt.subplot(212)
        plt.xlabel('Capital')
        plt.ylabel('Final policy (stake)')
        plt.bar(self.gambler.states, pi)
        plt.show()


if __name__ == '__main__':
    unittest.main()