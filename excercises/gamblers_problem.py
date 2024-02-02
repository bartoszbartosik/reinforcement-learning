import numpy as np
from matplotlib import pyplot as plt

from envs.gambler import Gambler
from mdp.markov_decision_process import MDP
import methods.dynamic_programming as dp
import methods.monte_carlo as mc


def main():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # #   G A M B L E R ' S   P R O B L E M   # # # # # # # # # # # # # # # # # #
    gambler = Gambler(100, 0.4)


    def gambler_reward(state, action, next_state):
        if next_state == 100:
            return 1
        else:
            return 0

    mdp_gambler = MDP(gambler, gambler_reward, 1)
    v, pi = dp.value_iteration(mdp_gambler)
    pi = [gambler.states[np.argmax(p)] for p in pi]

    plt.figure(1)
    plt.subplot(211)
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.plot(gambler.states, v)
    plt.subplot(212)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
    plt.bar(gambler.states, pi)
    plt.show()




if __name__ == '__main__':
    main()