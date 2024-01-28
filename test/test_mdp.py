import unittest

from envs.grid_a import GridA
from mdp.markov_decision_process import MDP

class TestMDP(unittest.TestCase):


    def setUp(self):

        grid_a = GridA(5, 5)

        def gridworld_reward(state: tuple, action, next_state):
            if next_state == state and (state != (0, 1) and state != (0, 3)):
                return -1
            elif state == (0, 1):
                return 10
            elif state == (0, 3):
                return 5
            else:
                return 0

        self.mdp_grid_a = MDP(grid_a, gridworld_reward, 0.9)

    def test_rewards(self):
        reward = self.mdp_grid_a.action(self.mdp_grid_a.env.actions[3])
        self.assertEqual(reward, 0)
        reward = self.mdp_grid_a.action(self.mdp_grid_a.env.actions[3])
        self.assertEqual(reward, 10)
        reward = self.mdp_grid_a.action(self.mdp_grid_a.env.actions[1])
        self.assertEqual(reward, -1)



if __name__ == '__main__':
    unittest.main()