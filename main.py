import numpy as np
from mdp import MDP
from grid_env import GridEnv
import pprint


def main():

    gridworld = GridEnv(5, 5)

    def gridworld_reward(state: list, action, next_state):
        if next_state == state and (state != [0, 1] and state != [0, 3]):
            return -1
        elif state == [0, 1] and action in GridEnv.AgentActions:
            return 10
        elif state == [0, 3] and action in GridEnv.AgentActions:
            return 5
        else:
            return 0

    mdp_gridworld = MDP(gridworld, gridworld_reward, 0.9)

    print(gridworld.grid)
    print('state value for state {}: {}'.format(mdp_gridworld.environment.state, mdp_gridworld.compute_state_value(mdp_gridworld.environment.state)))
    print('*action*')
    print('reward: {}'.format(mdp_gridworld.action(GridEnv.AgentActions.RIGHT)))
    print(gridworld.grid)
    print('state value for state {}: {}'.format(mdp_gridworld.environment.state, mdp_gridworld.compute_state_value(mdp_gridworld.environment.state)))
    print('*action*')
    print('reward: {}'.format(mdp_gridworld.action(GridEnv.AgentActions.DOWN)))
    print(gridworld.grid)
    print('state value for state {}: {}'.format(mdp_gridworld.environment.state, mdp_gridworld.compute_state_value(mdp_gridworld.environment.state)))
    print('*action*')
    print('reward: {}'.format(mdp_gridworld.action(GridEnv.AgentActions.UP)))
    print(gridworld.grid)
    print('state value for state {}: {}'.format(mdp_gridworld.environment.state, mdp_gridworld.compute_state_value(mdp_gridworld.environment.state)))
    print('*action*')
    print('reward: {}'.format(mdp_gridworld.action(GridEnv.AgentActions.RIGHT)))
    print(gridworld.grid)
    print('state value for state {}: {}'.format(mdp_gridworld.environment.state, mdp_gridworld.compute_state_value(mdp_gridworld.environment.state)))
    print('*action*')
    print('reward: {}'.format(mdp_gridworld.action(GridEnv.AgentActions.DOWN)))
    print(gridworld.grid)
    print('state value for state {}: {}'.format(mdp_gridworld.environment.state, mdp_gridworld.compute_state_value(mdp_gridworld.environment.state)))

    print(mdp_gridworld.v_values)
    mdp_gridworld.compute_state_values()

    print(np.round(mdp_gridworld.v_values, 1))

    print(mdp_gridworld.q_values)
    mdp_gridworld.compute_action_values()

    pp = pprint.PrettyPrinter()

    pp.pprint(mdp_gridworld.q_values)

    print(mdp_gridworld.policy)

    mdp_gridworld.policy_evaluation()
    print(mdp_gridworld.evaluation_values)

    # print(mdp_gridworld.compute_state_values())
    # print(mdp_gridworld.state_values)

    # print(mdp_gridworld.state_value2([0,0]))

    # print(mdp_gridworld.policy)
    # print(mdp_gridworld.policy[GridEnv.AgentActions.UP])


if __name__ == '__main__':
    main()