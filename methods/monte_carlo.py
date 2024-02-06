import numpy as np

from mdp.markov_decision_process import MDP


def first_visit_prediction(mdp: MDP, episodes, steps, policy=None):
    # If policy not given, assume equiprobable
    if policy is None:
        policy = 1/len(mdp.env.actions)*np.ones((len(mdp.env.states), len(mdp.env.actions)))

    # Initialize state-values matrix
    v = np.zeros(len(mdp.env.states), dtype=float)

    # Visits
    visits = np.zeros_like(v)

    for _ in range(episodes):
        # Generate MDP episode
        episode = mdp.generate_episode(steps=steps, policy=policy)
        states = [step[0] for step in reversed(episode)]
        # Initialize expected return
        G = 0

        # For each step (T-1, T-2, ..., 0) of an episode
        for T, (state, _, reward) in enumerate(reversed(episode)):
            # Increment expected return
            G = mdp.gamma * G + reward
            if state not in states[:T]:
                v[mdp.env.states.index(state)] += G
                visits[mdp.env.states.index(state)] += 1

    # Compute average expected return
    v /= visits

    return v


def every_visit_prediction(mdp: MDP, episodes, steps, policy=None):
    # If policy not given, assume equiprobable
    if policy is None:
        policy = 1/len(mdp.env.actions)*np.ones((len(mdp.env.states), len(mdp.env.actions)))

    # Initialize state-values matrix
    v = np.zeros(len(mdp.env.states), dtype=float)

    # Visits
    visits = np.zeros_like(v)

    for _ in range(episodes):
        # Generate MDP episode
        episode = mdp.generate_episode(steps=steps, policy=policy)

        # Initialize expected return
        G = 0

        # For each step (T-1, T-2, ..., 0) of an episode
        for state, _, reward in reversed(episode):
            # Increment expected return
            G = mdp.gamma * G + reward
            v[mdp.env.states.index(state)] += G
            visits[mdp.env.states.index(state)] += 1

    # Compute average expected return
    v /= visits

    return v


def exploring_starts(mdp, episodes, steps, policy=None):
    # If policy not given, assume equiprobable
    if policy is None:
        policy = 1/len(mdp.env.actions)*np.ones((len(mdp.env.states), len(mdp.env.actions)))

    # Initialize action-values matrix
    q = np.zeros((len(mdp.env.states), len(mdp.env.actions)))

    # Visits
    visits = np.zeros_like(q)

    for _ in range(episodes):
        # Generate MDP episode
        episode = mdp.generate_episode(steps=steps, policy=policy)
        state_actions = [(step[0], step[1]) for step in reversed(episode)]

        # Initialize expected return
        G = 0

        # For each step (T-1, T-2, ..., 0) of an episode
        for T, (state, action, reward) in enumerate(reversed(episode)):
            # Increment expected return
            G = mdp.gamma * G + reward
            if (state, action) not in state_actions[:T]:
                q[mdp.env.states.index(state)][mdp.env.actions.index(action)] += G
                visits[mdp.env.states.index(state)][mdp.env.actions.index(action)] += 1

    # Compute average expected return
    q /= visits

    return q


def on_policy_first_visit(mdp, episodes, steps, epsilon, policy=None):
    # If policy not given, assume equiprobable
    if policy is None:
        policy = 1/len(mdp.env.actions)*np.ones((len(mdp.env.states), len(mdp.env.actions)))

    # Initialize action-values matrix
    q = np.zeros((len(mdp.env.states), len(mdp.env.actions)))

    # Visits
    visits = np.zeros_like(q)

    for _ in range(episodes):
        # Generate MDP episode
        episode = mdp.generate_episode(steps=steps, policy=policy)
        state_actions = [(step[0], step[1]) for step in reversed(episode)]

        # Initialize expected return
        G = 0

        # For each step (T-1, T-2, ..., 0) of an episode
        for T, (state, action, reward) in enumerate(reversed(episode)):
            # Increment expected return
            G = mdp.gamma * G + reward
            if (state, action) not in state_actions[:T]:
                q[mdp.env.states.index(state)][mdp.env.actions.index(action)] += G
                visits[mdp.env.states.index(state)][mdp.env.actions.index(action)] += 1
                action_max = np.argmax(q[mdp.env.states.index(state)]/visits[mdp.env.states.index(state)])
                for a in mdp.env.actions:
                    if mdp.env.actions.index(a) == action_max:
                        policy[mdp.env.states.index(state)][mdp.env.actions.index(a)] = 1 - epsilon + epsilon/len(mdp.env.actions)
                    else:
                        policy[mdp.env.states.index(state)][mdp.env.actions.index(a)] = epsilon/len(mdp.env.actions)

    # Compute average expected return
    q /= visits

    return q, policy


def off_policy_prediction(mdp, episodes, steps, policy=None):
    # If target policy not given, assume equiprobable
    if policy is None:
        policy = 1/len(mdp.env.actions)*np.ones((len(mdp.env.states), len(mdp.env.actions)))

    # Behavioral policy
    b = 1/len(mdp.env.actions)*np.ones((len(mdp.env.states), len(mdp.env.actions)))

    # Initialize action-values matrix
    q = np.zeros((len(mdp.env.states), len(mdp.env.actions)))
    c = np.zeros_like(q)

    for _ in range(episodes):
        episode = mdp.generate_episode(steps, b)

        G = 0
        w = 1

        # For each step (T-1, T-2, ..., 0) of an episode
        for state, action, reward in reversed(episode):
            if w == 0:
                break

            state_id, action_id = mdp.env.states.index(state), mdp.env.actions.index(action)
            G = mdp.gamma * G + reward
            c[state_id][action_id] += w
            q[state_id][action_id] += w/c[state_id, action_id] * (G - q[state_id][action_id])

            w *= policy[state_id][action_id]/b[state_id][action_id]

    return q


def off_policy_control(mdp, episodes, steps):
    # Equiprobable target policy
    policy = 1/len(mdp.env.actions)*np.ones((len(mdp.env.states), len(mdp.env.actions)))

    # Equiprobable behavioral policy
    b = 1/len(mdp.env.actions)*np.ones((len(mdp.env.states), len(mdp.env.actions)))

    # Initialize action-values matrix
    q = np.zeros((len(mdp.env.states), len(mdp.env.actions)))
    c = np.zeros_like(q)

    for _ in range(episodes):
        episode = mdp.generate_episode(steps, b)

        G = 0
        w = 1

        # For each step (T-1, T-2, ..., 0) of an episode
        for state, action, reward in reversed(episode):
            state_id, action_id = mdp.env.states.index(state), mdp.env.actions.index(action)
            G = mdp.gamma * G + reward
            c[state_id][action_id] += w
            q[state_id][action_id] += w/c[state_id, action_id] * (G - q[state_id][action_id])

            max_action_id = np.argmax(q[state_id])
            policy[state_id] = np.zeros_like(policy[state_id])
            policy[state_id][max_action_id] = 1

            if action_id != max_action_id:
                break

            w *= 1/b[state_id][action_id]

    return q, policy

