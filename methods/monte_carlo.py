import numpy as np

from mdp.markov_decision_process import MDP


def first_visit_prediction(mdp: MDP, episodes, steps, policy=None):
    # If policy not given, assume equiprobable
    if policy is None:
        policy = mdp.equiprobable_policy()

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
        policy = mdp.equiprobable_policy()

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
        policy = mdp.equiprobable_policy()

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
        policy = mdp.equiprobable_policy()

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
        policy = mdp.equiprobable_policy()

    # Behavioral policy
    b = mdp.equiprobable_policy()

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

            s, a = mdp.env.states.index(state), mdp.env.actions.index(action)
            G = mdp.gamma * G + reward
            c[s][a] += w
            q[s][a] += w/c[s, a] * (G - q[s][a])

            w *= policy[s][a]/b[s][a]

    return q


def off_policy_control(mdp, episodes, steps):
    # Equiprobable target policy
    policy = mdp.equiprobable_policy()

    # Equiprobable behavioral policy
    b = mdp.equiprobable_policy()

    # Initialize action-values matrix
    q = np.zeros((len(mdp.env.states), len(mdp.env.actions)))
    c = np.zeros_like(q)

    for _ in range(episodes):
        episode = mdp.generate_episode(steps, b)

        G = 0
        w = 1

        # For each step (T-1, T-2, ..., 0) of an episode
        for state, action, reward in reversed(episode):
            s, a = mdp.env.states.index(state), mdp.env.actions.index(action)
            G = mdp.gamma * G + reward
            c[s][a] += w
            q[s][a] += w/c[s, a] * (G - q[s][a])

            max_a = np.argmax(q[s])
            policy[s] = np.zeros_like(policy[s])
            policy[s][max_a] = 1

            if a != max_a:
                break

            w *= 1/b[s][a]

    return q, policy

