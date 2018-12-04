import numpy as np
# import irl.mdp.UrbanEnv as env


def choose_action(s, q_table, env, epsilon):
    """

    :return:
    """
    if np.random.uniform() < epsilon:
        state_action = q_table[s]
        # state_action = q_table[env.sub_action_space(s)]
        # state_action = state_action.reindex(np.random.permutation(state_action.index))  # some actions have same value
        action, value = max(state_action.items(), key=lambda x: x[1])
    else:
        action = env.index_action[np.random.choice(env.sub_action_space(s))]
        value = q_table[s][action]

    return action, value


def q_learning(env, discount, reward, learning_rate=0.001, epsilon=0.9, epoch=50):
    """
    how to decide convergence condition?
    until culmulative reward for each episode stop increasing?
    :return:
    """
    # initialize Q value
    # Q = defaultdict(lambda: np.zeros(env.action_space.n))
    q_table = dict()

    for s in env.state_space:
        action_value = dict()
        for a in env.sub_action_space(s):
            action_value[env.index_action[a]] = np.random.rand()
        q_table[s] = action_value

    while epoch > 0:
        for s in env.state_space:
            a, v = choose_action(s, q_table, env, epsilon)
            r = reward[a]
            # TD update
            # q_table[a] += learning_rate * (
            # r + discount * np.max(q_table[env.sub_action_space(env.index_state[s])]) - q_table[a])
            q_table[s][a] += learning_rate * (
                        r + discount * v - q_table[s][a])
        epoch -= 1

    return q_table


def find_policy(env, q_table):
    """

    :param env:
    :param q_table:
    :return: nested dict policy->State->Action
    """
    policy = dict()

    for s in env.state_space:
        action_prob = dict()
        exp_value_sum = sum(np.exp(list(q_table[s].values())))
        if exp_value_sum == 0:
            print(s, q_table[s])
            action_prob[env.index_action[env.sub_action_space(s)[0]]] = 1
            policy[s] = action_prob
        else:
            for a in env.sub_action_space(s):
                _a = env.index_action[a]
                action_prob[_a] = np.exp(q_table[s][_a]) / exp_value_sum
            policy[s] = action_prob

    return policy
