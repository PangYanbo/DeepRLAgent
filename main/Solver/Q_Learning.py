import numpy as np
# import irl.mdp.UrbanEnv as env


def choose_action(s, q_table, env, epsilon):
    """

    :return:
    """
    if np.random.uniform() < epsilon:
        state_action = q_table[env.sub_action_space(s)]
        state_action = state_action.reindex(np.random.permutation(state_action.index))  # some actions have same value
        action = state_action.idxmax()
    else:
        action = np.random.choice(env.sub_action_space(s))

    return action


def q_learning(env, reward, learning_rate, discount, epsilon, epoch):
    """
    how to decide convergence condition?
    until culmulative reward for each episode stop increasing?
    :return:
    """
    # initialize Q value
    q_table = np.zeros(env.n_actions)
    hist_reward = []

    while epoch > 0:
        t = 12
        s = 0
        sum_reward = 0
        while t < 48:
            a = choose_action(s, q_table, env, epsilon)
            r = reward[a]
            _s = env.index_action[a].get_destination()
            q_table[a] = q_table[a] + learning_rate * (r + discount *
                                                       np.max(q_table[env.sub_action_space(s)]) - q_table[a])
            s = _s
            t += 1
            sum_reward += r
        hist_reward.append(sum_reward)
        epoch -= 1

    return q_table
