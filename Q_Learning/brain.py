import numpy as np
# import irl.mdp.UrbanEnv as env


def choose_action(s, q_table, env, epsilon):
    """

    :return:
    """
    s = env.index_state[s]
    if np.random.uniform() < epsilon:
        state_action = dict((k, q_table[k]) for k in env.sub_action_space(s) if k in q_table)
        # state_action = q_table[env.sub_action_space(s)]
        # state_action = state_action.reindex(np.random.permutation(state_action.index))  # some actions have same value
        action, value = max(state_action.items(), key=lambda x: x[1])
    else:
        action = np.random.choice(env.sub_action_space(s))
        value = q_table[action]
    return action, value


def q_learning(env, reward, learning_rate, discount, epsilon, epoch):
    """
    how to decide convergence condition?
    until culmulative reward for each episode stop increasing?
    :return:
    """
    # initialize Q value
    # Q = defaultdict(lambda: np.zeros(env.action_space.n))
    q_table = dict.fromkeys(range(env.n_action), np.random.rand())
    hist_reward = []
    avg_reward = []

    while epoch > 0:
        print(q_table)
        t = 12
        s = 0
        sum_reward = 0
        while t < 47:
            a, v = choose_action(s, q_table, env, epsilon)
            r = reward[t][env.index_action[a]]
            _s = env.index_action[a].get_destination()
            # TD update
            # q_table[a] += learning_rate * (
            # r + discount * np.max(q_table[env.sub_action_space(env.index_state[s])]) - q_table[a])
            q_table[a] += learning_rate * (
                        r + discount * v - q_table[a])
            s = env.state_index[_s]
            t += 1
            sum_reward += r
        hist_reward.append(sum_reward)
        avg_reward.append(np.average(hist_reward))
        epoch -= 1

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(hist_reward)), avg_reward)
    plt.ylabel('Averaged Reward of a Episode')
    plt.xlabel('training steps')
    plt.show()

    return q_table
