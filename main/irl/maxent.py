import numpy as np
from main.Env.gridworld import State
import datetime

from main.Solver import value_iteration


def t_irl(graph, feature_matrix, trajectories, epochs, learning_rate, path):
    param = path + "param.csv"
    with open(param, "wb")as f:
        # initial reward function parameters
        alpha = dict()
        for t in range(12, 48):
            alpha[t] = np.zeros(31)

        r = dict()
        for t in range(12, 48):
            r[t] = dict().fromkeys(graph.get_edges(), 0)

        # derive feature expectations from demonstrated trajectories
        feature_expectations = t_fi_fe_ex(feature_matrix, trajectories)

        # calculate initial state frequency
        start_state_count = dict().fromkeys(graph.get_edges(), 0)

        for trajectory in trajectories:
            if 12 in trajectory and trajectory[12][1] in graph.get_edges():
                edge = trajectory[12][1]
                start_state_count[edge] += 1

        prev_svf = dict()

        for edge in graph.get_edges():
            prev_svf[edge] = float(start_state_count[edge]) / len(trajectories)

        # current_svf = prev_svf.copy()
        current_svf = dict().fromkeys(graph.get_edges(), 0)

        for t in range(12, 47):
            count = 0
            norms = []
            while True:
                count += 1
                # print "feature_expectations", feature_expectations[t]

                policy_feature_expectations = np.zeros(31)

                for j in range(31):
                    for edge in current_svf:
                        policy_feature_expectations[j] += feature_matrix[t][edge][j] * current_svf[edge]

                grad = feature_expectations[t+1] - policy_feature_expectations

                norms.append(np.linalg.norm(grad))

                if count % 100 == 0:
                    print("feature_expectations", feature_expectations[t])
                    print(grad)

                alpha[t] += learning_rate * grad

                for edge in graph.get_edges():
                    r[t][edge] = feature_matrix[t][edge].dot(alpha[t])

                starttime = datetime.datetime.now()
                current_svf = t_fi_ex_svf(graph, r, 0.9, prev_svf, t)
                endtime = datetime.datetime.now()
                print("svf time: ", str(endtime-starttime))
                # print "policy_expectations", t, policy_feature_expectations
                # print "grad", t, grad

                if count > epochs or np.sqrt(grad.dot(grad)) < 1e-4:
                    break
            # import matplotlib.pyplot as plt0
            # plt.figure()
            # plt.title("learning_rate: "+str(learning_rate)+", iteration: "+str(epochs))
            # # plt.ylim(0, 0.1)
            # plt.plot(range(len(norms)), norms)
            # plt.savefig(path + "_" + str(t) + "_" + str(learning_rate) + "_" + str(epochs) + ".png")

            print("alpha", t, alpha[t])
            prev_svf = current_svf.copy()
            for p in alpha[t]:
                f.write((str(p) + ",").encode())
            f.write("\n".encode())

        for t in range(12, 48):
            for edge in graph.get_edges():
                r[t][edge] = feature_matrix[t][edge].dot(alpha[t])
                # print "t:", t, "edge: ", edge, "reward:", r[t][edge]

    return r


def t_fi_fe_ex(feature_matrix, trajectories):
    """

    :param feature_matrix
    :param trajectories:
    :return:
    """
    slot_list = []
    for t in range(12, 48):
        slot_list.append(t)

    feature_expectations = dict()
    for i in range(12, 48):
        feature_expectations[i] = np.zeros(31)
        for trajectory in trajectories:
            if i in trajectory:
                edge = trajectory[i][1]
                if edge in feature_matrix[i]:
                    feature_expectations[i] += feature_matrix[i][edge]
                    # print "time", i, "edge: ", edge, "vector:", feature_matrix[edge]
        feature_expectations[i] /= len(trajectories)
    feature_expectations[47] = feature_expectations[46]

    return feature_expectations


def t_fi_ex_svf(graph, reward, discount, prev_svf, step):
    """
    :param prev_svf:
    :param graph:
    :param reward:
    :param discount:
    :param step:
    :return:
    """

    starttime = datetime.datetime.now()
    policy = value_iteration.find_temporal_policy(graph, reward, discount, step, stochastic=True)
    endtime = datetime.datetime.now()
    print(str(endtime-starttime))

    expected_svf = dict().fromkeys(graph.get_edges(), 0)

    for edge in prev_svf:
        dest = edge.get_destination()
        for _edge in policy[step][dest]:
            expected_svf[_edge] += prev_svf[edge]*policy[step][dest][_edge]

    # for edge in graph.get_edges():
    #     dest = policy[edge.get_destination()].get_destination()
    #     expected_svf[dest] += prev_svf[edge]*policy[edge.get_destination()]
    #
    # edge_svf = dict.fromkeys(graph.get_edges(), 0)
    #
    # for node in graph.get_nodes():
    #     edge = policy[node]
    #     edge_svf[edge] = expected_svf[node]

    return expected_svf


def find_expected_avf(gridworld, reward):
    """
    trajectory length 36
    :param gridworld:
    :param reward: temporal reward
    :return: dict[t]-> prob(action)
    """
    policy = gridworld.find_policy(reward, threshold=1e-2, v=None)

    expected_avf = np.zeros([36, gridworld.n_action])

    for trajectory in gridworld.trajectories:
        expected_avf[12, [trajectory[0][1]]] += 1

    expected_avf = expected_avf / float(len(gridworld.trajectories))

    for t in range(13, 48):
        for a in range(gridworld.n_action):
            state = State(gridworld.index_action[a].get_destination(), t-1)
            s = gridworld.state_index[state]
            for _a in range(gridworld.n_action): # last time step action
                expected_avf[t, a] += expected_avf[t-1, _a] * policy(s, a)

    return expected_avf


def irl(gridworld, epochs, learning_rate, path):
    """
    reward(t,action) -> R
    :param gridworld:
    :param epochs:
    :param learning_rate:
    :param path:
    :return:
    """
    param = path + "param.csv"
    feature_matrix = gridworld.feature_matrix
    with open(param, "wb")as f:
        # initial reward function parameters
        alpha = np.zeros([gridworld.n_features])

        r = np.zeros([37, gridworld.n_action])

        # derive feature expectations from demonstrated trajectories
        feature_expectations = gridworld.find_feature_expections()

        expected_svf = find_expected_avf(gridworld, r)

        count = 0
        norms = []
        while True:
            count += 1
            # print "feature_expectations", feature_expectations[t]

            policy_feature_expectations = np.zeros(gridworld.n_features)

            for j in range(gridworld.n_features):
                for s in range(gridworld.n_action):
                    for a in gridworld.sub_action_space(s):
                        policy_feature_expectations[j] += feature_matrix[s][a][j] * expected_svf[s][a]

            grad = feature_expectations - policy_feature_expectations

            norms.append(np.linalg.norm(grad))

            if count % 100 == 0:
                print("feature_expectations", feature_expectations)
                print(grad)

            alpha += learning_rate * grad

            # print "policy_expectations", t, policy_feature_expectations
            # print "grad", t, grad

            if count > epochs or np.sqrt(grad.dot(grad)) < 1e-4:
                break

        # plt.figure()
        # plt.title("learning_rate: "+str(learning_rate)+", iteration: "+str(epochs))
        # # plt.ylim(0, 0.1)
        # plt.plot(range(len(norms)), norms)
        # plt.savefig(path + "_" + str(t) + "_" + str(learning_rate) + "_" + str(epochs) + ".png")

        print("alpha", alpha)

        for p in alpha:
            f.write((str(p) + ",").encode())
        f.write("\n".encode())

        for t in range(12, 48):
            for action in range(gridworld.n_action):
                r[t][action] = feature_matrix[t][action].dot(alpha)
                # print "t:", t, "edge: ", edge, "reward:", r[t][edge]

    return r


def optimal_value(graph, reward, discount):
    """

    :param graph:
    :param reward:
    :param discount:
    :return:
    """
    value = []

    temporal_v = {}.fromkeys(range(12, 48), 0)

    v = {}.fromkeys(graph.get_nodes(), temporal_v)

    for t in reversed(range(12, 48)):
        for node in graph.get_nodes():
            max_v = float("-inf")
            for edge in graph.get_node(node).get_edges():
                max_v = max(max_v, reward[t][edge] + discount * v[node][t])
            v[node][t] = max_v
            value.append(v[node][t])

    return v


def find_policy(graph, reward, discount, v=None, stochastic=True):
    """
    :param graph:
    :param reward:
    :param discount:
    :param v:
    :param stochastic:
    :return:
    """
    if v is None:
        v = optimal_value(graph, reward, discount)
    if stochastic:
        Q = dict()
        for t in range(12, 48):
            Q[t] = dict()
            for node in graph.get_nodes():
                Q[t][node] = dict()
                for edge in graph.get_node(node).get_edges():
                    Q[t][node][edge] = reward[t][edge] + discount * v[edge.get_destination()][t]
                max_value = max(Q[t][node].items(), key=lambda x: x[1])[1]
                temp = 0
                for edge in graph.get_node(node).get_edges():
                    Q[t][node][edge] -= max_value
                    temp += np.exp(Q[t][node][edge])

                for edge in graph.get_node(node).get_edges():
                    Q[t][node][edge] = np.exp(Q[t][node][edge]) / temp
        return Q
