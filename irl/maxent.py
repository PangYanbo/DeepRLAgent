"""
Implements maximum entropy inverse reinforcement learning (Ziebart et al., 2008)


"""


import numpy as np


from . import value_iteration


def t_irl(graph, feature_matrix, trajectories, epochs, learning_rate, path):

    param = path + "param.csv"
    with open(param, "wb")as f:
        # initial reward function parameters
        alpha = dict().fromkeys(range(12,48), np.zeros(11))

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

        current_svf = prev_svf.copy()

        #action visited count

        for trajectory in trajectories:
            for t in range(12,47):
                if t in trajectory.keys():





        for t in range(12, 47):
            count = 0
            while True:
                count += 1
                # print "feature_expectations", feature_expectations[t]

                policy_feature_expectations = np.zeros(11)

                for j in range(11):
                    for edge in current_svf.keys():
                        policy_feature_expectations[j] += feature_matrix[edge][j] * current_svf[edge]

                grad = feature_expectations[t] - policy_feature_expectations

                if np.sqrt(grad.dot(grad)) < 1e-5:
                    alpha[t][4] = 10
                    break

                # print np.sqrt(grad.dot(grad))

                alpha[t] += learning_rate * grad

                for edge in graph.get_edges():
                    r[t][edge] = feature_matrix[edge].dot(alpha[t])

                current_svf = t_fi_ex_svf(graph, r, 0.9, prev_svf, t)

                # print "policy_expectations", t, policy_feature_expectations
                # print "grad", t, grad

                if count > epochs or np.sqrt(grad.dot(grad)) < 1e-5:
                    break
            print "alpha", t, alpha[t]
            prev_svf = current_svf.copy()
            for p in alpha[t]:
                f.write(str(p) + ",")
            f.write("\n")

        for edge in graph.get_edges():

            r[t][edge] = feature_matrix[edge].dot(alpha[t])

    return r


def fi_fe_ex(feature_matrix, trajectories):
    """

    :param feature_matrix
    :param trajectories:
    :return:
    """

    feature_expectations = np.zeros(11)
    for trajectory in trajectories:
        for i in trajectory.keys():
            edge = trajectory[i][1]
            if edge in feature_matrix.keys():
                feature_expectations += feature_matrix[edge]
    feature_expectations /= len(trajectories)
    return feature_expectations


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
    count = 0
    for i in range(12, 48):
        feature_expectations[i] = np.zeros(12)
        for trajectory in trajectories:
            if i in trajectory.keys():
                edge = trajectory[i][1]
                if edge in feature_matrix.keys():
                    feature_expectations[i][0:11] = feature_expectations[i][0:11] + feature_matrix[edge]
                    feature_expectations[i][12] =
        feature_expectations[i] /= len(trajectories)
    print count
    return feature_expectations


def find_expected_svf(graph, reward, discount, trajectories):
    """

    :param graph:
    :param reward:
    :param discount:
    :param trajectories: trajectory ={'12':(node,edge),'13':(node,edge),...'47':(node,edge)}
    :return:
    """

    n_trajectories = len(trajectories)
    trajectory_length = 35

    policy = value_iteration.find_policy(graph, reward, discount, v=None, stochastic=True)

    start_state_count = dict().fromkeys(graph.get_edges(), 0)

    for trajectory in trajectories:
        if 12 in trajectory:
            edge = trajectory[12][1]
            start_state_count[edge] += 1

    p_start_state = dict()
    for edge in graph.get_edges():
        p_start_state[edge] = float(start_state_count[edge]) / n_trajectories

    expected_svf = dict()
    for t in range(trajectory_length):
        expected_svf[t] = p_start_state.copy()

    for t in range(1, trajectory_length):
        for edge in graph.get_edges():
            expected_svf[t][edge] = 0
        for edge in graph.get_edges():
            dest = edge.get_destination()
            for _edge in policy[dest].keys():
                expected_svf[t][_edge] += expected_svf[t-1][edge] * policy[dest][_edge]

        # print "aaaaaaaaaaaaaaaaaaaaa", sum(e for k, e in expected_svf[t].items())

    exp_svf = {}.fromkeys(graph.get_edges(), 0)
    # e_f = {}.fromkeys(graph.get_edges(), 0)
    #
    for edge in graph.get_edges():
        for t in range(trajectory_length):
            exp_svf[edge] += expected_svf[t][edge]
    #
    # for node in graph.get_nodes():
    #     edge = policy[node]
    #     e_f[edge] = exp_svf[node]

    return exp_svf


def t_fi_ex_svf(graph, reward, discount, prev_svf, step):
    """
    :param prev_svf:
    :param graph:
    :param reward:
    :param discount:
    :param step:
    :return:
    """

    policy = value_iteration.find_temporal_policy(graph, reward, discount, step, stochastic=True)

    expected_svf = dict().fromkeys(graph.get_edges(), 0)

    for edge in prev_svf.keys():
        dest = edge.get_destination()
        for _edge in policy[step][dest].keys():
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


def irl(graph, feature_matrix, trajectories, epochs, learning_rate):
    alpha = np.array([0, 0, 0, 0, 0, 0, 0, -1.0, -4.0, -4.0, -4.0])
    print "initial alpha", alpha
    feature_expectations = fi_fe_ex(feature_matrix, trajectories)

    r = dict()

    for i in range(epochs):
        print "feature_expectations", feature_expectations

        for edge in graph.get_edges():
            r[edge] = feature_matrix[edge].dot(alpha)

        expected_svf = find_expected_svf(graph, r, 0.9, trajectories)

        policy_feature_expectations = np.zeros(11)
        for j in range(11):
            for edge in graph.get_edges():
                policy_feature_expectations[j] += feature_matrix[edge][j] * expected_svf[edge]

        grad = feature_expectations - policy_feature_expectations
        alpha += learning_rate * grad

        print "grad", grad
        print "policy expectations", policy_feature_expectations
        print "alpha", alpha
        # print "reward",r
    for edge in graph.get_edges():
        r[edge] = feature_matrix[edge].dot(alpha)

    return alpha
