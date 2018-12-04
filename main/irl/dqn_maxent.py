import numpy as np


def irl(graph, env, trajectories, epochs, learning_rate, path):
    param = path + "param.csv"
    with open(param, "wb")as f:
        # initial reward function parameters
        alpha = dict()
        for t in range(12, 48):
            alpha[t] = np.zeros(15)

        r = dict()
        for t in range(12, 48):
            r[t] = dict().fromkeys(graph.get_edges(), 0)

        # derive feature expectations from demonstrated trajectories
        feature_expectations = find_feature_expection(env.feature_matrix, trajectories)

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

                    policy_feature_expectations = np.zeros(15)

                    for j in range(15):
                        for edge in current_svf.keys():
                            policy_feature_expectations[j] += env.feature_matrix[t][edge][j] * current_svf[edge]

                    grad = feature_expectations[t + 1] - policy_feature_expectations

                    norms.append(np.linalg.norm(grad))

                    if count % 100 == 0:
                        print("feature_expectations", feature_expectations[t])
                        print(grad)

                    alpha[t] += learning_rate * grad

                    for edge in graph.get_edges():
                        r[t][edge] = env.feature_matrix[t][edge].dot(alpha[t])

                    current_svf = find_expected_svf(graph, r, 0.9, prev_svf, t)


def find_feature_expection(feature_matrix, trajectories):
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
        feature_expectations[i] = np.zeros(15)
        for trajectory in trajectories:
            if i in trajectory.keys():
                edge = trajectory[i][1]
                if edge in feature_matrix.keys():
                    feature_expectations[i] += feature_matrix[edge]
                    # print "time", i, "edge: ", edge, "vector:", feature_matrix[edge]
        feature_expectations[i] /= len(trajectories)
    feature_expectations[47] = feature_expectations[46]

    return feature_expectations


def find_expected_svf(graph, reward, discount, prev_svf, step):
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