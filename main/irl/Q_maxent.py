from Q_Learning.one_step_q import q_learning, find_policy
import numpy as np


def irl(env, epochs, discount, learning_rate, path):
    param = path + "param.csv"
    with open(param, "wb")as f:
        # initial reward function parameters
        alpha = dict()
        for t in range(12, 48):
            alpha[t] = np.zeros(env.n_features)

        r = env.reward(alpha)

        # derive feature expectations from demonstrated trajectories
        feature_expectations = find_feature_expectation(env)

        # calculate initial state frequency
        start_state_count = dict().fromkeys(env.graph.get_edges(), 0)

        for trajectory in env.trajectories:
            if 12 in trajectory and trajectory[12][1] in env.graph.get_edges():
                edge = trajectory[12][1]
                start_state_count[edge] += 1

        prev_svf = dict()

        for edge in env.graph.get_edges():
            prev_svf[edge] = float(start_state_count[edge]) / len(env.trajectories)

        # current_svf = prev_svf.copy()
        current_svf = dict().fromkeys(env.graph.get_edges(), 0)

        for t in range(12, 47):
            count = 0
            norms = []
            while True:
                count += 1
                # print "feature_expectations", feature_expectations[t]

                policy_feature_expectations = np.zeros(env.n_features)

                for edge in current_svf:
                    a = env.action_index[edge]
                    # print(env.feature_matrix[t][a] * current_svf[edge])
                    policy_feature_expectations += env.feature_matrix[t][a] * current_svf[edge]

                grad = feature_expectations[t + 1] - policy_feature_expectations
                print("grad", t, grad)

                norms.append(np.linalg.norm(grad))

                if count % 100 == 0:
                    print("feature_expectations", feature_expectations[t])
                    print(grad)

                alpha[t] += learning_rate * grad

                for edge in env.graph.get_edges():
                    a = env.action_index[edge]
                    r[t][edge] = env.feature_matrix[t][a].dot(alpha[t])

                current_svf = find_expected_svf(env, r[t], discount, prev_svf)

                if count > epochs or np.sqrt(grad.dot(grad)) < 1e-4:
                    break

            print("alpha", t, alpha[t])
            prev_svf = current_svf.copy()
            for p in alpha[t]:
                f.write((str(p) + ",").encode())
            f.write("\n".encode())

        return env.reward(alpha)


def find_feature_expectation(env):
    slot_list = []
    for t in range(12, 48):
        slot_list.append(t)

    feature_expectations = dict()
    for i in range(12, 48):
        feature_expectations[i] = np.zeros(env.n_features)
        for trajectory in env.trajectories:
            if i in trajectory:
                a = env.action_index[trajectory[i][1]]
                if a in env.feature_matrix[i-12]:
                    feature_expectations[i] += env.feature_matrix[i-12][a]
                    # print "time", i, "edge: ", edge, "vector:", feature_matrix[edge]
        feature_expectations[i] /= len(env.trajectories)
    feature_expectations[47] = feature_expectations[46]

    return feature_expectations


def find_expected_svf(env, reward, discount, prev_svf):
    q_table = q_learning(env, discount, reward)
    policy = find_policy(env, q_table)

    expected_svf = dict().fromkeys(env.graph.get_edges(), 0)

    for edge in prev_svf:
        dest = edge.get_destination()
        for _edge in policy[dest]:
            expected_svf[_edge] += prev_svf[edge] * policy[dest][_edge]

    return expected_svf
