from irl.mdp import load,tools
import irl.value_iteration
import irl.mdp.gridworld as gridworld
import sys
import os
import numpy as np
import random
import nltk
sys.path.append("D:/Ubicomp/Inverse-Reinforcement-Learning-master")


def markov_nll(trajectories):

    nll = 0

    policy_trajectory = []
    validation_trajectory = []

    for trajectory in trajectories:
        if random.random() < 0.5:
            policy_trajectory.append(trajectory)
        else:
            validation_trajectory.append(trajectory)

    pairs = []

    for trajectory in policy_trajectory:
        for slot in trajectory.keys():
            origin = trajectory[slot][1].get_origin()
            edge = trajectory[slot][1]
            temp = (origin, edge)
            pairs.append(temp)

    cfd = nltk.ConditionalFreqDist(pairs)

    length = 0

    for trajectory in validation_trajectory:
        prob = 1
        number = 0
        for slot in trajectory.keys():
            origin = trajectory[slot][1].get_origin()
            edge = trajectory[slot][1]

            count = 0
            total = 0

            for j in cfd[origin]:
                if edge.__eq__(j):
                    count = cfd[origin][j]
                    number += 1

                total += cfd[origin][j]
            if count > 0:
                print count, total
                prob *= float(count) / float(total)
            # print prob
        if number > 33:
            length += 1
            nll += np.log(prob)

    return -nll / length


def irl_nll(policy, trajectories):
    """

    :param policy: temporal policy
    :param trajectories: demonstrated trajectory set
    :return: negtive log-likelihood
    """
    nll = 0

    for trajectory in trajectories:
        prob = 1
        for step in range(12, 47):
            if step in trajectory.keys():
                state = trajectory[step][0]
                action = trajectory[step][1]

                if state in policy[step].keys():
                    if action in policy[step][state].keys():
                        prob *= policy[step][state][action]

        nll += np.log(prob)

    return -nll / len(trajectories)


def evaluation(path):
    id_traj = load.load_directory_trajectory(path + "training/")  # validation directory

    files = os.listdir(path+"param/")

    for filename in files:
        parampath = path + "param/" + filename
        if not os.path.isdir(parampath):

            trajectories = random.sample(id_traj.values(), 500)

            g = load.load_graph_traj(trajectories)
            gw = gridworld.Gridworld(g, 0.9)
            feature_matrix = gw.feature_matrix(g)

            t_alpha = {}
            with open(parampath, 'r') as f:
                t = 12
                for line in f:
                    line = line.strip('\n')
                    tokens = line.split(",")
                    param = np.zeros(11)
                    for j in range(11):
                        if len(tokens) > j:
                            param[j] = tokens[j]
                    t_alpha[t] = param.copy()
                    t += 1

            r = dict()
            for t in range(12, 48):
                r[t] = dict().fromkeys(g.get_edges(), 0)

            for edge in g.get_edges():
                for t in range(12, 48):
                    if t in t_alpha.keys():
                        r[t][edge] = feature_matrix[edge].dot(t_alpha[t])

            print "#######################################################"
            policy = irl.value_iteration.find_temporal_policy(g, r, 0.9, 46, stochastic=True)

            nll = irl_nll(policy, trajectories)
            m_nll = markov_nll(trajectories)

            print len(trajectories), nll, m_nll


def traj_dist(traj_tuple):
    dist = 0
    traj1 = traj_tuple[0]
    traj2 = traj_tuple[1]

    for i in range(12, 47):
        if i in traj1.keys() and i in traj2.keys():

            dist += tools.calculate_mesh_distance(traj1[i][0], traj2[i][0])

    return dist/35.0


def mean_error_dist(path):
    validation_id_traj = load.load_directory_trajectory(path + "training/")
    sim_id_traj = load.load_directory_trajectory(path + "sim/")
    markov_id_traj = load.load_directory_trajectory(path + "markovchain/")

    diff_list = []
    min_id = None

    validation_trajectoris = random.sample(validation_id_traj.values(),20)
    trajectories = random.sample(markov_id_traj.values(), 20)

    count = 0

    for validation_id in validation_id_traj.keys():
        count += 1
        if count == 50:
            break
        min_dist = sys.maxint
        dist = traj_dist((random.choice(trajectories), validation_id_traj[validation_id]))
        for trajectory in trajectories:
            dist = traj_dist((trajectory, validation_id_traj[validation_id]))
            if dist < min_dist:
                min_dist = dist
                min_id = validation_id
                print dist

        diff_list.append(dist)

        validation_id_traj.pop(min_id)

    diff_list.sort()

    print diff_list
    print np.average(diff_list)


def main(target):
    try:
        path = "D:/PT_Result/" + target + "/"

        # evaluation(path)

        mean_error_dist(path)

    except Exception:
        print "main class wrong"
        raise

if __name__ == '__main__':
    main("student")
