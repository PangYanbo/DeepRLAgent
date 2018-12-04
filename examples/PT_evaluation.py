import os
import random
import sys

import numpy as np

import irl.mdp.gridworld as gridworld
import irl.solver.value_iteration
from irl.mdp import load
from utils import tools, load
from examples. import cfd

sys.path.append("D:/Ubicomp/Inverse-Reinforcement-Learning-master")


def markov_nll(train_traj, valid_traj):

    nll = 0

    pairs = []

    for trajectory in train_traj:
        for slot in trajectory.keys():
            origin = trajectory[slot][1].get_origin()
            edge = trajectory[slot][1]
            temp = (origin, edge)
            pairs.append(temp)

    prob = cfd(pairs)

    length = 0

    for trajectory in valid_traj:
        prob = 1
        number = 0
        for slot in trajectory.keys():
            origin = trajectory[slot][1].get_origin()
            edge = trajectory[slot][1]

            count = 0
            total = 0

            for j in cfd[origin]:
                if edge.__eq__(j):
                    count = prob[origin][j]
                    number += 1

                total += cfd[origin][j]
            if count > 0:
                # print count, total
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
                    if action in policy[step][state].keys() and policy[step][state][action] > 0.01:
                        prob *= policy[step][state][action]

        nll += np.log(prob)

    return -nll / len(trajectories)


def dcm_nll(path, trajectories):
    policy_trajectory, validation_trajectory = train_test_split(trajectories, test_size=0.5)

    files = os.listdir(path + "param/")

    parampath = path + "param/" + random.choice(files)

    nll = 0
    if not os.path.isdir(parampath):

        # trajectories = random.sample(trajectories, 50)
        try:
            g = load.load_graph_traj(validation_trajectory)

            pop = dcm.pop_feature()
            office = dcm.get_business()

            param = np.zeros(16)
            with open(parampath, 'r') as f:
                for line in f:
                    line = line.strip('\n')
                    tokens = line.split(",")
                    param = np.zeros(16)
                    for j in range(16):
                        if len(tokens) > j:
                            param[j] = tokens[j]

            for trajectory in validation_trajectory:
                prob = 1
                start = trajectory[12][0]
                for step in range(12, 47):
                    if step in trajectory.keys():
                        state = trajectory[step][0]
                        action = trajectory[step][1]

                        policy = dcm.policy(g, param, state, start, step, pop, office)

                        if action in policy[step].keys():
                            if policy[step][action] > 0:
                                if action.get_mode() != "stay":
                                    prob *= (policy[step][action])/ 8.0
                                else:
                                    prob *= 0.875

                nll += np.log(prob)
        except KeyError:
            print "KeyError"

        return -nll / len(validation_trajectory)

def evaluation():

    id_traj = load.load_trajectory("/home/t-iho/Result/trainingdata/Hiroshima500m/20140819HINAN/all_files.csv")

    target_traj = load.load_trajectory('/home/t-iho/Result/trainingdata/Hiroshima/20180706HINAN/all_files.csv')

    markov_score = markov_nll(id_traj.values(), target_traj.valuese())



    irl_score = irl_nll("", target_traj)

def evaluation(path):
    id_traj = load.load_directory_trajectory(path + "training/")

    files = os.listdir(path+"param/")

    with open(path + "evaluation_nll_result.csv", "w") as w:
        w.write("No."+","+"length" + "," + "irl" + "," + "markov" + "\n")
        count = 0
        for filename in files:
            parampath = path + "param/" + filename
            if not os.path.isdir(parampath):

                trajectories = random.sample(id_traj.values(), 500)

                policy_trajectory, validation_trajectory = train_test_split(trajectories, test_size=0.5)

                g = load.load_graph_traj(validation_trajectory)
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
                policy = irl.solver.value_iteration.find_temporal_policy(g, r, 0.9, 46, stochastic=True)

                nll = irl_nll(policy, validation_trajectory)
                m_nll = markov_nll(trajectories)
                d_nll = dcm_nll("/home/ubuntu/Data/PT_Result/discrete_choice_model/", trajectories)

                print len(trajectories), nll, m_nll, d_nll

                w.write(str(count)+","+str(len(trajectories))+","+str(nll)+","+str(m_nll)+"\n")
                count += 1


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
    print "load sim"
    sim_id_traj = load.load_directory_trajectory(path + "sim/")
    markov_id_traj = load.load_directory_trajectory(path + "markovchain/")

    diff_list = []
    min_id = None

    # validation_trajectoris = random.sample(validation_id_traj.values(), 200)
    trajectories = sim_id_traj.values()

    count = 0

    for validation_id in validation_id_traj.keys():
        count += 1
        if count == 5000:
            break
        min_dist = sys.maxint
        dist = 0
        # dist = traj_dist((random.choice(trajectories), validation_id_traj[validation_id]))
        for trajectory in trajectories:
            if trajectory[12][0] == validation_id_traj[validation_id][12][0]:
                dist = traj_dist((trajectory, validation_id_traj[validation_id]))
                if dist < min_dist:
                    min_dist = dist
                    # min_id = validation_id
                    print dist
        if dist ==0:
            continue
        diff_list.append(dist)

        # validation_id_traj.pop(min_id)

    diff_list.sort()

    print diff_list
    print np.average(diff_list)


def main(target):
    try:
        path = "/home/ubuntu/Data/PT_Result/" + target + "/"

        # evaluation(path)

        mean_error_dist(path)

    except Exception:
        print "main class wrong"
        raise


if __name__ == '__main__':
    main("commuter")
