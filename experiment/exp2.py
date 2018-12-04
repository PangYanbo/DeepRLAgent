"""
experiment 2
train agent by trajectories collected from zone A and simulate in zone B
mesh unit
evaluate method: average trajectory distance
competitive method: expansion, markov chain, discrete choice model,
"""
import os
import random
import sys

import nltk
import numpy as np
from sklearn.cross_validation import train_test_split

import irl.maxent as maxent
import irl.mdp.gridworld as gridworld
from irl.mdp import load
from utils import tools, load


def read_list(path):
    mesh_list = []
    with open(path, "r")as f:
        for line in f.readlines():
            tokens = line.strip("\r\n").split(",")
            mesh = tokens[2]
            mesh_list.append(mesh)
    return mesh_list


def read_param(path):
    t_alpha = {}
    with open(path, 'r') as f:
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
    return t_alpha


def traj_dist(traj_tuple):
    dist = 0
    traj1 = traj_tuple[0]
    traj2 = traj_tuple[1]

    for i in range(12, 47):
        if i in traj1.keys() and i in traj2.keys():

            dist += tools.calculate_mesh_distance(traj1[i][0], traj2[i][0])

    return dist/35.0


def makepairs(trajectories):
    pairs = []
    for trajectory in trajectories:
        for slot in trajectory.keys():
            origin = trajectory[slot][1].get_origin()
            edge = trajectory[slot][1]
            temp = (origin, edge)
            pairs.append(temp)
    return pairs


def generate(cfd,aid, path, word, num=36):
    out = open(path, 'w')
    for i in range(num):
        # make an array with the words shown by proper count

        arr = []

        for j in cfd[word]:
            for k in range(cfd[word][j]):
                arr.append(j)

        # choose the word randomly from the conditional distribution

        if len(arr) > 0:
            word = arr[int((len(arr)) * random.random())]

            out.write(aid+','+str(12+i)+','+word.get_origin()+','+word.get_destination()+','+word.get_mode()+'\n')
            word = word.get_destination()
        else:
            out.write(str(i) + ',' + str(
                12 + i) + ',' + word + ',' + word + ',' + "stay" + '\n')
    out.close()


def main():
    root = "/home/ubuntu/Data/pflow_data/pflow-csv/"

    mesh_list = read_list("/home/ubuntu/Data/Tokyo/MeshCode/Tokyo.csv")

    list_dirs = os.walk(root)
    count = 0
    print mesh_list
    for root, dirs, files in list_dirs:
        with open("/home/ubuntu/Data/PT_Result/exp1/result.csv", "w") as f:
            for d in dirs:
                if d in mesh_list:
                    file_list = os.listdir(os.path.join(root, d))
                    if len(file_list) > 100 and "train_irl.csv" in file_list:
                        count += 1
                        id_traj = load.load_trajectory(os.path.join(root, d) + "/train_irl.csv")

                        train, validation = train_test_split(id_traj.values(), test_size=0.4)

                        g = load.load_graph_traj(train)
                        gw = gridworld.Gridworld(g, 0.9)
                        feature_matrix = gw.feature_matrix(g)

                        path = "/home/ubuntu/Data/PT_Result/exp1/"

                        # train
                        if not os.path.exists(path + "parameter/" + d + "param.csv"):
                            maxent.t_irl(g, feature_matrix, train, 200, 0.2, path + "parameter/" + d)

                        # simulation

                        t_alpha = read_param(path + "parameter/" + os.listdir(path+"parameter/")[0])

                        r = dict()
                        for t in range(12, 48):
                            r[t] = dict().fromkeys(g.get_edges(), 0)

                        for edge in g.get_edges():
                            for t in range(12, 48):
                                if t in t_alpha.keys():
                                    r[t][edge] = feature_matrix[edge].dot(t_alpha[t])

                        if not os.path.exists(path + "sim/" + d + "/"):
                            os.mkdir(path + "sim/" + d + "/")

                        for i in range(80):
                            tools.generate_temporal_traj(g, r, d, 0.5, path + "sim/" + d + "/", d + "_" + str(i))

                        # markov chain
                        if not os.path.exists(path + "markov/" + d + "/"):
                            os.mkdir(path + "markov/" + d + "/")

                        for i in range(80):
                            pairs = makepairs(train)

                            cfd = nltk.ConditionalFreqDist(pairs)

                            generate(cfd, str(i), path + "markov/" + d + "/" + str(i) + ".csv", d)

                        # expansion validation
                        expansion10_trajecotry = random.sample(train, int(len(train)*0.1))

                        diff_list = []

                        for validation_traj in validation:
                            min_dist = sys.maxint
                            for traj in expansion10_trajecotry:
                                dist = traj_dist((traj, validation_traj))

                                if dist < min_dist:
                                    min_dist = dist

                            diff_list.append(min_dist)

                        expansion10_score = np.average(diff_list)

                        expansion50_trajecotry = random.sample(train, int(len(train) * 0.5))

                        diff_list = []

                        for validation_traj in validation:
                            min_dist = sys.maxint
                            for traj in expansion50_trajecotry:
                                dist = traj_dist((traj, validation_traj))

                                if dist < min_dist:
                                    min_dist = dist

                            diff_list.append(min_dist)

                        expansion50_score = np.average(diff_list)

                        # validation

                        markov_id_traj = load.load_directory_trajectory(path + "markov/" + d + "/")

                        diff_list = []

                        print markov_id_traj.keys()
                        for traj in validation:
                            min_dist = sys.maxint
                            for markov_id in markov_id_traj.keys():

                                dist = traj_dist((traj, markov_id_traj[markov_id]))

                                if dist < min_dist:
                                    min_dist = dist
                            diff_list.append(min_dist)

                        markov_score = np.average(diff_list)

                        sim_id_traj = load.load_directory_trajectory(path + "sim/" + d + "/")

                        diff_list = []

                        for traj in validation:
                            min_dist = sys.maxint
                            for sim_id in sim_id_traj.keys():
                                dist = traj_dist((traj, sim_id_traj[sim_id]))

                                if dist < min_dist:
                                    min_dist = dist
                            if min_dist > 10:
                                continue
                            diff_list.append(min_dist)

                        sim_score = np.average(diff_list)

                        print d+","+str(sim_score)+","+str(markov_score)+","+str(expansion10_score)+","+str(expansion50_score)
                        f.write(d+","+str(sim_score)+","+str(markov_score)+","+str(expansion10_score)+","+str(expansion50_score))
                        f.write("\n")

                        if count > 80:
                            f.close()


if __name__ == '__main__':
    main()
