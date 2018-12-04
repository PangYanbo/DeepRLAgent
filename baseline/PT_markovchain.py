# -*- coding:utf-8 -*-
import os
import random
import sys
import codecs
import numpy as np


from utils import load
sys.path.append("D:/Ubicomp/Inverse-Reinforcement-Learning-master")


def makepairs(trajectories):
    pairs = []
    for trajectory in trajectories:
        for slot in trajectory.keys():
            origin = trajectory[slot][1].get_origin()
            edge = trajectory[slot][1]
            temp = (origin, edge)
            pairs.append(temp)
    return pairs


def cfd(pairs):
    s_a_count = dict()
    for pair in pairs:
        state = pair[0]
        action = pair[1]
        if state not in s_a_count:
            a_prob = dict()
            a_prob[action] = 1
            s_a_count[state] = a_prob
        else:
            if action not in s_a_count[state]:
                s_a_count[state][action] = 1
            else:
                s_a_count[state][action] += 1

    s_a_prob = dict()
    for state in s_a_count:
        temp = sum(s_a_count[state].values())
        action_prob = dict()
        s_a_prob[state] = action_prob
        for action in s_a_count[state]:
            s_a_prob[state][action] = s_a_count[state][action] /temp

    return s_a_prob


def pick_by_weight(d):
    d_choices = []
    d_probs = []
    for k, v in d.items():
      d_choices.append(k)
      d_probs.append(v)
    return np.random.choice(d_choices, 1, p=d_probs)[0]


def generate(cfd, path, word, num=36):
    out = open(path, 'w')

    for i in range(num):
        # make an array with the words shown by proper count
        if word in cfd:
            d = cfd[word]
            word = pick_by_weight(d)
            print(word)
            out.write(str(i) + ',' + str(
                12 + i) + ',' + word.get_origin() + ',' + word.get_destination() + ',' + word.get_mode() + '\n')
            word = word.get_destination()
        else:
            out.write(str(i) + ',' + str(
                12 + i) + ',' + word + ',' + word + ',' + "stay" + '\n')






        # arr = []
        #
        # for j in cfd[word]:
        #     for k in range(cfd[word][j]):
        #         arr.append(j)
        #
        # # choose the word randomly from the conditional distribution
        #
        # if len(arr) > 0:
        #     word = arr[int((len(arr)) * random.random())]
        #
        #     out.write(str(i)+','+str(12+i)+','+word.get_origin()+','+word.get_destination()+','+word.get_mode()+'\n')
        #     word = word.get_destination()
        # else:
        #     out.write(str(i) + ',' + str(
        #         12 + i) + ',' + word + ',' + word + ',' + "stay" + '\n')
    out.close()


def read_list(path):
    mesh_list = []
    with codecs.open(path, "r", "Shift-jis", 'ignore')as f:
        for line in f.readlines():
            tokens = line.strip("\r\n").split(",")
            mesh = tokens[2]
            mesh_list.append(mesh)
    return mesh_list


def main(target):

    root = "/home/ubuntu/Data/PT_resutl/" + target + "/"

    mesh_list = read_list("/home/ubuntu/Data/Tokyo/MeshCode/Tokyo.csv")

    print(mesh_list)
    with open("/home/ubuntu/Data/pflow_data/init_distribution.csv") as f:
        title = f.readline()
        for line in f.readlines():

            print("#############################")
            print(line)
            line = line.strip('\n')
            tokens = line.split(',')
            mesh_id = tokens[0]

            print(mesh_id)

            if mesh_id in mesh_list:

                if os.path.exists("/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/"):
                    print("/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/")
                    id_traj = load.load_trajectory(
                        "/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/train_irl.csv")

                    if len(id_traj.values()) < 10:
                        continue

                    trajectories = id_traj.values()

                    # 20% of samples by divide 5
                    number = (int(tokens[1]) + int(tokens[2]) + int(tokens[3]) + int(tokens[4])) / 5

                    for i in range(int(number)):
                        path = "/home/ubuntu/Data/PT_Result/markov_chain/" + mesh_id + str(i) + ".csv"

                        pairs = makepairs(trajectories)

                        prob = cfd(pairs)

                        print(prob)
                        generate(prob, path, mesh_id)


if __name__ == '__main__':
    main("student")

