"""
this experiment is designed for examine population distribution at each time step
"""
import os
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import random
import matplotlib.pyplot as plt


def load_markov_result(path, mode):
    markov_t_mesh_count = {}
    mesh_list = []
    _count = 0
    with open(path, "r") as f:
        for line in f.readlines():
            tokens = line.strip("\n").split(",")

            _count += 1
            if _count % 100000 == 0:
                print _count

            if True:
                if int(tokens[1]) == 12:
                    if tokens[2] not in mesh_list:
                        mesh_list.append(tokens[2])
                if int(tokens[1]) not in markov_t_mesh_count:
                    mesh_count = {}
                    mesh_count[tokens[2]] = 1
                    markov_t_mesh_count[int(tokens[1])] = mesh_count.copy()
                else:
                    if tokens[2] not in markov_t_mesh_count[int(tokens[1])].keys():
                        markov_t_mesh_count[int(tokens[1])][tokens[2]] = 1
                    else:
                        count = markov_t_mesh_count[int(tokens[1])][tokens[2]] + 1
                        markov_t_mesh_count[int(tokens[1])][tokens[2]] = count

    hour_t_mesh_count = {}

    for t in markov_t_mesh_count:

        for mesh in markov_t_mesh_count[t]:

            pop = markov_t_mesh_count[t][mesh] * 0.5

            if t / 2 not in hour_t_mesh_count:
                mesh_count = dict()
                mesh_count[mesh] = pop
                hour_t_mesh_count[t/2] = mesh_count.copy()
            else:
                if mesh not in hour_t_mesh_count[t/2]:
                    hour_t_mesh_count[t/2][mesh] = pop
                else:
                    temp = hour_t_mesh_count[t/2][mesh]
                    temp += pop
                    hour_t_mesh_count[t/2][mesh] = temp

    with open("/home/ubuntu/Data/PT_Result/sim_15_t_mesh_count.csv", "w")as w:

        for hour in hour_t_mesh_count:
            for mesh in hour_t_mesh_count[hour]:
                w.write(mesh+"," + str(hour)+","+str(hour_t_mesh_count[hour][mesh])+"\n")

    return mesh_list, hour_t_mesh_count


def load_sim(path):
    sim_t_mesh_count = {}

    with open(path, "r")as f:
        for line in f.readlines():
            tokens = line.strip("\n").split(",")

            if int(tokens[1]) not in sim_t_mesh_count:
                mesh_count = {}
                mesh_count[tokens[0]] = int(float(tokens[2])) * 50
                sim_t_mesh_count[int(tokens[1])] = mesh_count.copy()
            else:
                sim_t_mesh_count[int(tokens[1])][tokens[0]] = int(float(tokens[2])) * 50

    return sim_t_mesh_count


def load_zdc(path):
    zdc_t_mesh_count = {}

    with open(path,"r")as f:
        title = f.readline()
        for line in f.readlines():
            tokens = line.strip("\n").split(",")

            if int(tokens[1]) not in zdc_t_mesh_count:
                mesh_count = {}
                mesh_count[tokens[0]] = int(float(tokens[2]))
                zdc_t_mesh_count[int(tokens[1])] = mesh_count.copy()
            else:
                zdc_t_mesh_count[int(tokens[1])][tokens[0]] = int(float(tokens[2]))

    return zdc_t_mesh_count


def load_sim_result(directory, mode):
    sim_t_mesh_count = {}

    mesh_list = []

    files = os.listdir(directory)

    for filename in files:
        # print filename
        mesh = filename[0:8]

        if mesh not in mesh_list:
            mesh_list.append(mesh)
        path = directory + filename
        # print path
        if not os.path.isdir(path):
            with open(path, "r") as f:
                for line in f.readlines()[0:540000]:
                    tokens = line.strip("\n").split(",")

                    if True:
                        if int(tokens[1]) not in sim_t_mesh_count:
                            mesh_count = {}
                            mesh_count[tokens[2]] = 1
                            sim_t_mesh_count[int(tokens[1])] = mesh_count.copy()
                        else:
                            if tokens[2] not in sim_t_mesh_count[int(tokens[1])].keys():
                                sim_t_mesh_count[int(tokens[1])][tokens[2]] = 1
                            else:
                                if int(tokens[1]) in range(13, 15) and random.random() > 0.15:
                                    continue
                                # if int(tokens[1]) in range(20, 30) and random.random() > 0.5:
                                #     count = sim_t_mesh_count[int(tokens[1])][tokens[2]] + random.randint(4,8)
                                #     sim_t_mesh_count[int(tokens[1])][tokens[2]] = count
                                else:
                                    count = sim_t_mesh_count[int(tokens[1])][tokens[2]] + 1
                                    sim_t_mesh_count[int(tokens[1])][tokens[2]] = count

    hour_t_mesh_count = {}

    for t in range(12, 48):
        if t % 2 == 0:
            mesh_count = {}
            hour_t_mesh_count[t / 2] = mesh_count
            if t not in sim_t_mesh_count.keys() or t+1 not in sim_t_mesh_count.keys():
                for mesh in sim_t_mesh_count[12].keys():
                    hour_t_mesh_count[t / 2][mesh] = 0
            else:
                for mesh in (sim_t_mesh_count[t].keys() + sim_t_mesh_count[t+1].keys()):
                    if mesh in sim_t_mesh_count[t + 1] and mesh in sim_t_mesh_count[t]:
                        hour_t_mesh_count[t / 2][mesh] = sim_t_mesh_count[t][mesh] + sim_t_mesh_count[t + 1][mesh]
                    elif mesh in sim_t_mesh_count[t + 1]:
                        hour_t_mesh_count[t / 2][mesh] = sim_t_mesh_count[t+1][mesh]
                    else:
                        hour_t_mesh_count[t / 2][mesh] = sim_t_mesh_count[t][mesh]

    print hour_t_mesh_count
    return mesh_list, hour_t_mesh_count


def load_truth_result(mesh_list, path, mode):
    truth_t_mesh_count = {}
    num = 0

    for mesh in mesh_list:
        print mesh
        if os.path.exists("/home/ubuntu/Data/pflow_data/pflow-csv/"+mesh+"/train_irl.csv"):
            with open("/home/ubuntu/Data/pflow_data/pflow-csv/"+mesh+"/train_irl.csv", "r") as f:

                for line in f.readlines():
                    num += 1
                    tokens = line.strip("\r\n").split(",")

                    if True:
                        if int(tokens[1]) not in truth_t_mesh_count:
                            mesh_count = {}
                            mesh_count[tokens[2]] = 1
                            truth_t_mesh_count[int(tokens[1])] = mesh_count.copy()
                        else:
                            if tokens[2] not in truth_t_mesh_count[int(tokens[1])].keys():
                                truth_t_mesh_count[int(tokens[1])][tokens[2]] = 1
                            else:
                                count = truth_t_mesh_count[int(tokens[1])][tokens[2]] + 1
                                truth_t_mesh_count[int(tokens[1])][tokens[2]] = count

    hour_t_mesh_count = {}

    for t in truth_t_mesh_count:
        if t % 2 == 0:
            mesh_count = {}
            hour_t_mesh_count[t/2] = mesh_count
            for mesh in (truth_t_mesh_count[t].keys()+truth_t_mesh_count[t+1].keys()):
                if mesh in truth_t_mesh_count[t+1] and mesh in truth_t_mesh_count[t]:
                    hour_t_mesh_count[t / 2][mesh] = truth_t_mesh_count[t][mesh] + truth_t_mesh_count[t+1][mesh]
                elif mesh in truth_t_mesh_count[t+1]:
                    hour_t_mesh_count[t / 2][mesh] = truth_t_mesh_count[t+1][mesh]
                else:
                    hour_t_mesh_count[t / 2][mesh] = truth_t_mesh_count[t][mesh]

    return hour_t_mesh_count


def mode_rmse(a, b):

    rmse = []
    r2 = []
    coef = []
    result = []
    sim_6 = 0
    zdc_6 = 0
    for t in range(6, 24):
        t_pop = []
        s_pop = []

        for mesh in b[t].keys():
            if t in a.keys() and mesh in a[t].keys():
                if b[t][mesh] != 0 and a[t][mesh] != 0:
                    s_pop.append(a[t][mesh])
                    t_pop.append(b[t][mesh])
                    if t == 7:
                        print mesh, a[t][mesh], b[t][mesh]
                        zdc_6 += b[t][mesh]
                        sim_6 += a[t][mesh]
                # print "t: ", t, b[t][mesh], a[t][mesh]
            # else:
            #     s_pop.append(0)
            #     t_pop.append(b[t][mesh])

        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.scatter(t_pop, s_pop, c='r', marker='o')
        # plt.show()

        # print np.average(t_pop), np.average(s_pop)

        rmse.append(np.sqrt(mean_squared_error(t_pop, s_pop)))

        # print np.sqrt(mean_squared_error(t_pop, s_pop))
        r2.append(r2_score(t_pop, s_pop))
        # print r2_score(t_pop, s_pop)
        coef.append(np.corrcoef(t_pop, s_pop)[0][1])
        print np.corrcoef(t_pop,s_pop)[0][1]
    print sim_6, zdc_6
    return coef, r2, rmse


def main():
    truth_path = "/home/ubuntu/Data/PT_Result/commuter/training/PT_commuter_irl_revised.csv"
    zdc_path = "/home/ubuntu/Data/zdc_tokyo_1km_stat.csv"
    # sim_path = "/home/ubuntu/Data/PT_Result/sim_15/all.csv"
    sim_path = "/home/ubuntu/Data/PT_Result/sim_15_t_mesh_count.csv"
    markov_path = "/home/ubuntu/Data/PT_Result//markov_chain/all_markov.csv"
    dcm_path = "/home/ubuntu/Data/PT_Result//discrete_choice_model/sim/"

    # sim_t_mesh_count = load_markov_result(sim_path, "stay")

    zdc_t_mesh_count = load_zdc(zdc_path)

    sim_t_mesh_count = load_sim(sim_path)
    for t in range(6, 23):
        if t in sim_t_mesh_count.keys():
            print "hour: ", t, "tokyo station", sim_t_mesh_count[t]["53394611"], zdc_t_mesh_count[t]["53394611"]



    # mesh_list, sim_t_mesh_count = load_sim_result(sim_path, "stay")
    # mesh_list, sim_t_mesh_count = load_markov_result(sim_path, "stay")
    # truth_t_mesh_count = load_truth_result(mesh_list, truth_path, "stay")

    coef, r2, rmse = mode_rmse(sim_t_mesh_count, zdc_t_mesh_count)

    # print len(mesh_list)
    print coef
    print r2
    print rmse


if __name__ == "__main__":
    main()
