import sys
sys.path.append("D://Ubicomp//Inverse-Reinforcement-Learning-master")
from irl.mdp import load
from utils import tools, load

import os
import matplotlib.pyplot as plt
import math
import numpy as np


def load_markovchain(directory):
    id_traj = {}
    count = 0

    files = os.listdir(directory)

    for filename in files:
        path = directory + filename

        with open(path) as f:
            count += 1
            for line in f.readlines():
                try:
                    line = line.strip('\n')
                    tokens = line.split(",")
                    agent_id = filename
                    timeslot = int(tokens[0])
                    start = tokens[1]



                    if agent_id not in id_traj.keys():
                        trajectory = {}
                        id_traj[agent_id] = trajectory
                        id_traj[agent_id][timeslot] = start
                    else:
                        id_traj[agent_id][timeslot] = start
                except(AttributeError, ValueError, IndexError, TypeError):
                    print ("errrrrrrrrrrrrrrrroooooooooooooooooooooooooooorrrrrrrrrrrrrrrrrrrrrrrrrrrr........")
                f.close()

    return id_traj


def traj_visual(trajectory, ax):
    X = []
    Y = []
    Z = []
    for i in range(12, 47):
        if i in trajectory.keys():
            x, y = tools.parse_MeshCode(trajectory[i][0])
            X.append(x)
            Y.append(y)
            Z.append(i - 12)
    ax.plot(X, Y, Z, color="red")


def traj_dist(traj_tuple):
    dist = 0
    traj1 = traj_tuple[0]
    traj2 = traj_tuple[1]

    for i in range(12, 47):
        if i in traj1.keys() and i in traj2.keys():

            dist += tools.calculate_mesh_distance(traj1[i][0], traj2[i][0])

    return dist/35.0


def adjust_traj(trajectory):
    temp_t = 13
    t_trajectory = {}
    t_trajectory[12] = trajectory[12]
    for i in range(12, 47):
        if trajectory[i][1].get_mode() != "stay":
            if i >= temp_t:
                t_trajectory[i] = trajectory[i]
            else:
                t_trajectory[temp_t+1] = trajectory[i]

            if trajectory[i][1].get_mode() == "walk":
                temp_t += int(math.ceil(tools.calculate_mesh_distance(trajectory[i][1].get_origin(),
                                                                      trajectory[i][1].get_destination()) / 5.0 * 2))
            if trajectory[i][1].get_mode() == "vehicle":
                temp_t += int(math.ceil(tools.calculate_mesh_distance(trajectory[i][1].get_origin(),
                                                                      trajectory[i][1].get_destination()) / 40.0 * 2))
            if trajectory[i][1].get_mode() == "train":
                temp_t += int(math.ceil(tools.calculate_mesh_distance(trajectory[i][1].get_origin(),
                                                                      trajectory[i][1].get_destination()) / 60.0 * 2))
    return t_trajectory


def main():

    path_sim = "D:/ClosePFLOW/53393574/sim/"
    path_validation = "D:/ClosePFLOW/53393574/validation/"
    path_observed = "D:/PT_Result/commuter/sim/"

    if not os.path.exists("D:/ClosePFLOW/53393574/comparison/"):
        os.mkdir("D:/ClosePFLOW/53393574/comparison/")

    observed_id_traj = load.load_directory_trajectory(path_observed)
    sim_id_traj = load.load_directory_trajectory(path_sim)
    validation_id_traj = load.load_directory_trajectory(path_validation)

    markov_id_traj = load_markovchain("D:/ClosePFLOW/53393574/markovchain/")
    print len(markov_id_traj)
    # diff_list = []
    # min_id = None
    # trajectories = random.sample(observed_id_traj.values(), 10)
    # for validation_id in validation_id_traj.keys():
    #
    #     min_dist = sys.maxint
    #     dist = traj_dist((random.choice(trajectories), validation_id_traj[validation_id]))
    #     for trajectory in trajectories:
    #         dist = traj_dist((trajectory, validation_id_traj[validation_id]))
    #         if dist < min_dist:
    #             min_dist = dist
    #             min_id = validation_id
    #
    #     diff_list.append(dist)
    #
    #     validation_id_traj.pop(min_id)
    #
    # diff_list.sort()
    #
    # print diff_list
    # print np.average(diff_list)








    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    count = 0
    for uid in observed_id_traj.keys():
        count += 1
        if count > 80:
            break
        X = []
        Y = []
        Z = []

        trajectory = observed_id_traj[uid]
        # print "trajectory", trajectory
        # print "adjusted trajectory", adjust_traj(trajectory)
        for i in range(12, 47):
            if i in trajectory.keys():
                x, y = tools.parse_MeshCode(trajectory[i][0])
                X.append(x)
                Y.append(y)
                Z.append(i-12)
        ax.plot(X, Y, Z)
    ax.set_zlabel('Hour')
    ax.set_ylabel('Lat')
    ax.set_xlabel('Lon')
    plt.show()


    # for uid in validation_id_traj.keys():
    #     X = []
    #     Y = []
    #     Z = []
    #
    #
    #     trajectory = validation_id_traj[uid]
    #     # print "trajectory", trajectory
    #     # print "adjusted trajectory", adjust_traj(trajectory)
    #     for i in range(12, 47):
    #         if i in trajectory.keys():
    #             x, y = tools.parse_MeshCode(trajectory[i][0])
    #             X.append(x)
    #             Y.append(y)
    #             Z.append(i-12)
    #     ax.plot(X, Y, Z)
    #
    # plt.show()
    #
    # for uid in sim_id_traj.keys():
    #     X = []
    #     Y = []
    #     Z = []
    #     trajectory = sim_id_traj[uid]
    #     for i in range(12, 46):
    #         if i in trajectory.keys():
    #             x, y = tools.parse_MeshCode(trajectory[i][0])
    #             X.append(x)
    #             Y.append(y)
    #             Z.append(i-12)
    #     ax.plot(X, Y, Z, color="blue")
    #
    # plt.show()
    diff_list = []
    #
    # for ob_id in observed_id_traj:
    #     fig = plt.figure()
    #
    #     ax = fig.add_subplot(111, projection='3d')
    #
    #     min_dist = sys.maxint
    #     for validation_id in validation_id_traj.keys():
    #         dist = traj_dist((observed_id_traj[ob_id], validation_id_traj[validation_id]))
    #         if dist < min_dist:
    #             min_dist = dist
    #             min_id = validation_id
    #
    #     diff_list.append(min_dist)
    #     if min_id in validation_id_traj.keys():
    #         X = []
    #         Y = []
    #         Z = []
    #         trajectory = validation_id_traj[min_id]
    #         # print "trajectory", trajectory
    #         # print "adjusted trajectory", adjust_traj(trajectory)
    #         for i in range(12, 47):
    #             if i in trajectory.keys():
    #                 x, y = tools.parse_MeshCode(trajectory[i][0])
    #                 print x, y
    #                 X.append(x)
    #                 Y.append(y)
    #                 Z.append(i - 12)
    #         ax.plot(X, Y, Z, color="red")
    #
    #         X = []
    #         Y = []
    #         Z = []
    #         trajectory = observed_id_traj[ob_id]
    #         for i in range(12, 46):
    #             if i in trajectory.keys():
    #                 x, y = tools.parse_MeshCode(trajectory[i][0])
    #                 X.append(x)
    #                 Y.append(y)
    #                 Z.append(i - 12)
    #         ax.plot(X, Y, Z, color="blue")
    #
    #         ax.set_zlabel('time')
    #         ax.set_ylabel('lat')
    #         ax.set_xlabel('lon')
    #         plt.show()
    #
    #         validation_id_traj.pop(min_id)
    #
    # diff_list.sort()
    #
    # print len(diff_list)
    #
    # x = range(len(diff_list))
    # plt.plot(x, diff_list)
    # plt.show()

    diff_list = []

    for sim_id in sim_id_traj:
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')

        min_dist = sys.maxint
        for validation_id in validation_id_traj.keys():
            dist = traj_dist((sim_id_traj[sim_id], validation_id_traj[validation_id]))
            if dist < min_dist:
                min_dist = dist
                min_id = validation_id

        diff_list.append(min_dist)
        print min_id
        if min_id in validation_id_traj.keys():

            X = []
            Y = []
            Z = []
            trajectory = validation_id_traj[min_id]
            # print "trajectory", trajectory
            # print "adjusted trajectory", adjust_traj(trajectory)
            for i in range(12, 47):
                if i in trajectory.keys():
                    x, y = tools.parse_MeshCode(trajectory[i][0])
                    X.append(x)
                    Y.append(y)
                    Z.append(i - 12)
            ax.plot(X, Y, Z, color="red", linewidth=3.5)

            X = []
            Y = []
            Z = []
            trajectory = sim_id_traj[sim_id]
            for i in range(12, 46):
                print trajectory.keys()
                if i in trajectory.keys():
                    x, y = tools.parse_MeshCode(trajectory[i][0])
                    print i,";;;;;;;;;;;;;;;"
                    print trajectory[i]
                    X.append(x)
                    Y.append(y)
                    Z.append(i - 12)
            ax.plot(X, Y, Z, color="blue", linewidth=3.5)
            ax.set_zlabel('time')
            ax.set_ylabel('lat')
            ax.set_xlabel('lon')
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(6)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(6)
            for tick in ax.zaxis.get_major_ticks():
                tick.label1.set_fontsize(6)
            ax.xaxis.set_ticks_position('none')
            plt.xlim((139, 140))
            plt.ylim((35.3, 36))
            plt.title(min_dist)
            plt.savefig("D:/ClosePFLOW/53393574/comparison/traj_compare"+sim_id+"_"+".png")
            plt.show()

            validation_id_traj.pop(min_id)

    diff_list.sort()

    print np.average(diff_list)

    # x = range(len(diff_list))
    # bins = np.arange(0,20,1)
    # plt.hist(diff_list,bins=bins,alpha=0.5)
    # plt.show()

if __name__ == '__main__':
    main()
