import sys
sys.path.append("D://Ubicomp//Inverse-Reinforcement-Learning-master")
from irl.mdp import load
from utils import tools, load

import os
import matplotlib.pyplot as plt


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


def traj_visual(trajectories):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for uid in trajectories.keys():

        X = []
        Y = []
        Z = []

        trajectory = trajectories[uid]
        print trajectory
        for i in range(12, 47):
            if i in trajectory.keys():
                x, y = tools.parse_MeshCode(trajectory[i][0])
                X.append(x)
                Y.append(y)
                Z.append(i - 12)
        ax.plot(X, Y, Z)

    plt.show()
    # ax.set_zlabel('Hour')
    # ax.set_ylabel('Lat')
    # ax.set_xlabel('Lon')
    # plt.title("3D illustration of trips over urban area")
    # plt.show()


def main():

    path = "D:/training data/KDDI/#201111.CDR-data/abf7380g/"

    train_traj = load.load_directory_trajectory(path + "slot/")

    traj_visual(train_traj)

if __name__ == '__main__':
    main()
