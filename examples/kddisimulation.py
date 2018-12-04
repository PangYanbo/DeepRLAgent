import os
import sys

import numpy

import irl.mdp.gridworld as gridworld
from utils import tools, load

sys.path.append("D:/Ubicomp/Inverse-Reinforcement-Learning-master")


def simulation(path):
    id_traj = load.load_directory_trajectory(path + "slot/")

    files = os.listdir(path+"param/")

    if not os.path.exists(path + "sim/"):
        os.mkdir(path + "sim/")

    for filename in files:
        parampath = path + "param/" + filename
        if not os.path.isdir(parampath):

            trajectories = id_traj.values()
            g = load.load_graph_traj(trajectories)
            gw = gridworld.Gridworld(g, 0.9)
            feature_matrix = gw.feature_matrix(g)

            alpha = load.load_param(parampath)
            print(alpha)

            r = dict()
            for t in range(12, 48):
                r[t] = dict().fromkeys(g.get_edges(), 0)

            for t in range(12, 48):
                for edge in g.get_edges():
                    if t in alpha.keys():
                        r[t][edge] = feature_matrix[t][edge].dot(alpha[t])
            print(r)

            for i in range(10):
                print("****************")
                directory = "/home/ubuntu/Data/KDDI/#201111.CDR-data/abf7380g/sim/"
                if not os.path.exists(directory):
                    os.mkdir(directory)
                tools.simple_trajectory(g, r, "53397561", "/home/ubuntu/Data/KDDI/#201111.CDR-data/abf7380g/sim/", "53397561" +
                                        "_" + str(i))

            start = "53397561"
            tools.generate_temporal_traj(g, r, start, 0.5, path + "sim/", filename[0:2])


def main(date, discount, epochs, learning_rate, train=True):
    try:

        simulation("/home/ubuntu/Data/KDDI/#201111.CDR-data/abf7380g/")

    except Exception:
        print("main class wrong")
        raise


if __name__ == '__main__':
    main("2", 0.9, 100, 0.3, train=False)