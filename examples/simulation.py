import datetime
import os
import random
import sys

import numpy

import irl.mdp.gridworld as gridworld
from irl.mdp import load
from utils import tools, load

sys.path.append("D:/Ubicomp/Inverse-Reinforcement-Learning-master")


def simulation(path):
    id_traj = load.load_directory_trajectory(path + "training/")

    files = os.listdir(path+"param/")

    for filename in files:
        parampath = path + "param/" + filename
        if not os.path.isdir(parampath):

            trajectories = random.sample(id_traj.values(), 50)
            g = load.load_graph_traj(trajectories)
            gw = gridworld.Gridworld(g, 0.9)
            feature_matrix = gw.feature_matrix(g)

            t_alpha = {}
            with open(parampath, 'r') as f:
                t = 12
                for line in f:
                    line = line.strip('\n')
                    tokens = line.split(",")
                    param = numpy.zeros(11)
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

            start = "53393575"
            tools.generate_temporal_traj(g, r, start, 0.5, path + "sim/", filename[0:2])


def main(date, discount, epochs, learning_rate, train=True):
    try:
        starttime = datetime.datetime.now()

        simulation("D:/ClosePFLOW/53393575/")
        # path = "D:/ClosePFLOW/"
        #
        # dirs = os.listdir(path)
        #
        # for dirname in dirs:
        #     directory = path + dirname + "/"
        #     print directory
        #
        #     if os.path.exists(directory+"param.csv"):
        #         id_traj = load.load_directory_trajectory(directory + "training/")
        #         trajectories = id_traj.values()
        #         g = load.load_graph_traj(trajectories)
        #         gw = gridworld.Gridworld(g, discount)
        #         feature_matrix = gw.feature_matrix(g)
        #
        #         t_alpha = {}
        #         with open(directory+"param.csv", 'r') as f:
        #             t = 12
        #             for line in f:
        #                 line = line.strip('\n')
        #                 tokens = line.split(",")
        #                 param = numpy.zeros(11)
        #                 for j in range(11):
        #                     if len(tokens) > j:
        #                         param[j] = tokens[j]
        #                 t_alpha[t] = param.copy()
        #                 t += 1
        #
        #         r = dict()
        #         for t in range(12, 48):
        #             r[t] = dict().fromkeys(g.get_edges(), 0)
        #
        #         for edge in g.get_edges():
        #             for t in range(12, 48):
        #                 if t in t_alpha.keys():
        #                     r[t][edge] = feature_matrix[edge].dot(t_alpha[t])
        #
        #         start = dirname
        #         tools.generate_temporal_traj(g, r, start, 0.5, directory + "sim/", "agent_id")

    except Exception:
        print "main class wrong"
        raise

if __name__ == '__main__':
    main("2", 0.9, 100, 0.3, train=False)