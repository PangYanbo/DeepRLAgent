from irl.mdp import load, tools, graph
import irl.maxent as maxent
import irl.mdp.gridworld as gridworld
import sys
import os
import datetime
import numpy
import random
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

            start = "53397561"
            tools.generate_temporal_traj(g, r, start, 0.5, path + "sim/", filename[0:2])


def main(date, discount, epochs, learning_rate, train=True):
    try:

        simulation("D:/training data/KDDI/#201111.CDR-data/abf7380g/")

    except Exception:
        print "main class wrong"
        raise

if __name__ == '__main__':
    main("2", 0.9, 100, 0.3, train=False)