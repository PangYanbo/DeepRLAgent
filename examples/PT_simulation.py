from irl.mdp import load, tools
import irl.mdp.gridworld as gridworld
import sys
import os
import datetime
import numpy
import random
import pandas
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

            initial = []

            for traj in trajectories:
                if 12 in traj.keys():
                    initial.append(traj[12][1].get_origin())

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

            for i in range(10):
                start = random.choice(initial)
                tools.generate_temporal_traj(g, r, start, 0.5, path + "sim/", str(i) + filename[0:2])


def main(target):
    try:
        starttime = datetime.datetime.now()

        path = "D:/PT_Result/" + target + "/"

        simulation(path)

        endtime = datetime.datetime.now()

        print endtime - starttime

    except Exception:
        print "main class wrong"
        raise

if __name__ == '__main__':
    main("commuter")
