import irl.mdp.gridworld as gridworld
from utils import tools, load
import os
import datetime
import numpy


def load_param(path):
    alpha = {}

    with open(path, 'r') as f:
        alpha[12] = numpy.zeros(31)
        alpha[12][4] = 100
        t = 13
        for line in f:
            line = line.strip('\n')
            tokens = line.split(",")
            param = numpy.zeros(31)
            for j in range(31):
                if len(tokens) > j and tokens[j] != "":
                    param[j] = float(tokens[j])
            alpha[t] = param.copy()
            # print "parameter", t, param
            t += 1

    return alpha


def simulation(trajectories, path, start, count):

    if os.path.exists(path):

        parampath = path + start + "_0_param.csv"
        try:
            g = load.load_graph_traj(trajectories)
            g.set_start(start)
            gw = gridworld.Gridworld(g, 0.9)
            feature_matrix = gw.feature_matrix(g)

            alpha = load_param(parampath)

            r = dict()
            for t in range(12, 48):
                r[t] = dict().fromkeys(g.get_edges(), 0)

            for t in range(12, 48):
                for edge in g.get_edges():
                    if t in alpha.keys():
                        r[t][edge] = feature_matrix[edge].dot(alpha[t])
            # print r
            for i in range(count):
                print("****************")
                directory = "/home/t-iho/Result/sim/" + start
                if not os.path.exists(directory):
                    os.mkdir(directory)
                tools.simple_trajectory(g, r, start, "/home/t-iho/Result/sim/" + start + "/", start + "_" + str(i))

        except KeyError:
            return 0


def main(mesh_id):
    try:
        starttime = datetime.datetime.now()

        id_traj = load.load_directory_trajectory("/home/t-iho/Result/trainingdata/20180122/" + mesh_id + "/")
        # print id_traj

        trajectories = id_traj.values()

        sim_path = "/home/t-iho/Result/RewardParameter/20180122/"
        simulation(trajectories, sim_path, mesh_id, 20)

        endtime = datetime.datetime.now()

        print(endtime - starttime)

    except Exception:
        print("main class wrong")
        raise


if __name__ == '__main__':
    main(11)
