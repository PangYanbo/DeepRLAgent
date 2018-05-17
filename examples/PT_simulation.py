from irl.mdp import load, tools
import irl.mdp.gridworld as gridworld
import sys
import os
import datetime
import numpy
import random

sys.path.append("D:/Ubicomp/Inverse-Reinforcement-Learning-master")


def simulation(trajectories, path, start, count):

    files = os.listdir(path+"param/")

    for filename in files:
        parampath = path + "param/" + filename
        if not os.path.isdir(parampath):

            # trajectories = random.sample(trajectories, 50)
            try:
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

                for i in range(count):
                    # start = random.choice(initial)
                    tools.generate_temporal_traj(g, r, start, 0.5, path + "sim/", str(i) + filename[0:2])

            except KeyError:
                return 0


def main():
    try:
        starttime = datetime.datetime.now()

        id_traj = load.load_directory_trajectory("D:/PT_Result/trajectory/")

        with open("C:/Users/PangYanbo/Desktop/Tokyo/Census5339/2015meshpop.csv") as f:
            title = f.readline()
            for line in f.readlines():

                print "#############################"
                print line
                line = line.strip('\n')
                tokens = line.split(',')
                mesh_id = tokens[0]

                trajectories = []

                for uid in id_traj:
                    if 12 in id_traj[uid].keys():
                        if id_traj[uid][12][0].__eq__(mesh_id):
                            trajectories.append(id_traj[uid])

                if len(trajectories) == 0:
                    continue

                jobless = int(tokens[1])
                workers = int(tokens[2])
                students = int(tokens[3])

                jobless_path = "D:/PT_Result/others/"
                simulation(trajectories, jobless_path, mesh_id, jobless)

                workers_path = "D:/PT_Result/commuter/"
                simulation(trajectories, workers_path, mesh_id, workers)

                students_path = "D:/PT_Result/student/"
                simulation(trajectories, students_path, mesh_id, students)

        endtime = datetime.datetime.now()

        print endtime - starttime

    except Exception:
        print "main class wrong"
        raise

if __name__ == '__main__':
    main()
