import sys
sys.path.append("/home/ubuntu/PycharmProjects/RLAgent")
import irl.mdp.gridworld as gridworld
from utils import tools, load
import os
import datetime
import numpy
import random


def read_list(path):
    mesh_list = []
    with open(path, "r")as f:
        for line in f.readlines():
            tokens = line.strip("\r\n").split(",")
            mesh = tokens[2]
            mesh_list.append(mesh)
    return mesh_list


def simulation(trajectories, path, start, count, job):

    files = os.listdir(path+"param/")

    parampath = path + "param/" + random.choice(files)
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
                tools.simple_trajectory(g, r, start, "/home/ubuntu/Data/PT_Result/exp3", start + "_" + job + "_" + str(i))

        except KeyError:
            return 0


def main():
    try:
        starttime = datetime.datetime.now()

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

                if mesh_id in mesh_list:

                    if os.path.exists("/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/train_irl.csv"):
                        print("/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/")
                        # id_traj = load.load_trajectory("/home/ubuntu/Data/PT_Result/training/PT_commuter_irl_revised.csv")
                        id_traj = load.load_trajectory("/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/train_irl.csv")

                        if len(id_traj.values()) < 10:
                            continue

                        trajectories = id_traj.values()

                        # 20% of samples by divide 5
                        jobless = (int(tokens[3]) + int(tokens[4]))
                        workers = int(tokens[1])
                        students = int(tokens[2])

                        jobless_path = "/home/ubuntu/Data/PT_Result/others/"
                        simulation(trajectories, jobless_path, mesh_id, jobless, "others")

                        workers_path = "/home/ubuntu/Data/PT_Result/commuter/"
                        simulation(trajectories, workers_path, mesh_id, workers, "commuters")

                        students_path = "/home/ubuntu/Data/PT_Result/student/"
                        simulation(trajectories, students_path, mesh_id, students, "students")

        endtime = datetime.datetime.now()

        print(endtime - starttime)

    except Exception:
        print("main class wrong")
        raise


if __name__ == '__main__':
    main()
