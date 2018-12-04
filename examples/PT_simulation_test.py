import sys
sys.path.append("/home/ubuntu/PycharmProjects/RLAgent")
import irl.mdp.gridworld as gridworld
from irl.mdp import load
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


def simulation(trajectories, path, start, count):

    if os.path.exists(path+"param_15/"):

        files = os.listdir(path+"param_15/")

        parampath = path + "param_15/" + random.choice(files)

        if not os.path.isdir(parampath) and parampath[-4:] == ".csv" and os.path.getsize(parampath)/float(1024) > 6:
            try:

                g = load.load_graph_traj(trajectories)
                g.set_start(start)
                gw = gridworld.Gridworld(g, 0.9)
                feature_matrix = gw.feature_matrix(g)

                t_alpha = {}
                print parampath
                with open(parampath, 'r') as f:
                    t_alpha[12] = numpy.zeros(15)
                    t_alpha[12][4] = 100
                    t = 13
                    for line in f:
                        line = line.strip('\n')
                        tokens = line.split(",")
                        param = numpy.zeros(15)
                        # print len(tokens)
                        # print tokens[14], tokens[13]
                        for j in range(15):
                            if len(tokens) > j and tokens[j] != "":
                                param[j] = float(tokens[j])
                        t_alpha[t] = param.copy()
                        # print "parameter", t, param
                        t += 1

                r = dict()
                for t in range(12, 48):
                    r[t] = dict().fromkeys(g.get_edges(), 0)

                for t in range(12, 48):
                    for edge in g.get_edges():
                        if t in t_alpha.keys():
                            r[t][edge] = feature_matrix[edge].dot(t_alpha[t])
                # print r
                for i in range(count):
                    print "****************"
                    tools.simple_trajectory(g, r, start, "/home/ubuntu/Data/PT_Result/sim_15/", start + "_" + str(i))

            except KeyError:
                return 0


def main(number):
    try:
        starttime = datetime.datetime.now()

        mesh_list = read_list("/home/ubuntu/Data/Tokyo/MeshCode/Tokyo.csv")

        print mesh_list
        with open("/home/ubuntu/Data/Tokyo/Census/National_Census.csv") as f:
            title = f.readline()
            for line in f.readlines()[130*(number-1):130*number-1]:
                line = line.strip('\n')
                tokens = line.split(',')
                mesh_id = tokens[0]
                pop = int(float(tokens[1])) / 50

                if mesh_id in mesh_list:
                    print mesh_id
                    if os.path.exists("/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/train_irl.csv"):
                        # print("/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/")

                        id_traj = load.load_trajectory("/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/train_irl.csv")
                        # print id_traj

                        if len(id_traj.values()) < 10:
                            continue

                        trajectories = id_traj.values()

                        sim_path = "/home/ubuntu/Data/PT_Result/"
                        simulation(trajectories, sim_path, mesh_id, pop)

                    else:
                        log = open("/home/ubuntu/Data/PT_Result/unsim_mesh.csv","a")
                        log.write(mesh_id +"\n")
                        # students_path = "/home/ubuntu/Data/PT_Result/student/"
                        # simulation(trajectories, students_path, mesh_id, students, "students")

        endtime = datetime.datetime.now()

        print endtime - starttime

    except Exception:
        print "main class wrong"
        raise


if __name__ == '__main__':
    main(11)
