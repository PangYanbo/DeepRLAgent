"""
Run maximum entropy inverse reinforcement learning on the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""
import datetime
import os
import random
import sys

import numpy

import irl.maxent as maxent
import irl.mdp.gridworld as gridworld
from irl.mdp import load, graph
from utils import tools, load

sys.path.append("D:/Ubicomp/Inverse-Reinforcement-Learning-master")
# sys.path.append("/home/t-iho/Ubicomp/Inverse-Reinforcement-Learning-master")




def training(pos):

    id_trajectory = load.load_trajectory(1000)

    graph_trajectories = tools.choose_trajectory(1000, id_trajectory)

    _graph = load.load_graph_traj(graph_trajectories)

    sample_trajectories = tools.choose_trajectory(100, id_trajectory)

    gw = gridworld.Gridworld(_graph, 0.9)

    feature_matrix = gw.feature_matrix(_graph)

    alpha = maxent.irl(_graph, feature_matrix, sample_trajectories, 1, 0.05)

    path = str("D:/Ubicomp/alpha"+str(pos)+".txt")
    type(path)
    print path
    numpy.savetxt(path, alpha)

    _graph = graph.Graph([],{},False,False)

    del _graph

    return alpha


def generating(g, id_trajectory, alpha, mesh):
    gw = gridworld.Gridworld(g, 0.9)
    feature_matrix = gw.feature_matrix(g)
    reward = dict()
    for edge in g.get_edges():
        reward[edge] = feature_matrix[edge].dot(alpha)

    mesh_parameter = tools.duration_gaussian(id_trajectory)
    tools.generate_traj(g, reward, mesh, mesh_parameter)


def main(date, discount, epochs, learning_rate, train=True):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    discount: MDP discount factor. float.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    """
    # this part is used for calculate uniform reward parameter

    id_trajectory = load.load_trajectory(10000)

    print tools.motion_model_policy(id_trajectory)

    for i in range(1000):
        graph_trajectories = tools.choose_trajectory(1000, id_trajectory)

        g = load.load_graph_traj(graph_trajectories)

        sample_trajectories = sample(graph_trajectories, 100)

        gw = gridworld.Gridworld(g, 0.9)

        feature_matrix = gw.feature_matrix(g)

        alpha = maxent.irl(g, feature_matrix, sample_trajectories, 40, 0.05)

        path = str("D:/Ubicomp/alpha" + str(i) + ".txt")

        numpy.savetxt(path, alpha)

    """

    """
    this part is usedfor temporal reward parameter training
    """

    try:
        starttime = datetime.datetime.now()
        path = "D:/ClosePFLOW/"

        dirs = os.listdir(path)

        for dirname in dirs:
            directory = path + dirname + "/"
            print directory

            if not os.path.exists(directory+"sim/"):
                os.mkdir(directory+"sim/")

            tools.move_files(directory)

            if os.path.exists(directory+"training/"):
                id_traj = load.load_directory_trajectory(directory + "training/")
                if (len(id_traj) >= 40 and not os.path.exists(directory + "param.csv")) or os.path.getsize(directory + "param.csv") >2038:
                    trajectories = id_traj.values()
                    g = load.load_graph_traj(trajectories)
                    gw = gridworld.Gridworld(g, discount)
                    feature_matrix = gw.feature_matrix(g)

                    # train#
                    print "training ", directory
                    maxent.t_irl(g, feature_matrix, trajectories, epochs, learning_rate, directory)






        indicator = 0
        i = 0

        while indicator <= 5000:
            sample_id = []
            trajectories = []
            for k in range(indicator, indicator+100):
                sample_id.append(id_list[k])

            for sid in sample_id:
                trajectories.append(id_traj.get(sid))

            start_state = []

            for traj in trajectories:
                start_state.append(traj[12][0])

            training_data = "C:/Users/PangYanbo/Desktop/UbiResult/TrainingTrajectoriesGroup_" + str(i) + ".csv"

            with open(training_data, "wb")as f:
                for k in range(100):
                    for j in range(12, 47):
                        if j in trajectories[k].keys():
                            f.write(str(j)+','+trajectories[k][j][1].get_origin()+','+trajectories[k][j][1].get_destination() +
                                    ','+trajectories[k][j][1].get_mode()+'\n')

            # initial environment based on trajectories

            g = load.load_graph_traj(trajectories)
            gw = gridworld.Gridworld(g, discount)
            feature_matrix = gw.feature_matrix(g)

            print g

            if train:

                # training the model

                maxent.t_irl(g, feature_matrix, trajectories, epochs, learning_rate, date)
            else:

                # simulation

                for start in start_state:

                    # read alpha from saved file
                    root = "C:/Users/PangYanbo/Desktop/UbiResult/param/"
                    para_list = list(os.path.join(root, name) for name in os.listdir(root))
                    for filename in para_list:
                        if os.path.isdir(filename):
                            para_list.remove(filename)

                    param_path = random.choice(para_list)

                    agent_id = param_path[43:-4]

                    print agent_id, param_path

                    t_alpha = {}
                    with open(param_path, 'r') as f:
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
                    tools.generate_temporal_traj(g, r, start, 0.5, i, agent_id)

            i += 1
            indicator += 50

        endtime = datetime.datetime.now()

        print "finished reading files with time of" + str(endtime - starttime)
    except Exception:
        print "something wrong"
        raise

if __name__ == '__main__':
    main("2", 0.9, 100, 0.3, train=False)
