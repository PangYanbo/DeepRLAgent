import datetime
import os
import random
import sys

import irl.maxent as maxent
import irl.mdp.gridworld as gridworld
from utils import load

# sys.path.append("D:/Ubicomp/Inverse-Reinforcement-Learning-master")
sys.path.append("/home/ubuntu/PycharmProjects/RLAgent")


def main(discount, epochs, learning_rate, target):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    discount: MDP discount factor. float.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    try:
        starttime = datetime.datetime.now()
        path = "/home/ubuntu/Data/PT_Result/" + target + "/"

        if not os.path.exists(path + "sim/"):
            os.mkdir(path + "sim/")

        if not os.path.exists(path + "param/"):
            os.mkdir(path + "param/")

        if os.path.exists(path + "training/"):
            id_traj = load.load_directory_trajectory("/home/ubuntu/Data/PT_Result/student/training/")

            # parameter set numbers
            for i in range(10000):
                trajectories = random.sample(id_traj.values(), 500)
                print trajectories
                g = load.load_graph_traj(trajectories)
                gw = gridworld.Gridworld(g, discount)
                feature_matrix = gw.feature_matrix(g)

                # train#
                print ("training ", path)
                maxent.t_irl(g, feature_matrix, trajectories, epochs, learning_rate, path+"param/" + str(i))

        endtime = datetime.datetime.now()

        print ("finished reading files with time of" + str(endtime - starttime))
    except Exception:
        print ("mian class wrong")
        raise


if __name__ == '__main__':
    main(0.9, 200, 0.2, "student")
