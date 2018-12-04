import datetime
import os
import random
import sys

import irl.maxent as maxent
import irl.mdp.gridworld as gridworld
from utils import load

sys.path.append("D:/Ubicomp/Inverse-Reinforcement-Learning-master")


def main(discount, epochs, learning_rate):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    discount: MDP discount factor. float.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    try:
        starttime = datetime.datetime.now()
        path = "/home/ubuntu/Data/KDDI/#201111.CDR-data/vks2564k/slot/"

        id_traj = load.load_directory_trajectory(path)

        print(len(id_traj))

        trajectories = id_traj.values()
        g = load.load_graph_traj(trajectories)
        g.set_start("53397561")
        gw = gridworld.Gridworld(g, discount)
        feature_matrix = gw.feature_matrix(g)

        if not os.path.exists(path + "param/"):
            os.mkdir(path + "param/")

        maxent.t_irl(g, feature_matrix, trajectories, epochs, learning_rate, path+"param/")

        endtime = datetime.datetime.now()

        print("finished reading files with time of" + str(endtime - starttime))
    except Exception:
        print("mian class wrong")
        raise


if __name__ == '__main__':
    main(0.9, 400, 3)
