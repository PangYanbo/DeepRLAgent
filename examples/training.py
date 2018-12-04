import datetime
import os
import random
import sys

import irl.maxent as maxent
import irl.mdp.gridworld as gridworld
from irl.mdp import load
from utils import tools, load

sys.path.append("D:/Ubicomp/Inverse-Reinforcement-Learning-master")


def main(date, discount, epochs, learning_rate, train=True):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    discount: MDP discount factor. float.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    try:
        starttime = datetime.datetime.now()
        path = "D:/ClosePFLOW/53393575/"

        if not os.path.exists(path + "sim/"):
            os.mkdir(path + "sim/")

        if not os.path.exists(path + "param/"):
            os.mkdir(path + "param/")

        tools.move_files(path)

        if os.path.exists(path + "training/"):
            id_traj = load.load_directory_trajectory(path + "training/")

            # parameter set numbers
            for i in range(26):
                trajectories = random.sample(id_traj.values(), 50)
                g = load.load_graph_traj(trajectories)
                gw = gridworld.Gridworld(g, discount)
                feature_matrix = gw.feature_matrix(g)

                # train#
                print "training ", path
                maxent.t_irl(g, feature_matrix, trajectories, epochs, learning_rate, path+"param/" + str(i))

        endtime = datetime.datetime.now()

        print "finished reading files with time of" + str(endtime - starttime)
    except Exception:
        print "mian class wrong"
        raise

if __name__ == '__main__':
    main("2", 0.9, 100, 0.3, train=False)
