import sys
import os
import datetime
import random
from irl.mdp import load, tools
import irl.maxent as maxent
import irl.mdp.gridworld as gridworld
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
        path = "D:/training data/KDDI/#201111.CDR-data/abf7380g/slot/"

        id_traj = load.load_directory_trajectory(path)

        print len(id_traj)

        trajectories = random.sample(id_traj.values(), 20)
        g = load.load_graph_traj(trajectories)
        gw = gridworld.Gridworld(g, discount)
        feature_matrix = gw.feature_matrix(g)

        # train#
        print("training ", path)

        if not os.path.exists(path + "param/"):
            os.mkdir(path + "param/")

        maxent.t_irl(g, feature_matrix, trajectories, epochs, learning_rate, path+"param/")

        endtime = datetime.datetime.now()

        print("finished reading files with time of" + str(endtime - starttime))
    except Exception:
        print("mian class wrong")
        raise

if __name__ == '__main__':
    main(0.9, 100, 0.3)
