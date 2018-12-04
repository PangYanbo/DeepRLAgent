import sys
sys.path.append("/home/ubuntu/PycharmProjects/RLAgent")
from utils import load
import irl.maxent as maxent
import irl.mdp.gridworld as gridworld
import os
import datetime
import random
# sys.path.append("D:/Ubicomp/Inverse-Reinforcement-Learning-master")


def read_list(path, number):
    mesh_list = []
    with open(path, "r")as f:
        for line in f.readlines()[200*(number-1):200*number-1]:
            tokens = line.strip("\r\n").split(",")
            mesh = tokens[2]
            mesh_list.append(mesh)
    return mesh_list


def main(epochs, learning_rate, discount, number):
    """
    discount: MDP discount factor. float.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    try:
        starttime = datetime.datetime.now()

        mesh_list = read_list("/home/ubuntu/Data/Tokyo/MeshCode/Tokyo.csv", number)
        print(len(mesh_list))
        print(mesh_list)

        for mesh_id in mesh_list:
            # if not os.path.exists("/home/ubuntu/Data/PT_Result/commuter/test_sim/" + mesh_id + "/"):
            #     os.mkdir("/home/ubuntu/Data/PT_Result/commuter/test_sim/" + mesh_id + "/")
            #
            # if not os.path.exists("/home/ubuntu/Data/PT_Result/commuter/test_param/" + mesh_id + "/"):
            #     os.mkdir("/home/ubuntu/Data/PT_Result/commuter/test_param/" + mesh_id + "/")

            if os.path.exists("/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/train_irl.csv"):
                id_traj = load.load_trajectory("/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/train_irl.csv")

                # parameter set numbers
                if len(id_traj) > 200:
                # for i in range(len(id_traj)/50):
                    trajectories = random.sample(id_traj.values(), 200)
                    g = load.load_graph_traj(trajectories)
                    g.set_start(mesh_id)
                    print(g.get_start())
                    gw = gridworld.Gridworld(g, discount,"")
                    feature_matrix = gw.feature_matrix(g)

                    # train#

                    maxent.t_irl(g, feature_matrix, trajectories, epochs, learning_rate, "/home/ubuntu/Data/PT_Result/param_15/" + mesh_id + "_" + str(1)+"_")

            fo = open("/home/ubuntu//Data/PT_Result/finished_mesh.csv", "a")
            fo.write(mesh_id+"/n")
            fo.close()

        endtime = datetime.datetime.now()

        print ("finished reading files with time of" + str(endtime - starttime))
    except Exception:
        print("mian class wrong")
        raise


if __name__ == '__main__':
    main(100,3,0.9,1)
