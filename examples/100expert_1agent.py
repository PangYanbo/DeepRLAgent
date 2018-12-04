from utils import load, tools, writeout
import irl.maxent as maxent
import irl.mdp.gridworld as gridworld
import os
import datetime
import random
# sys.path.append("D:/Ubicomp/Inverse-Reinforcement-Learning-master")


def main(mesh_id):
    """
    discount: MDP discount factor. float.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    discount = .9
    epochs = 400
    learning_rate = 3

    try:
        starttime = datetime.datetime.now()

        if not os.path.exists("/home/ubuntu/Data/PT_Result/100expert_1agent/" + mesh_id + "/"):
            os.mkdir("/home/ubuntu/Data/PT_Result/100expert_1agent/" + mesh_id + '/')

        if os.path.exists("/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/train_irl.csv"):
            id_traj = load.load_trajectory("/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/train_irl.csv")

            # parameter set numbers
            for i in range(3):
                print(type(list(id_traj.values())))
                trajectories = random.sample(list(id_traj.values()), 100)

                # save out expert data
                writeout.write_trajs(trajectories, "/home/ubuntu/Data/PT_Result/100expert_1agent/"
                                     + mesh_id + "/training_data.csv")

                g = load.load_graph_traj(trajectories)
                g.set_start(mesh_id)
                print(g.get_start())
                gw = gridworld.Gridworld(g, discount)
                feature_matrix = gw.feature_matrix(g)

                # train#

                maxent.t_irl(g, feature_matrix, trajectories, epochs, learning_rate,
                             "/home/ubuntu/Data/PT_Result/100expert_1agent/" + mesh_id + "/" + str(i+3)+"_")

                # alpha = load.load_param("/home/ubuntu/Data/PT_Result/100expert_1agent/" + mesh_id + "/" + str(i) +
                #                         "_" + 'param.csv')

                # r = dict()
                # for t in range(12, 48):
                #     r[t] = dict().fromkeys(g.get_edges(), 0)
                #
                # for t in range(12, 48):
                #     for edge in g.get_edges():
                #         if t in alpha.keys():
                #             r[t][edge] = feature_matrix[t][edge].dot(alpha[t])
                #
                # for j in range(20):
                #     tools.simple_trajectory(g, r, mesh_id, "/home/ubuntu/Data/PT_Result/100expert_1agent/" + mesh_id +
                #                             "/", mesh_id + "_" + str(j))

        endtime = datetime.datetime.now()

        print ("finished reading files with time of" + str(endtime - starttime))
    except Exception:
        print("mian class wrong")
        raise


if __name__ == '__main__':
    main("53393574")
