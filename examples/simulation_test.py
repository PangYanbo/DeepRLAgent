import irl.mdp.gridworld as gridworld
from utils import tools, load
import os
import datetime


def simulation(trajectories, path, start, count):

    if os.path.exists(path):

        parampath = path
        try:
            g = load.load_graph_traj(trajectories)
            g.set_start(start)
            gw = gridworld.Gridworld(g, 0.9, "")
            feature_matrix = gw.feature_matrix(g)

            alpha = load.load_param(parampath)
            print(alpha)

            r = dict()
            for t in range(12, 48):
                r[t] = dict().fromkeys(g.get_edges(), 0)

            for t in range(12, 48):
                for edge in g.get_edges():
                    if t in alpha.keys():
                        r[t][edge] = feature_matrix[t][edge].dot(alpha[t])
            print(r)
            for i in range(count):
                print("****************")
                directory = "/home/ubuntu/Data/PT_Result/100expert_1agent/" + start + "/sim/"
                if not os.path.exists(directory):
                    os.mkdir(directory)
                tools.simple_trajectory(g, r, start, "/home/ubuntu/Data/PT_Result/100expert_1agent/" + start +
                                        "/", start + "_" + str(i+50))

        except KeyError:
            return 0


def main(mesh_id):
    try:
        starttime = datetime.datetime.now()

        id_traj = load.load_trajectory("/home/ubuntu/Data/PT_Result/100expert_1agent/"
                                       + mesh_id + "/training_data.csv")
        # print id_traj

        trajectories = id_traj.values()

        sim_path = "/home/ubuntu/Data/PT_Result/100expert_1agent/" + mesh_id + "/" + str(5) + "_" + 'param.csv'
        simulation(trajectories, sim_path, mesh_id, 10)

        endtime = datetime.datetime.now()

        print(endtime - starttime)

    except Exception:
        print("main class wrong")
        raise


if __name__ == '__main__':
    main("53393574")
