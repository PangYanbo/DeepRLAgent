import sys
sys.path.append("/home/ubuntu/PycharmProjects/DeepRLAgent/")
import datetime
import main.Env.graph as graph
import os
import numpy as np


def load_directory_trajectory(directory):

    id_traj = {}
    count = 0
    number = 0
    files = os.listdir(directory)

    for filename in files:
        # print filename
        path = directory + filename
        if not os.path.isdir(path):
            with open(path) as f:
                count += 1
                if count > 1000:
                    break
                for line in f.readlines():
                        number += 1
                        if number > 360000:
                            break
                        line = line.strip('\r\n')
                        tokens = line.split(",")
                        if len(tokens) == 5 and tokens[0]!="":
                            print(tokens)
                            agent_id = tokens[0]
                            timeslot = int(tokens[1])
                            start = tokens[2]
                            end = tokens[3]
                            mode = tokens[4]
                            e = graph.Edge(start, end, mode)

                            traj = (start, e)

                            if agent_id not in id_traj.keys():
                                trajectory = {}
                                id_traj[agent_id] = trajectory
                                id_traj[agent_id][timeslot] = traj
                            else:
                                id_traj[agent_id][timeslot] = traj

                f.close()

    return id_traj


def load_trajectory(path):

    id_traj = {}
    count = 0

    starttime = datetime.datetime.now()
    # with open("home/t-iho/Result/trainingdata/trainingdata"+date+".csv") as f:
    with open(path) as f:
        for line in f.readlines():
            try:
                count += 1
                if count % 100000 == 0:
                    print("finish " + str(count) + " lines")

                if count > 360000:
                    break

                line = line.strip('\r\n')
                tokens = line.split(",")
                agent_id = tokens[0]
                timeslot = int(tokens[1])
                start = tokens[2]
                end = tokens[3]
                mode = tokens[4]

                e = graph.Edge(start, end, mode)

                traj = (start, e)

                if agent_id not in id_traj.keys():
                    trajectory = {}
                    id_traj[agent_id] = trajectory
                    id_traj[agent_id][timeslot] = traj
                else:
                    id_traj[agent_id][timeslot] = traj

            except(AttributeError, ValueError, IndexError, TypeError):
                print("errrrrrrrrrrrrrrrroooooooooooooooooooooooooooorrrrrrrrrrrrrrrrrrrrrrrrrrrr........")

    f.close()
    endtime = datetime.datetime.now()
    print("finished reading trajectories with time of" + str(endtime - starttime))
    print("id count", len(id_traj))
    return id_traj


def load_graph_traj(trajectories):
    nodes = {}
    edges = {}

    for trajectory in trajectories:
        for slot in trajectory.keys():
            mode = trajectory[slot][1].get_mode()
            origin = trajectory[slot][1].get_origin()
            destination = trajectory[slot][1].get_destination()
            if origin not in nodes:
                nodes[origin] = origin
            if destination not in nodes:
                nodes[destination] = destination
            e = graph.Edge(origin, destination, mode)
            e2 = graph.Edge(destination, origin, mode)
            if e not in edges:
                edges[e] = e
            if e2 not in edges:
                edges[e2] = e2

    g = graph.Graph(nodes.values(), edges.values(), directed=False, normalization=True)

    print("edge number = ", g.get_edge_number())
    print("node number = ", g.get_node_number())

    return g


def load_graph(lines):
    """

    :return:
    """
    nodes = {}
    edges = {}

    count = 0
    with open("D:/training data/PTtraj3.csv") as f:
        for line in f.readlines()[0:lines]:
            count += 1
            if count % 10000 == 0:
                print(count)

            tokens = line.strip('\n').split(",")
            mode = tokens[4]
            origin = tokens[2]
            destination = tokens[3]
            if origin not in nodes:
                nodes[origin] = origin
            if destination not in nodes:
                nodes[destination] = destination
            e = graph.Edge(origin, destination, mode)
            e2 = graph.Edge(destination, origin, mode)
            if e not in edges:
                edges[e] = e
            if e2 not in edges:
                edges[e2] = e2

    f.close()
    try:
        g = graph.Graph(nodes.values(), edges.values(), directed=False)
    except(AttributeError, ValueError, IndexError, TypeError):
        print("errrrrrrrrrrrrrrrroooooooooooooooooooooooooooorrrrrrrrrrrrrrrrrrrrrrrrrrrr........")

    print("edge number = ", g.get_edge_number())
    print("node number = ", g.get_node_number())

    return g


def load_param(path):
    alpha = {}

    with open(path, 'r') as f:
        t = 12
        for line in f:
            line = line.strip('\n')
            tokens = line.split(",")
            param = np.zeros(31)
            for j in range(31):
                if len(tokens) > j and tokens[j] != "":
                    param[j] = float(tokens[j])
            alpha[t] = param.copy()
            # print "parameter", t, param
            t += 1

    return alpha

