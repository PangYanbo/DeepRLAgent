import os
import random
import sys
sys.path.append("/home/ubuntu/PycharmProjects/RLAgent")
import datetime
from utils import load
import numpy as np
from fractions import Fraction
from math import radians, cos, sin, asin, sqrt
import irl.mdp.graph as graph


def read_list(path):
    mesh_list = []
    with open(path, "r")as f:
        for line in f.readlines():
            tokens = line.strip("\r\n").split(",")
            mesh = tokens[2]
            mesh_list.append(mesh)
    return mesh_list


def pop_feature():
    mesh_info = {}
    with open('/home/ubuntu/Data/Tokyo/MESHPOP/MeshPop.csv', 'r') as f:
        for line in f:
            line = line.strip('\n')
            tokens = line.split(",")
            mesh_info[tokens[0]] = float(tokens[2])

    return mesh_info


def get_business():
    mesh_info = {}
    with open('/home/ubuntu/Data/Tokyo/OFFICECOUNTPOP/OFFICECOUNTPOP.csv', 'r') as f:
        for line in f:
            line = line.strip('\n')
            tokens = line.split(",")
            mesh_info[tokens[0]] = (float(tokens[1]), float(tokens[2]))

    return mesh_info


def parse_MeshCode(mesh_code):
    """
    convert mesh no to coordinate
    :return: lat and lon
    """

    LAT_HEIGHT_MESH1 = Fraction(2, 3)
    LNG_WIDTH_MESH1 = Fraction(1, 1)
    LAT_HEIGHT_MESH2 = Fraction(1, 12)
    LNG_WIDTH_MESH2 = Fraction(1, 8)
    LAT_HEIGHT_MESH3 = Fraction(1, 120)
    LNG_WIDTH_MESH3 = Fraction(1, 80)
    LAT_HEIGHT_MESH4 = Fraction(1, 240)
    LNG_WIDTH_MESH4 = Fraction(1, 160)
    LAT_HEIGHT_MESH5 = Fraction(1, 480)
    LNG_WIDTH_MESH5 = Fraction(1, 320)
    strlen = len(mesh_code)

    if strlen == 0 or strlen > 11:
        return None
    x = 0.000000001
    y = 0.000000001
    if strlen >= 4:
        y += float(LAT_HEIGHT_MESH1 * int(mesh_code[0: 2]))
        x += 100 + int(mesh_code[2: 4])

    if strlen >= 6:
        y += float(LAT_HEIGHT_MESH2 * int(mesh_code[4: 5]))
        x += float(LNG_WIDTH_MESH2 * int(mesh_code[5: 6]))

    if strlen >= 8:
        y += float(LAT_HEIGHT_MESH3 * int(mesh_code[6: 7]))
        x += float(LNG_WIDTH_MESH3 * int(mesh_code[7: 8]))

    if strlen >= 9:
        n = int(mesh_code[8: 9])
        y += float(LAT_HEIGHT_MESH4 * (0 if n <= 2 else 1))
        x += float(LNG_WIDTH_MESH4 * (0 if n % 2 == 1 else 1))

    if strlen >= 10:
        n = int(mesh_code[9: 10])
        y += float(LAT_HEIGHT_MESH5 * (0 if n <= 2 else 1))
        x += float(LNG_WIDTH_MESH5 * (0 if n % 2 == 1 else 1))

    return x, y


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return float((c * r * 1000 + random.randint(500, 1000))) / 1000.0


def dist(x, y):
    dest_lon, dest_lat = parse_MeshCode(y)
    ori_lon, ori_lat = parse_MeshCode(x)
    distance = 1.0 / haversine(dest_lon, dest_lat, ori_lon, ori_lat)
    return distance


def choose_action(start, state, time, g, x, pop, office):
    """

    :param start:
    :param state:
    :param time:
    :param g:
    :param x:
    :param pop:
    :param office:
    :return: activity type, edge
    """

    walk = np.exp(x[0])
    vehicle = np.exp(x[1])
    train = np.exp(x[2])

    logsum_mode = np.log(walk + vehicle + train)

    dest_utility = {}

    deno_dest = 0

    for edge in g.get_node(state).get_edges():

        dest = edge.get_destination()
        distance = dist(state, dest)
        if dest in pop.keys() and dest != start:
            dest_utility[dest] = x[3] * logsum_mode + office[dest][1] * x[5] + pop[dest] * x[6] + distance * x[7] + x[8]
        else:
            continue

        deno_dest += np.exp(dest_utility[dest])

    logsum_commute_dest = np.log(deno_dest)
    logsum_other_dest = np.log(deno_dest)

    period_0_9 = 1 if time < 18 else 0
    period_9_17 = 1 if time >= 18 and time < 34 else 0
    period_17_24 = 1 if time >= 34 else 0

    commute = np.exp(
        x[9] * logsum_commute_dest + x[11] * period_0_9 + x[12] * period_9_17 + x[13] * period_17_24 + x[14] + x[15])
    other = np.exp(
        x[10] * logsum_other_dest + x[10] * period_0_9 + x[11] * period_9_17 + x[12] * period_17_24 + x[14] + x[15])
    home = np.exp(x[4] * logsum_mode + x[11] * period_0_9 + x[12] * period_9_17 + x[13] * period_17_24 + x[14] + x[15]) if state!=start else 0
    # stop = np.exp(x[11] * period_0_9 + x[12] * period_9_17 + x[13] * period_17_24 + x[14] + x[15])
    stop = 0

    # print "home", start, state, home, commute, other

    deno = commute + other + home + stop

    activity_prob = {}

    activity_prob["commute"] = commute
    activity_prob["home"] = home
    activity_prob["other"] = other
    activity_prob["stop"] = stop

    activity = max(activity_prob.items(), key=lambda x: x[1])[0]
    # print activity_prob
    # print activity

    action_prob = dict()

    # stop
    e_stop = graph.Edge(state, state, "stay")
    action_prob[e_stop] = stop / deno

    # home
    e_home_walk = graph.Edge(state, start, "walk")
    action_prob[e_home_walk] = (home / deno) * (walk / walk+vehicle+train)

    e_home_vehicle = graph.Edge(state, start, "vehicle")
    action_prob[e_home_vehicle] = (home / deno) * (vehicle / walk + vehicle + train)

    e_home_train = graph.Edge(state, start, "train")
    action_prob[e_home_train] = (home / deno) * (train / walk + vehicle + train)

    # commute and other
    for edge in g.get_node(state).get_edges():
        if edge not in action_prob.keys() and edge.get_destination() in dest_utility.keys() and edge.get_destination() != start:

            if edge.get_mode() == "walk":
                action_prob[edge] = (commute + other) * (np.exp(dest_utility[edge.get_destination()]) / np.exp(logsum_commute_dest)) * (walk / (walk + vehicle + train)) / 8.0
            elif edge.get_mode() == "vehicle":
                action_prob[edge] = (commute + other) * (np.exp(dest_utility[edge.get_destination()]) / np.exp(logsum_commute_dest)) * (vehicle / (walk + vehicle + train)) / 8.0
            elif edge.get_mode() == "train":
                action_prob[edge] = (commute + other) * (np.exp(dest_utility[edge.get_destination()]) / np.exp(logsum_commute_dest)) * (train / (walk + vehicle + train)) / 8.0

    # print max(action_prob.items(), key=lambda x:x[1])[0]
    return max(activity_prob.items(), key=lambda x:x[1])[0], max(action_prob.items(), key=lambda x:x[1])[0]


def policy(g, param, state, start, t, pop, office):
    policy = {}

    activity, action_prob = choose_action(start, state, t, g, param, pop, office)
    policy[t] = {}
    policy[t] = action_prob

    return policy


def generate_trajectory(trajectories, path, start, agent_id):
    files = os.listdir(path + "param/")

    parampath = path + "param/" + random.choice(files)
    if not os.path.isdir(parampath):

        # trajectories = random.sample(trajectories, 50)
        try:
            g = load.load_graph_traj(trajectories)

            pop = pop_feature()
            office = get_business()

            param = np.zeros(16)
            with open(parampath, 'r') as f:
                for line in f:
                    line = line.strip('\n')
                    tokens = line.split(",")
                    param = np.zeros(16)
                    for j in range(16):
                        if len(tokens) > j:
                            param[j] = tokens[j]
                print param

            current_state = start

            with open(path + "sim/" + agent_id + "_dcm.csv", "w")as f:
                t = 12
                while t in range(12, 48):
                    print "********************************************"
                    activity, action = choose_action(start, current_state, t, g, param, pop, office)
                    if t in range(12, 24):
                        departure_time = random.randint(12, 18)

                        print t, action, start, action.get_destination() == start

                        for _t in range(12, departure_time):
                            f.write(agent_id + "," + str(_t) + "," + start + "," + start + "," + "stay" + "," + "stay" + "\n")

                        f.write(agent_id+","+str(departure_time)+","+action.get_origin()+","+action.get_destination()+","+action.get_mode()+"," + activity+"\n")

                        for _t in range(departure_time+1, 32):
                            f.write(agent_id + "," + str(_t) + "," + action.get_destination() + "," + action.get_destination() + "," + "stay" + "," + "stay" + "\n")

                        t = 32
                        current_state = action.get_destination()
                        continue

                    if t in range(32, 48):
                        departure_time = random.randint(32, 48)
                        activity, action = choose_action(start, current_state, t, g, param, pop, office)

                        print t, action, start, action.get_destination() == start

                        for _t in range(32, departure_time):
                            f.write(agent_id + "," + str(_t) + "," + start + "," + start + "," + "stay" + "," + "stay" + "\n")

                        f.write(agent_id + "," + str(
                            departure_time) + "," + action.get_origin() + "," + action.get_destination() + "," + action.get_mode() + "," + activity + "\n")

                        for _t in range(departure_time+1, 48):
                            f.write(agent_id + "," + str(_t) + "," + action.get_destination() + "," + action.get_destination() + "," + "stay" + "," + "stay" + "\n")

                        t = 48
                        current_state = action.get_destination()
                        continue

                    # if t in range(35, 48):
                    #     departure_time = random.randint(35, 48)
                    #     activity, action = choose_action(start, current_state, t, g, param, pop, office)
                    #
                    #     print t, action, start, action.get_destination() == start
                    #
                    #     for _t in range(35, departure_time):
                    #         f.write(agent_id + "," + str(
                    #             _t) + "," + current_state + "," + current_state + "," + "stay" + "," + "stay" + "\n")
                    #
                    #     f.write(agent_id + "," + str(
                    #         departure_time) + "," + action.get_origin() + "," + action.get_destination() + "," + action.get_mode() + "," + activity + "\n")
                    #
                    #     for _t in range(departure_time + 1, 48):
                    #         f.write(agent_id + "," + str(
                    #             _t) + "," + action.get_destination() + "," + action.get_destination() + "," + "stay" + "," + "stay" + "\n")

                        t = 48
                        current_state = action.get_destination()
                        continue

        except KeyError:
            return 0


def main():
    try:
        starttime = datetime.datetime.now()

        mesh_list = read_list("/home/ubuntu/Data/Tokyo/MeshCode/Tokyo.csv")

        print mesh_list
        with open("/home/ubuntu/Data/pflow_data/init_distribution.csv") as f:
            title = f.readline()
            for line in f.readlines():

                print "#############################"
                print line
                line = line.strip('\n')
                tokens = line.split(',')
                mesh_id = tokens[0]

                if mesh_id in mesh_list:

                    if os.path.exists("/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/train_irl.csv"):
                        print("/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/")
                        id_traj = load.load_trajectory(
                            "/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/train_irl.csv")

                        if len(id_traj.values()) < 10:
                            continue

                        trajectories = id_traj.values()

                        # 20% of samples by divide 5
                        jobless = (int(tokens[3]) + int(tokens[4]))

                        path = "/home/ubuntu/Data/PT_Result/discrete_choice_model/"

                        for i in range(jobless):
                            generate_trajectory(trajectories, path, mesh_id, mesh_id+"_"+str(i))

        endtime = datetime.datetime.now()

        print endtime - starttime

    except Exception:
        print "main class wrong"
        raise


if __name__ == "__main__":
    main()
