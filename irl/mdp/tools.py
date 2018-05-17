import string

import irl.value_iteration
import os
import shutil

from math import radians, cos, sin, asin, sqrt, ceil,pow
from fractions import Fraction

import random
import numpy as np
import graph


def motion_model_policy(id_trajectory):
    policy = dict()
    state_freq = dict()
    for uid in id_trajectory:
        for t in id_trajectory[uid].keys():
            state = id_trajectory[uid][t][0]
            edge = id_trajectory[uid][t][1]
            if state not in state_freq:
                state_freq[state] = 1
            else:
                state_freq[state] += 1
            if state not in policy:
                policy[state] = dict()
                policy[state][edge] = 1
            else:
                if edge not in policy[state].keys():
                    policy[state][edge] = 1
                else:
                    policy[state][edge] += 1

    for state in state_freq.keys():
        for edge in policy[state].keys():
            policy[state][edge] /= float(state_freq[state])

    print "freq", state_freq
    return policy


def agent_num(id_trajectory):
    mesh_num = dict()
    for uid in id_trajectory:
        mesh = id_trajectory[uid][12][0]
        if mesh not in mesh_num:
            mesh_num[mesh] = 1
        else:
            mesh_num[mesh] += 1
    out = open('/home/t-iho/Result/sim/20170612initialpop.csv')
    for mesh in mesh_num:
        out.write(mesh+','+mesh_num[mesh] + '\n')
    out.close()
    return mesh_num


def generate_temporal_traj(g, reward, start, epsilon, path, agent_id):
    """
    53394519
    :param g:
    :param reward:
    :param start:
    :param epsilon:
    :param path:
    :param agent_id: id of reward parameter
    :return:
    """
    out = open(path + '/' + agent_id + 'Synthetic.csv', 'a')
    print out
    t = 12
    current_state = start
    history = []
    action_sequence = {}
    history.append(current_state)
    policy = irl.value_iteration.find_temporal_policy(g, reward, 0.9, 46, stochastic=True)
    # for i in range(12, 47):
    #     print i
    #     print policy[i]
    #     print "\n"
    while t < 47:
        # policy = irl.value_iteration.find_temporal_policy(g, reward, 0.9, 1e-2, None, True)
        action = random_temporal_weight(policy[t][current_state])

        if action != "stay":
            print
        # greedy method to decide explore or exploit
        # if random > epsilon, choose destination from history

        if t > 34 and action.get_destination() != history[0]:
            if random.random() > 0.5 and action.get_mode() != "stay":
                temp = 0
                while current_state != history[0]:
                    potential_state = [history[0]]
                    for state in potential_state:
                        for edge in g.get_node(state).get_edges():
                            temp += 1
                            if edge.get_destination() == current_state:
                                action = graph.Edge(edge.get_destination(), edge.get_origin(), edge.get_mode())
                                out.write(str(agent_id) + "," + str(t) + "," + current_state + ","
                                          + action.get_destination() + "," + action.get_mode() + "\n")
                                current_state = edge.get_origin()
                                history.append(current_state)
                                action_sequence[t] = action
                                t += 1
                                break

                            else:
                                if edge.get_destination() not in potential_state:
                                    potential_state.append(edge.get_destination())

                        potential_state.remove(state)

                        if action.get_destination() == history[0]:
                            break

                    if action.get_destination() == history[0] or temp > 100:
                        history.append(current_state)
                        action_sequence[t] = action
                        break

            if random.random() > 0.5 and action.get_mode() == "stay":
                temp = 0
                while current_state != history[0]:
                    potential_state = [history[0]]
                    for state in potential_state:
                        for edge in g.get_node(state).get_edges():
                            temp += 1
                            if edge.get_destination() == current_state:
                                print "possion random", int(np.random.poisson(3)), t
                                start_time = t+int(np.random.poisson(3))-1
                                a = t
                                for j in range(a, start_time):
                                    action = graph.Edge(current_state, current_state, "stay")
                                    history.append(current_state)
                                    print str(agent_id) + "," + str(j) + "," + current_state + "," + action.get_destination() + "," + action.get_mode() + "\n"
                                    out.write(str(agent_id) + "," + str(j) + "," + current_state + ","
                                              + current_state + "," + action.get_mode() + "\n")
                                    t += 1
                                action = graph.Edge(edge.get_destination(), edge.get_origin(), edge.get_mode())
                                out.write(str(agent_id) + "," + str(t) + "," + current_state + ","
                                          + action.get_destination() + "," + action.get_mode() + "\n")
                                current_state = edge.get_origin()
                                history.append(current_state)
                                action_sequence[t] = action
                                t += 1
                                break

                            else:
                                if edge.get_destination() not in potential_state:
                                    potential_state.append(edge.get_destination())

                        potential_state.remove(state)

                        if action.get_destination() == history[0]:
                            break

                    if action.get_destination() == history[0] or temp > 100:
                        history.append(current_state)
                        action_sequence[t] = action
                        break

        else:
            if action.get_mode() != "stay" and 34 >= t > 20:
                if action_sequence[t - 2].get_mode() != "stay" or \
                        action_sequence[t - 3].get_mode() != "stay" or action_sequence[t - 4].get_mode() != "stay" \
                        or action_sequence[t - 6].get_mode() != "stay" or action_sequence[t - 7].get_mode() != "stay":
                    if random.random() < 0.8:
                        if random.random() > epsilon:
                            potential = []
                            for key in policy[t][current_state]:
                                destination = key.get_destination()
                                if destination in history and destination != history[0]:
                                    potential.append(key)
                            if len(potential) > 0:
                                action = random.choice(potential)
                    else:
                        action = graph.Edge(current_state, current_state, "stay")
                else:
                    if random.random() > epsilon:
                        potential = []
                        for key in policy[t][current_state]:
                            destination = key.get_destination()
                            if destination in history and key.get_mode() != "stay":
                                potential.append(key)
                        if len(potential) > 0:
                            action = random.choice(potential)
            # avoid agent back to home at morning
            # else:
            #     if action.get_mode() != "stay" and t < 20:
            #         if action.get_destination() == history[0]:
            #             action = graph.Edge(current_state, current_state, "stay")
            #         else:
            #             if random.random() > 0.3:
            #                 action = graph.Edge(current_state, current_state, "stay")

            out.write(str(agent_id) + "," + str(t) + "," + current_state + "," + action.get_destination() + "," +
                      action.get_mode() + "\n")
            current_state = action.get_destination()
        history.append(current_state)
        action_sequence[t] = action

        if t > 34 and current_state == history[0] and random.random() > 0.7:
            for j in range(t, 47):
                out.write(str(agent_id) + "," + str(j) + "," + current_state + ","
                          + current_state + "," + "stay" + "\n")
            break
        t += 1
    out.close()


def generate_traj(graph, reward, start_state, mesh_parameter):
    salt = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    out = open('/home/t-iho/Result/IRLSynthetic/20170706synthetic.csv', "a")

    # v = irl.value_iteration.optimal_value(graph, reward, 0.9, threshold=1e-2)

    t = 12
    mean = mesh_parameter[start_state][t][0]
    std = mesh_parameter[start_state][t][1]
    duration = int(ceil(abs(np.random.normal(mean, std))))
    t += duration
    current_state = start_state
    for i in range(12, t):
        out.write(salt+","+str(i)+',' + current_state+',' + current_state+',' + 'stay' + '\n')
    while t < 47:
        for edge in graph.get_edges():
            if edge.get_destination() == start_state:
                a = reward[edge]
                reward[edge] = abs(pow(0.9, 47-t) * 4 * a)

        policy = irl.value_iteration.find_policy(graph, reward, 0.9, 1e-2, None, True)
        action = random_weight(policy[current_state])

        if len(policy[current_state]) > 1:

            while action.get_mode() == "stay" and action.get_destination() == action.get_origin():
                action = random_weight(policy[current_state])
            out.write(salt+","+str(t) + ',' + action.get_origin() + ',' + action.get_destination() + ',' + action.get_mode() + '\n')
            # print t, action.get_origin(), action.get_destination(), action.get_mode()

        current_state = action.get_destination()
        if current_state == start_state and t > 36:
            while t < 47:
                t += 1
                out.write(salt+","+str(t) + ',' + current_state + ',' + current_state + ',' + 'stay' + '\n')
                # print t, current_state, current_state, "stay"
        mean = mesh_parameter[start_state][t][0]
        std = mesh_parameter[start_state][t][1]
        duration = int(ceil(abs(np.random.normal(mean, std))))
        if t + duration < 47:
            for i in range(duration):
                t += 1
                out.write(salt+","+str(t) + ',' + current_state + ',' + current_state + ',' + 'stay' + '\n')
                # print t, current_state, current_state, "stay"
        else:
            while t < 47:
                t += 1
                out.write(salt+","+str(t) + ',' + current_state + ',' + current_state + ',' + 'stay' + '\n')
                # print t, action.get_origin(), action.get_destination(), "stay"


def choose_trajectory(length, id_trajectories):
    trajectories = []
    id_list = random.sample(id_trajectories, length)
    for uid in id_list:
        trajectories.append(id_trajectories.get(uid))
    return trajectories


def choose_mesh_trajectory(mesh, length, id_trajectories):
    samples = []

    for _id in id_trajectories:
        if 12 in id_trajectories[_id]:
            if id_trajectories[_id][12][0] == mesh:
                samples.append(id_trajectories[_id])

    if len(samples) >= length:
        return random.sample(samples, length)
    else:
        return choose_trajectory(length, id_trajectories)


def possible_mode(trajectories):
    mode_list = []
    for trajectory in trajectories:
        for i in range(12, 48):
            if i in trajectory:
                if trajectory[i][1].get_mode() not in mode_list:
                    mode_list.append(trajectory[i][1].get_mode())
    print mode_list
    return mode_list


def duration_gaussian(id_trajectories):
    """

    :param id_trajectories:
    :return:[mesh][t]?
    """
    mesh_slot_sample = dict()
    public_mesh_sample = dict()
    mesh_slot_parameter = dict()

    for uid in id_trajectories:
        prev_mode = ''
        prev_mesh = ''
        count = 0
        start_slot = 0

        for i in range(12, 48):
            if i in id_trajectories[uid]:
                if prev_mode != id_trajectories[uid][i][1].get_mode() and \
                                id_trajectories[uid][i][1].get_mode() == 'stay':
                    start_slot = i
                    count += 1

                elif prev_mode == id_trajectories[uid][i][1].get_mode() \
                        and id_trajectories[uid][i][1].get_mode()\
                        == 'stay':
                    count += 1

                elif prev_mode == 'stay' and id_trajectories[uid][i][1].get_mode() != 'stay':
                    sample = []
                    if prev_mesh not in mesh_slot_sample:
                        sample.append(count)
                        time_sample = dict()
                        time_sample[start_slot] = sample
                        mesh_slot_sample[prev_mesh] = time_sample
                    else:
                        if start_slot not in time_sample:
                            sample.append(count)
                            time_sample[start_slot] = sample
                            mesh_slot_sample[prev_mesh] = time_sample
                        else:
                            time_sample[start_slot].append(count)
                            mesh_slot_sample[prev_mesh] = time_sample

                    count = 0
                prev_mesh = id_trajectories[uid][i][1].get_origin()
                prev_mode = id_trajectories[uid][i][1].get_mode()

    for mesh in mesh_slot_sample:
        public_mesh_sample[mesh] = []
        for t in range(12, 48):
            if t in mesh_slot_sample[mesh]:
                public_mesh_sample[mesh].extend(mesh_slot_sample[mesh][t])

    for mesh in mesh_slot_sample:
        for t in range(12, 48):
            if t not in mesh_slot_sample[mesh]:
                mesh_slot_sample[mesh][t] = public_mesh_sample[mesh]

    for mesh in mesh_slot_sample:
        slot_parameter = dict()
        for t in range(12, 48):
            mean = np.mean(mesh_slot_sample[mesh][t])
            std = np.std(mesh_slot_sample[mesh][t])
            if std == 0:
                std += 0.1
            slot_parameter[t] = (mean, std)
        mesh_slot_parameter[mesh] = slot_parameter

    return mesh_slot_parameter


def generate_locchain(start_mesh, policy, move_policy, mesh_parameter):
    """

    :param start_mesh:
    :param move_policy:
    :param policy:
    :param mesh_parameter:
    :return:
    """
    current_location = start_mesh
    edge_chain = {}
    prev_action = 'stay'
    t = 12 + int(ceil(abs(np.random.normal(mesh_parameter[current_location][12][0],
                                           mesh_parameter[current_location][12][1]))))
    edge_chain[12] = graph.Edge(current_location, current_location, "stay")

    while t < 47:
        if prev_action == 'stay':
            edge = move_policy[current_location]
            edge_chain[t] = edge
            current_location = edge.get_destination()
        else:
            edge = policy[current_location]
            edge_chain[t] = edge
            current_location = edge.get_destination()
        print t, edge
        if edge.get_mode() == 'stay':
            t = t + int(ceil(abs(np.random.normal(mesh_parameter[current_location][t][0],
                                                  mesh_parameter[current_location][t][1]))))\
                if t + int(ceil(abs(np.random.normal(mesh_parameter[current_location][t][0],
                                                     mesh_parameter[current_location][t][1])))) < 48 else 47
        else:
            t += 1
        prev_action = edge.get_mode()
    return edge_chain


def calculate_edge_distance(edge):
    """
    calculate the cost of each action
    take distance as feature
    i:current state
    j:action(destination)
    :return:
    """
    if edge.get_mode() == "stay":
        return 0
    else:
        start = edge.get_origin()
        des = edge.get_destination()
        start_x, start_y = parse_MeshCode(start)
        des_x, des_y = parse_MeshCode(des)
        dist = haversine(start_x, start_y, des_x, des_y)
        return dist / 1000.0


def calculate_mesh_distance(origin, destination):
    """
    calculate the cost of each action
    take distance as feature
    i:current state
    j:action(destination)
    :return:
    """
    
    if origin == destination:
        dist = random.uniform(200, 1500)
    else:
        start_x, start_y = parse_MeshCode(origin)
        des_x, des_y = parse_MeshCode(destination)
        dist = haversine(start_x, start_y, des_x, des_y) if haversine(start_x, start_y, des_x, des_y) < 60000 else 60000
    return dist/1000.0


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
        y += float(LAT_HEIGHT_MESH1*int(mesh_code[0: 2]))
        x += 100 + int(mesh_code[2: 4])

    if strlen >= 6:
        y += float(LAT_HEIGHT_MESH2*int(mesh_code[4: 5]))
        x += float(LNG_WIDTH_MESH2*int(mesh_code[5: 6]))

    if strlen >= 8:
        y += float(LAT_HEIGHT_MESH3 * int(mesh_code[6: 7]))
        x += float(LNG_WIDTH_MESH3 * int(mesh_code[7: 8]))

    if strlen >= 9:
        n = int(mesh_code[8: 9])
        y += float(LAT_HEIGHT_MESH4*(0 if n <= 2 else 1))
        x += float(LNG_WIDTH_MESH4*(0 if n % 2 == 1 else 1))

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
    return c * r * 1000


def random_weight(weight_data):
    _total = sum(weight_data.values())
    for _k in weight_data.keys():
        if _k.get_mode() is "stay":
            _total -= weight_data[_k]
    _random = random.uniform(0, _total)
    _curr_sum = 0
    _ret = None

    for _k in weight_data.keys():
        if _k.get_mode() == "stay":
            continue
        _curr_sum += weight_data[_k]
        if _random <= _curr_sum:
            _ret = _k
            break
    return _ret


def random_temporal_weight(weight_data):
    _total = sum(weight_data.values())
    _random = random.uniform(0, _total)
    _curr_sum = 0
    _ret = None

    for _k in weight_data.keys():
        _curr_sum += weight_data[_k]
        if _random <= _curr_sum:
            _ret = _k
            break
    return _ret


def move_files(directory):
    files = os.listdir(directory)
    files_num = len(files) - 3
    print files_num

    if files_num >= 50:
        if not os.path.exists(directory + "training/"):
            os.mkdir(directory + "training/")
        if not os.path.exists(directory + "validation/"):
            os.mkdir(directory + "validation/")

        count = 0

        print files

        for filename in files:
            path = directory + filename
            print path
            if not os.path.isdir(path):
                print "not dir"
                if count < int(0.8*files_num):
                    shutil.move(path, directory + "training/")
                    print "move files", path
                    count += 1
                else:
                    shutil.move(path, directory + "validation/")

