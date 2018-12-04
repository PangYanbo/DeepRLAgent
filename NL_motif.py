import datetime

import feature_extractor
import numpy as np
from scipy.optimize import minimize

from irl.mdp import graph
from utils import tools, load


def trip_getter(trajectory, order):
    trips = []
    for t in trajectory.keys():
        if trajectory[t][1].get_origin() != trajectory[t][1].get_destination():
            trips.append(trajectory[t][1])
    if order <= len(trips):
        return trips[order-1]
    else:
        return graph.Edge("", "", "")


def pop_feature(mesh):
    """

    :param mesh:
    :return:
    """
    if str(mesh) in feature['pop'].keys():
        return feature['pop'][str(mesh)]
    else:
        return 0


def office_feature(mesh):
    """

    :param mesh:
    :return:
    """
    if str(mesh) in feature['office'].keys():
        return feature['office'][str(mesh)][0]
    else:
        return 0


def office_pop_feature(mesh):
    """

    :param mesh:
    :return:
    """
    if str(mesh) in feature['office'].keys():
        return feature['office'][str(mesh)][1]
    else:
        return 0


def school_feature(mesh):
    """

    :param mesh:
    :return:
    """
    if str(mesh) in feature['school'].keys():
        return feature['school'][str(mesh)]
    else:
        return 0


def passenger_feature(mesh):
    """

    :param mesh:
    :return:
    """
    if str(mesh) in feature['passenger'].keys():
        return feature['passenger'][str(mesh)]
    else:
        return 0


starttime = datetime.datetime.now()

# load graph->alterntives
id_trajectory = load.load_trajectory(300)
mobility_graph = load.load_graph_traj(id_trajectory.values())

initial_list = []

for uid in id_trajectory.keys():
    initial_list.append(id_trajectory[uid][12][0])

origin_set = set(initial_list)

# load features
pop = feature_extractor.pop_feature()
office = feature_extractor.get_business()
passenger = feature_extractor.passanger_feature()
school = feature_extractor.school_feature()

feature = {'pop': pop, 'office': office, 'passenger': passenger, 'school': school}

# define parameter
param = np.zeros(20)

# define variable


def likelihood(x):
    level_one = {}
    level_two = {}
    level_three = {}
    level_four = {}
    level_five = {}

    LL = 0

    # first level
    for edge in mobility_graph.get_edges():
        destination = edge.get_destination()
        mode = edge.get_mode()

        u = x[0] * float(pop_feature([destination])) + x[1] * float(office_feature([destination]))\
            + x[2] * float(office_pop_feature([destination])) + x[3] * float(school_feature([destination]))\
            + x[4] * float(passenger_feature([destination])) + x[5] * (edge.get_dist() if mode == "walk" else 0)\
            + x[6] * (edge.get_dist() if mode == "vehicle" else 0) + x[7] * (edge.get_dist() if mode == "train" else 0)\
            + x[8] * (tools.calculate_edge_distance(edge) / 5.0 if mode == "walk" else 0)\
            + x[9] * (tools.calculate_edge_distance(edge) / 40.0 if mode == "vehicle" else 0)\
            + x[10] * (tools.calculate_edge_distance(edge) / 60.0 if mode == "train" else 0)
        level_one[edge] = np.exp(u)

    # second level
    for edge in mobility_graph.get_edges():
        logsum = 0
        for _edge in mobility_graph.get_node(edge.get_destination()).get_edges():
            logsum += level_one[_edge]

        logsum = np.log(logsum)

        u = x[0] * float(feature['pop'][destination]) + x[1] * float(feature['office'][destination][0]) \
            + x[2] * float(feature['office'][destination][1]) + x[3] * float(feature['school'][destination]) \
            + x[4] * float(feature['passenger'][destination]) + x[5] * (edge.get_dist() if mode == "walk" else 0) \
            + x[6] * (edge.get_dist() if mode == "vehicle" else 0) + x[7] * (edge.get_dist() if mode == "train" else 0)\
            + x[8] * (tools.calculate_edge_distance(edge) / 5.0 if mode == "walk" else 0) \
            + x[9] * (tools.calculate_edge_distance(edge) / 40.0 if mode == "vehicle" else 0) \
            + x[10] * (tools.calculate_edge_distance(edge) / 60.0 if mode == "train" else 0) + x[11] * logsum

        level_two[edge] = np.exp(u)

    # third level
    for edge in mobility_graph.get_edges():
        logsum = 0
        for _edge in mobility_graph.get_node(edge.get_destination()).get_edges():
            logsum += level_two[_edge]

        logsum = np.log(logsum)

        u = x[0] * float(feature['pop'][destination]) + x[1] * float(feature['office'][destination][0]) \
            + x[2] * float(feature['office'][destination][1]) + x[3] * float(feature['school'][destination]) \
            + x[4] * float(feature['passenger'][destination]) + x[5] * (edge.get_dist() if mode == "walk" else 0) \
            + x[6] * (edge.get_dist() if mode == "vehicle" else 0) + x[7] * (
            edge.get_dist() if mode == "train" else 0) \
            + x[8] * (tools.calculate_edge_distance(edge) / 5.0 if mode == "walk" else 0) \
            + x[9] * (tools.calculate_edge_distance(edge) / 40.0 if mode == "vehicle" else 0) \
            + x[10] * (tools.calculate_edge_distance(edge) / 60.0 if mode == "train" else 0) + x[11] * logsum

        level_three[edge] = np.exp(u)

    # forth level
    for edge in mobility_graph.get_edges():
        logsum = 0
        for _edge in mobility_graph.get_node(edge.get_destination()).get_edges():
            logsum += level_three[_edge]

        logsum = np.log(logsum)

        u = x[0] * float(feature['pop'][destination]) + x[1] * float(feature['office'][destination][0]) \
            + x[2] * float(feature['office'][destination][1]) + x[3] * float(feature['school'][destination]) \
            + x[4] * float(feature['passenger'][destination]) + x[5] * (edge.get_dist() if mode == "walk" else 0) \
            + x[6] * (edge.get_dist() if mode == "vehicle" else 0) + x[7] * (
                edge.get_dist() if mode == "train" else 0) \
            + x[8] * (tools.calculate_edge_distance(edge) / 5.0 if mode == "walk" else 0) \
            + x[9] * (tools.calculate_edge_distance(edge) / 40.0 if mode == "vehicle" else 0) \
            + x[10] * (tools.calculate_edge_distance(edge) / 60.0 if mode == "train" else 0) + x[11] * logsum

        level_four[edge] = np.exp(u)

    # fifth level
    for edge in mobility_graph.get_edges():
        logsum = 0
        for _edge in mobility_graph.get_node(edge.get_destination()).get_edges():
            logsum += level_four[_edge]

        logsum = np.log(logsum)

        u = x[0] * float(feature['pop'][destination]) + x[1] * float(feature['office'][destination][0]) \
            + x[2] * float(feature['office'][destination][1]) + x[3] * float(feature['school'][destination]) \
            + x[4] * float(feature['passenger'][destination]) + x[5] * (edge.get_dist() if mode == "walk" else 0) \
            + x[6] * (edge.get_dist() if mode == "vehicle" else 0) + x[7] * (
                edge.get_dist() if mode == "train" else 0) \
            + x[8] * (tools.calculate_edge_distance(edge) / 5.0 if mode == "walk" else 0) \
            + x[9] * (tools.calculate_edge_distance(edge) / 40.0 if mode == "vehicle" else 0) \
            + x[10] * (tools.calculate_edge_distance(edge) / 60.0 if mode == "train" else 0) + x[11] * logsum

        level_five[edge] = np.exp(u)

    motif1 = {}
    motif2 = {}
    motif3 = {}
    motif4 = {}
    motif5 = {}
    motif6 = {}

    for origin in origin_set:
        # print("features", pop_feature(origin), office_feature(origin), office_pop_feature(origin), school_feature(origin))

        motif1[origin] = np.exp(x[12] * 1 + x[13] * 0 + x[14] * 0 + x[15] * pop_feature(origin) + x[16] * office_feature(origin)\
            + x[17] * office_pop_feature(origin) + x[18] * school_feature(origin) + x[19] * passenger_feature(origin))

        # motif 2
        logsum = 0

        for edge in mobility_graph.get_node(origin).get_edges():
            logsum += level_one[edge]

        motif2[origin] = np.exp(x[12] * 2 + x[13] * 2 + x[14] * 1 + x[15] * pop_feature(origin) + x[16] * office_feature(origin)\
            + x[17] * office_pop_feature(origin) + x[18] * school_feature(origin) + x[19] * passenger_feature(origin)\
            + x[11] * np.log(logsum))

        # motif 3
        logsum = 0
        for edge in mobility_graph.get_node(origin).get_edges():
            logsum += level_four[edge]

        motif3[origin] = np.exp(x[12] * 3 + x[13] * 4 + x[14] * 2 + x[15] * pop_feature(origin) + x[16] * office_feature(origin)\
            + x[17] * office_pop_feature(origin) + x[18] * school_feature(origin) + x[19] * passenger_feature(origin) \
            + x[11] * np.log(logsum))

        # motif 4
        logsum = 0
        for edge in mobility_graph.get_node(origin).get_edges():
            logsum += level_three[edge]

        motif4[origin] = np.exp(x[12] * 3 + x[13] * 3 + x[14] * 1 + x[15] * pop_feature(origin) + x[16] * office_feature(origin)\
            + x[17] * office_pop_feature(origin) + x[18] * school_feature(origin) + x[19] * passenger_feature(origin)\
            + x[11] * np.log(logsum))

        # motif 5
        logsum = 0
        for edge in mobility_graph.get_node(origin).get_edges():
            logsum += level_four[edge]

        motif5[origin] = np.exp(x[12] * 3 + x[13] * 4 + x[14] * 2 + x[15] * pop_feature(origin) + x[16] * office_feature(origin)\
            + x[17] * office_pop_feature(origin) + x[18] * school_feature(origin) + x[19] * passenger_feature(origin)\
            + x[11] * np.log(logsum))

        # motif 6
        logsum = 0
        for edge in mobility_graph.get_node(origin).get_edges():
            logsum += level_five[edge]
        motif6[origin] = np.exp(x[12] * 4 + x[13] * 5 + x[14] * 2 + x[15] * pop_feature(origin) + x[16] * office_feature(origin)\
            + x[17] * office_pop_feature(origin) + x[18] * school_feature(origin) + x[19] * passenger_feature(origin)\
            + x[11] * np.log(logsum))

    for uid in id_trajectory:

        P_destination1, P_destination2, P_destination3, P_destination4, P_destination5 = 1, 1, 1, 1, 1

        temp_locchain = motif.getLocChain(id_trajectory[uid])
        locchain = motif.continueCheck(temp_locchain)
        motif_num = motif.motifChecker(locchain)

        trip_1 = trip_getter(id_trajectory[uid], 1)
        trip_2 = trip_getter(id_trajectory[uid], 2)
        trip_3 = trip_getter(id_trajectory[uid], 3)
        trip_4 = trip_getter(id_trajectory[uid], 4)
        trip_5 = trip_getter(id_trajectory[uid], 5)

        origin = id_trajectory[uid][12][0]

        # print(motif1[origin], motif2[origin], motif3[origin], motif4[origin], motif5[origin], motif6[origin])

        P_motif = 1

        if motif_num == 1:
            P_motif = motif1[origin] / (motif1[origin] + motif2[origin] + motif3[origin] + motif4[origin] + motif5[origin] + motif6[origin])
            P_motif = (P_motif != 0) * P_motif + (P_motif == 0)
        elif motif_num == 2:
            P_motif = motif2[origin] / (motif1[origin] + motif2[origin] + motif3[origin] + motif4[origin] + motif5[origin] + motif6[origin])
            P_motif = (P_motif != 0) * P_motif + (P_motif == 0)
        elif motif_num == 3:
            P_motif = motif3[origin] / (motif1[origin] + motif2[origin] + motif3[origin] + motif4[origin] + motif5[origin] + motif6[origin])
            P_motif = (P_motif != 0) * P_motif + (P_motif == 0)
        elif motif_num == 4:
            P_motif = motif4[origin] / (motif1[origin] + motif2[origin] + motif3[origin] + motif4[origin] + motif5[origin] + motif6[origin])
            P_motif = (P_motif != 0) * P_motif + (P_motif == 0)
        elif motif_num == 5:
            P_motif = motif5[origin] / (motif1[origin] + motif2[origin] + motif3[origin] + motif4[origin] + motif5[origin] + motif6[origin])
            P_motif = (P_motif != 0) * P_motif + (P_motif == 0)
        else:
            P_motif = motif6[origin] / (motif1[origin] + motif2[origin] + motif3[origin] + motif4[origin] + motif5[origin] + motif6[origin])
            P_motif = (P_motif != 0) * P_motif + (P_motif == 0)

        deno = 0

        for edge in mobility_graph.get_node(id_trajectory[uid][12][0]).get_edges():
            deno += level_one[edge]

        if trip_1 in level_one.keys():
            P_destination1 = level_one[trip_1] / deno

            deno = 0

            for edge in mobility_graph.get_node(trip_1.get_destination()).get_edges():
                deno += level_two[edge]

            if trip_2 in level_two.keys():
                P_destination2 = level_two[trip_2] / deno

                deno = 0

                for edge in mobility_graph.get_node(trip_2.get_destination()).get_edges():
                    deno += level_three[edge]

                if trip_3 in level_three.keys():
                    P_destination3 = level_three[trip_3] / deno

                    deno = 0

                    for edge in mobility_graph.get_node(trip_3.get_destination()).get_edges():
                        deno += level_four[edge]

                    if trip_4 in level_four.keys():
                        P_destination4 = level_four[trip_4] / deno

                        deno = 0

                        for edge in mobility_graph.get_node(trip_4.get_destination()).get_edges():
                            deno += level_five[edge]

                        if trip_5 in level_five.keys():
                            P_destination5 = level_five[trip_5] / deno
        # print(P_motif,P_destination1,P_destination2,P_destination3,P_destination4,P_destination5)
        P_motif * P_destination1 * P_destination2 * P_destination3 * P_destination4 * P_destination5
        # print(P_motif * P_destination1 * P_destination2 * P_destination3 * P_destination4 * P_destination5)
        LL += np.log(P_motif * P_destination1 * P_destination2 * P_destination3 * P_destination4 * P_destination5)
    print(x)
    print(LL)
    return -LL


likelihood(param)
res = minimize(likelihood, param)
print("parameter is ", res.x)

end = datetime.datetime.now()
print("calculation time:", end-starttime)