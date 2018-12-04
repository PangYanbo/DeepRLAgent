import numpy as np
import math
from collections import deque
from itertools import chain

from utils import tools


class Env:

    def __init__(self, graph, p_reward):
        # mobility graph
        self.graph = graph
        # reward function parameter
        self.p_reward = p_reward
        # initial state
        self.start = np.array([12, int(self.graph.get_start())])

        self.n_features = 20
        self.n_actions = len(graph.get_edges())
        self.action_space_dict = dict(zip(graph.get_edges(), range(len(graph.get_edges()))))
        self.action_space = graph.get_edges()

        self.pop = self.pop_feature()
        self.school = self.school_feature()
        self.office = self.get_business()
        self.passanger = self.passanger_feature()
        self.landuse = self.landuse_feature()
        self.entertainment = self.entertainment()
        self.evacuate = self.evacuate()
        self.feature_matrix = self.feature_matrix(self.graph)

    def step(self, t, action):
        """
        observation = (t, node)
        :param t: current time step
        :param action: edge in graph
        :return: next state
                 reward: dict[edge]
                 done: home?
                 info
        """

        action = self.action_space[action]

        observation = np.array([t+1, int(action.get_destination())])

        reward = self.feature_matrix[t][action].dot(self.p_reward[t])
        # print("reward", t, action, reward)

        done = 0

        if t+1 == 47:
            done = True

        return observation, reward, done, {}

    def reset(self):
        return self.start

    def render(self, mode='human', close=False):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def flatten_features(self, features):
        return np.array(list(chain(*features)))

    def get_business(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), ('0', '0'))
        with open('/home/ubuntu/Data/Tokyo/OFFICECOUNTPOP/OFFICECOUNTPOP.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[1], tokens[2])

        return mesh_info

    def landuse_feature(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), ('0', '0', '0', '0', '0', '0', '0',
                                                                '0', '0', '0', '0', '0'))
        with open('/home/ubuntu/Data/Tokyo/LandUse/Landuse.csv') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(',')
                mesh_info[tokens[0]] = (tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[6], tokens[7],
                                        tokens[8], tokens[9], tokens[10], tokens[11], tokens[12])

        return mesh_info

    def entertainment(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), 0)
        with open('/home/ubuntu/Data/Tokyo/Entertainment/Entertainment.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = float(tokens[1])

        return mesh_info

    def evacuate(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), 0)
        with open('/home/ubuntu/Data/Tokyo/Evacuate/Evacuate.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = float(tokens[1])

        return mesh_info

    def road_feature(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), 0)
        with open('/home/ubuntu/Data/Tokyo/Road/roadlength.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[1], tokens[2], tokens[3])

        return mesh_info

    def pop_feature(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), 0)
        with open('/home/ubuntu/Data/Tokyo/MESHPOP/MeshPop.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[2])

        return mesh_info

    def passanger_feature(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), 0)
        with open('/home/ubuntu/Data/Tokyo/STOPINFO/dailyrailpassanger.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[1])

        return mesh_info

    def school_feature(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), 0)
        with open('/home/ubuntu/Data/Tokyo/SCHOOL/SchoolCount.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[1])

        return mesh_info

    ######################################upddate at 2018/09/26#########################################################

    def agent_hist_feature(self, agent_hist):
        features = deque(maxlen=len(agent_hist))
        for i in range(len(agent_hist)):
            features.append(self.observation_feature(agent_hist[i]))

        features = self.flatten_features(features)
        features = features[np.newaxis, :]

        return features

    def observation_feature(self, observation):
        """

        :param node: int mesh number
        :return:
        """
        f = np.zeros(20)
        destination = str(observation[1])

        # business features
        f[0] = float(self.office[destination][1])
        f[1] = float(self.office[destination][0])
        # school features
        f[2] = float(self.school[destination])
        # population features
        f[3] = float(self.pop[destination])
        # landuse features
        f[4] = float(self.landuse[destination][0])
        f[5] = float(self.landuse[destination][1])
        f[6] = float(self.landuse[destination][2])
        f[7] = float(self.landuse[destination][3])
        f[8] = float(self.landuse[destination][4])
        f[9] = float(self.landuse[destination][5])
        f[10] = float(self.landuse[destination][6])
        f[11] = float(self.landuse[destination][7])
        f[12] = float(self.landuse[destination][8])
        f[13] = float(self.landuse[destination][9])
        f[14] = float(self.landuse[destination][10])
        f[15] = float(self.landuse[destination][11])
        # syukyaku facilities
        f[16] = self.entertainment[destination] if observation[0] in range(18, 36) else 0
        # evacuate
        f[17] = self.evacuate[destination]
        # time of day feature
        f[18] = math.sin(math.pi*observation[0]/24.)
        f[19] = math.cos(math.pi*observation[0]/24.)
        return f

    def feature_vector(self, t, edge, start):
        """
        :param t:
        :param edge:
        :param start:
        :return:
        """
        destination = edge.get_destination()
        mode = edge.get_mode()
        f = np.zeros(31)
        time_cost = 0
        if mode == "stay":
            time_cost = 0
        if mode == "walk":
            time_cost = tools.calculate_edge_distance(edge) / 5.0
        if mode == "vehicle":
            time_cost = tools.calculate_edge_distance(edge) / 40.0
        if mode == "train":
            time_cost = tools.calculate_edge_distance(edge) / 60.0

        # business features
        f[0] = float(self.office[destination][1])
        f[1] = float(self.office[destination][0])
        # school features
        f[2] = float(self.school[destination])
        # population features
        f[3] = float(self.pop[destination])
        # landuse features
        f[4] = float(self.landuse[destination][0])
        f[5] = float(self.landuse[destination][1])
        f[6] = float(self.landuse[destination][2])
        f[7] = float(self.landuse[destination][3])
        f[8] = float(self.landuse[destination][4])
        f[9] = float(self.landuse[destination][5])
        f[10] = float(self.landuse[destination][6])
        f[11] = float(self.landuse[destination][7])
        f[12] = float(self.landuse[destination][8])
        f[13] = float(self.landuse[destination][9])
        f[14] = float(self.landuse[destination][10])
        f[15] = float(self.landuse[destination][11])
        f[16] = 0
        # syukyaku facilities
        f[17] = self.entertainment[destination] if t in range(18, 36) else 0
        # evacuate
        f[18] = self.evacuate[destination]

        f[19] = 1 if mode == "stay" else 0
        f[20] = 1 if mode == "walk" else 0
        f[21] = 1 if mode == "vehicle" else 0
        f[22] = 1 if mode == "train" else 0
        f[23] = time_cost
        f[24] = float(self.passanger[destination]) if mode == "train" else 0
        f[25] = edge.get_dist() if mode == "walk" else 0
        f[26] = edge.get_dist() if mode == "vehicle" else 0
        f[27] = edge.get_dist() if mode == "train" else 0

        f[28] = 1 if destination == start else 0
        f[29] = 1 if destination == start and mode == 'stay' else 0

        f[30] = 0

        return f

    def feature_matrix(self, graph):
        features = {}
        for t in range(12, 48):
            feature = {}
            for edge in graph.get_edges():
                f = self.feature_vector(t, edge, self.graph.get_start())
                feature[edge] = f
            features[t] = feature
        return features
