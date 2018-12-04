import os
import numpy as np
import datetime
import random
import math
import main.Env.graph as graph
import utils.tools as tools


# ----------------------------2018/08/22 update----------------------------------
# rewards are static over time
# state -> node in graph

class Env(object):
    """
    MDP for urban environment
    """

    def __init__(self, path):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        read features:
        road desnsity
        office count
        school count

        -> Urban MDP
        """
        self.path = path
        self.trajectories = list(self.load_trajectory().values())
        self.trajectory_length = len(self.trajectories[0])
        self.graph = self.load_graph_traj()

        self.n_node = self.graph.get_node_number()
        self.n_edges = self.graph.get_edge_number()

        self.state_space = list(self.graph.get_nodes().keys())
        print("state space", self.state_space)
        self.action_space = self.graph.get_edges()

        self.n_state = len(self.state_space)
        self.n_action = len(self.action_space)

        self.state_index = dict(zip(self.state_space, range(self.n_state)))
        self.index_state = {values: key for key, values in self.state_index.items()}

        self.action_index = dict(zip(self.action_space, range(self.n_action)))
        self.index_action = {values: key for key, values in self.action_index.items()}

        # too large, unable to calculate and store
        # self.transition_probability = np.array([[[self._transition_probability(i, j, k)
        #                                           for k in range(self.n_state)]
        #                                          for j in range(self.n_action)]
        #                                         for i in range(self.n_state)])

        # -----------------------------------------------------------------------------

        self.n_features = 31
        self.pop = self.pop_feature()
        self.school = self.school_feature()
        self.office = self.get_business()
        self.passanger = self.passanger_feature()
        self.landuse = self.landuse_feature()
        self.entertainment = self.entertainment()
        self.evacuate = self.evacuate()

        print('state space: ', self.n_state, 'action space: ', self.n_action)

        self.feature_matrix = self.feature_matrix()

    def __str__(self):
        return "Gridworld({}, {})".format(self.graph.get_node_number(), self.discount)

    def step(self):
        # TODO
        return 0

    def transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given action j.
        :param i: state
        :param j: action
        :param k: _state
        :return:
        """
        state = self.index_state[i]
        action = self.index_action[j]
        _state = self.index_state[k]

        if state.get_time() == _state.get_time()+1 and state.get_loc() == action.get_origin() and\
                action.get_destination() == _state.get_loc():
            return 1.
        else:
            return 0.0

    def sub_action_space(self, s):
        """

        :param s:
        :return:
        """
        sub_action_space = []

        for edge in self.graph.get_node(s).get_edges():
            sub_action_space.append(self.action_index[edge])

        return sub_action_space

    def load_trajectory(self):

        id_traj = {}
        count = 0

        starttime = datetime.datetime.now()
        # with open("home/t-iho/Result/trainingdata/trainingdata"+date+".csv") as f:
        with open(self.path) as f:
            for line in f.readlines():
                try:
                    count += 1
                    if count % 10000 == 0:
                        print("finish " + str(count) + " lines")

                    if count > 3600:
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

    def load_graph_traj(self):
        """"

        """
        nodes = {}
        edges = {}

        for trajectory in self.trajectories:
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

    def find_feature_expections(self):
        """
        Find the feature expectations for the given trajectories.
        -> Feature expecation with state-action pair
        :return:
        """
        feature_expectations = np.zeros(self.n_features)

        for uid in self.trajectories:
            for traj in self.trajectories[uid]:
                feature_expectations += self.feature_matrix[traj[0].get_time()-12][self.action_index[traj[1]]]

        return feature_expectations

    def get_business(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), ('0', '0'))
        with open('/home/ubuntu/Data/Tokyo/OFFICECOUNTPOP/OFFICECOUNTPOP.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[1], tokens[2])

        return mesh_info

    def landuse_feature(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), ('0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
                                                                '0'))
        with open('/home/ubuntu/Data/Tokyo/LandUse/Landuse.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[1], tokens[2], tokens[3], tokens[4], tokens[5], tokens[6], tokens[7],
                                        tokens[8], tokens[9], tokens[10], tokens[11], tokens[12])

        return mesh_info

    def entertainment(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), 0.0)
        with open('/home/ubuntu/Data/Tokyo/Entertainment/Entertainment.csv', 'r') as f:
            # with open('C:/Users/PangYanbo/Desktop/Tokyo/MESHPOP/MeshPop.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = float(tokens[1])

        return mesh_info

    def evacuate(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), 0.0)
        with open('/home/ubuntu/Data/Tokyo/Evacuate/Evacuate.csv', 'r') as f:
            # with open('C:/Users/PangYanbo/Desktop/Tokyo/MESHPOP/MeshPop.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = float(tokens[1])

        return mesh_info

    def road_feature(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), ("0", "0", "0"))
        with open('/home/t-iho/Data/IRL/roadlength.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[1], tokens[2], tokens[3])

        return mesh_info

    def pop_feature(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), "0")
        with open('/home/ubuntu/Data/Tokyo/MESHPOP/MeshPop.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[2])

        return mesh_info

    def passanger_feature(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), "0")
        with open('/home/ubuntu/Data/Tokyo/STOPINFO/dailyrailpassanger.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[1])

        return mesh_info

    def school_feature(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), "0")
        with open('/home/ubuntu/Data/Tokyo/SCHOOL/SchoolCount.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[1])

        return mesh_info

    def _feature_vector(self, t, action):
        """

        :param t: int
        :param action: int
        :return:
        """
        # if action in self.index_action.keys():
        if True:
            edge = self.index_action[action]

            destination = edge.get_destination()
            mode = edge.get_mode()
            f = np.zeros(self.n_features)

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
            # night population features
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
            # attractive facilities
            f[16] = self.entertainment[destination]
            # evacuate
            f[17] = self.evacuate[destination]

            f[18] = 1 if mode == "stay" else 0
            f[19] = 1 if mode == "walk" else 0
            f[20] = 1 if mode == "vehicle" else 0
            f[21] = 1 if mode == "train" else 0
            f[22] = time_cost * 10
            f[23] = float(self.passanger[destination]) if mode == "train" else 0
            f[24] = edge.get_dist() if mode == "walk" else 0
            f[25] = edge.get_dist() if mode == "vehicle" else 0
            f[26] = edge.get_dist() if mode == "train" else 0
            f[27] = math.sin(math.pi * t / 24.)
            f[28] = math.cos(math.pi * t / 24.)
            # disaster
            # print edge, start, destination == start

            return f

    def state_feature(self, state, t):
        destination = state

        f = np.zeros(18)
        try:
            if destination in self.office.keys():
                f[0] = float(self.office[destination][1])
                f[1] = float(self.office[destination][0])
            # school features
            if destination in self.school.keys():
                f[2] = float(self.school[destination])
            # night population features
            if destination in self.pop.keys():
                f[3] = float(self.pop[destination])
            # landuse features
            if destination in self.landuse.keys():
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
            # attractive facilities
            if destination in self.entertainment.keys():
                f[16] = self.entertainment[destination] if t in range(18, 36) else 0
            # evacuate
            if destination in self.evacuate.keys():
                f[17] = self.evacuate[destination]
        except KeyError:
            print(state, "not in keys")
        return f

    def feature_matrix(self):
        fm = np.zeros([48, self.n_action, self.n_features])

        for t in range(0, 48):
            for a in range(self.n_action):
                fm[t-12, a] = self._feature_vector(t-12, a)

        return fm

    def load_alpha(self):
        alpha = {}
        path = "/home/ubuntu/Data/PT_Result/dynamics_param_previous"
        files = os.listdir(path)
        with open(path + "/" + random.choice(files), 'r') as f:
            t = 12
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                param = np.zeros(28)
                for j in range(28):
                    if len(tokens) > j:
                        param[j] = tokens[j]
                alpha[t] = param.copy()
                t += 1
        self.alpha = alpha

    def reward(self, alpha):
        reward = dict()

        for t in range(12, 47):
            reward[t] = dict()
            for edge in self.action_space:
                reward[t][edge] = alpha[t].dot(self._feature_vector(t, self.action_index[edge]))

        return reward

    def draw_reward(self, mesh_list):
        reward = dict()
        for t in range(12, 47):
            reward[t] = dict()
            with open("/home/ubuntu/Result/reward_"+str(t)+".csv", "w")as f:
                for state in mesh_list:
                    if True:
                        alpha = self.alpha[t][0:18]
                        feature = self.state_feature(state, t)
                        reward[t][state] = alpha.dot(feature)
                        f.write(state+","+str(reward[t][state]))
                        f.write("\n")


def read_list(path):
    mesh_list = []
    with open(path, "r")as f:
        for line in f:
            tokens = line.strip("\r\n").split(",")
            mesh = tokens[2]
            mesh_list.append(mesh)
    return mesh_list


if __name__ == '__main__':
    env = Env(0.9, "/home/ubuntu/Data/all_train_irl.csv")
    mesh_list = read_list("/home/ubuntu/Data/Tokyo/MeshCode/Tokyo2.csv")
    env.load_alpha()
    env.draw_reward(mesh_list)

