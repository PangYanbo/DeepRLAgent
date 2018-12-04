import numpy as np
import datetime
import irl.mdp.graph as graph
import utils.tools as tools


# ----------------------------2018/08/16 update----------------------------------
class State:

    def __init__(self, loc, time):
        self.loc = loc
        self.time = time

    def get_loc(self):
        return self.loc

    def get_time(self):
        return self.time

    def __str__(self):
        return "State({},{})".format(self.get_loc(), str(self.get_time()))

    def __hash__(self):
        return hash(self.loc + str(self.time))


class Gridworld(object):
    """
    Gridworld MDP.
    """

    def __init__(self, g, discount, path):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        read features:
        road desnsity
        office count
        school count

        -> Gridworld
        """
        self.path = path
        self.graph = g
        self.discount = discount

        # ----------------------- 2018/08/16 update------------------------------------
        self.n_node = g.get_node_number()
        self.n_edges = g.get_edge_number()

        self.state_space = self.init_state_space()
        print(self.state_space)
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

        self.trajectories = self.load_trajectory()
        self.feature_matrix = self.feature_matrix()

    def __str__(self):
        return "Gridworld({}, {})".format(self.graph.get_node_number(), self.discount)

    # ----------------------- 2018/08/16 update------------------------------------
    def init_state_space(self):
        state_space = []
        for node in self.graph.get_nodes():
            for t in range(12, 48):
                state_space.append(State(node, t))
        return state_space

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
        node = self.index_state[s].get_loc()

        for edge in self.graph.get_node(node).get_edges():
            sub_action_space.append(self.action_index[edge])

        return sub_action_space

    # solve this MDP
    def optimal_value(self, reward, threshold=1e-2):
        """

        :param reward:
        :param threshold:
        :return:
        """
        v = np.zeros(self.n_state)

        diff = float('inf')

        while diff > threshold:
            diff = 0
            for s in range(self.n_state):
                max_v = float("-inf")
                for a in self.sub_action_space(s):
                    tp = np.array([self.transition_probability(s, a, _s) for _s in range(self.n_state)])
                    max_v = max(max_v, np.dot(tp, np.average(reward, axis=1) + self.discount*v))

                new_diff = abs(v[s]-max_v)
                if new_diff > diff:
                    diff = new_diff
                v[s] = max_v

        return v

    def find_policy(self, reward, threshold=1e-2, v=None):
        """

        :param reward: -> SxA
        :param threshold:
        :param v:
        :return: -> SxA
        """
        if v is None:
            v = self.optimal_value(reward, threshold)

        Q = np.zeros((self.n_state, self.n_action))
        for i in range(self.n_state):
            for j in self.sub_action_space(i):
                p = np.array([self.transition_probability(i, j, k) for k in range(self.n_state)])
                Q[i, j] = p.dot(np.average(reward, axis=1) + self.discount*v)
        Q -= Q.max(axis=1).reshape((self.n_state, 1))
        Q = np.exp(Q) / np.exp(Q).sum(axis=1).reshape((self.n_state, 1))

        return Q

    # -----------------------------update 2018/08/17-------------------------------
    def load_trajectory(self):
        """
        trajectory format: (state, action)-> State(node, timestamp), Edge(origin, destination, mode)
        :return: dict->id_trajectories, trajectory -> list() of traj tuple(state, action)
        """
        id_traj = {}
        count = 0

        starttime = datetime.datetime.now()
        with open(self.path) as f:
            for line in f.readlines():
                try:
                    count += 1
                    if count % 100000 == 0:
                        print("finish " + str(count) + " lines")

                    line = line.strip('\r\n')
                    tokens = line.split(",")
                    agent_id = tokens[0]
                    timeslot = int(tokens[1])
                    start = tokens[2]
                    end = tokens[3]
                    mode = tokens[4]

                    state = State(start, timeslot)
                    e = graph.Edge(start, end, mode)

                    traj = (state, e)

                    if agent_id not in id_traj.keys():
                        trajectory = []
                        id_traj[agent_id] = trajectory
                        id_traj[agent_id].append(traj)
                    else:
                        id_traj[agent_id].append(traj)

                except(AttributeError, ValueError, IndexError, TypeError):
                    print("Loading Trajectory Error")

        f.close()
        endtime = datetime.datetime.now()
        print("finished loading" + str(len(id_traj)) + "trajectories in" + str(endtime - starttime))

        return id_traj

    def find_feature_expections(self):
        """
        Find the feature expectations for the given trajectories.
        -> Feature expecation with state-action pair
        :return:
        """
        feature_expectations = np.zeros(self.n_features)

        for trajectory in self.trajectories:
            for traj in trajectory:
                feature_expectations += self.feature_matrix[traj]

        return feature_expectations

    # -----------------------------------------------------------------------------

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

    def feature_vector(self, t, edge, start):
        """

        :param t:
        :param edge:
        :param start:
        :return:
        """

        destination = edge.get_destination()
        mode = edge.get_mode()
        f = np.zeros(self.n_features)
        time_cost = 0
        if mode == "stay":
            time_cost = 0
        if mode == "walk":
            time_cost = tools.calculate_edge_distance(edge)/5.0
        if mode == "vehicle":
            time_cost = tools.calculate_edge_distance(edge)/40.0
        if mode == "train":
            time_cost = tools.calculate_edge_distance(edge)/60.0

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
        f[16] = 0
        # attractive facilities
        f[17] = self.entertainment[destination] if t in range(18, 36) else 0
        # evacuate
        f[18] = self.evacuate[destination]

        f[19] = 1 if mode == "stay" else 0
        f[20] = 1 if mode == "walk" else 0
        f[21] = 1 if mode == "vehicle" else 0
        f[22] = 1 if mode == "train" else 0
        f[23] = time_cost * 10
        f[24] = float(self.passanger[destination]) if mode == "train" else 0
        f[25] = edge.get_dist() if mode == "walk" else 0
        f[26] = edge.get_dist() if mode == "vehicle" else 0
        f[27] = edge.get_dist() if mode == "train" else 0
        f[28] = 1 if destination == start else 0
        f[29] = 1 if destination == start and mode == "stay" else 0
        # disaster
        f[30] = 0
        # print edge, start, destination == start

        return f

    def feature_matrix(self):
        features = {}
        for t in range(12, 48):
            feature = {}
            for edge in self.graph.get_edges():
                f = self.feature_vector(t, edge, self.graph.get_start())
                feature[edge] = f
            features[t] = feature
        return features

    # ----------------------- 2018/08/17 update------------------------------------
    def _feature_vector(self, state, action):
        """

        :param state: int
        :param action: int
        :return:
        """

        node = self.index_state[state][0]
        t = self.index_state[state][1]

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
        f[16] = 0
        # attractive facilities
        f[17] = self.entertainment[destination] if t in range(18, 36) else 0
        # evacuate
        f[18] = self.evacuate[destination]

        f[19] = 1 if mode == "stay" else 0
        f[20] = 1 if mode == "walk" else 0
        f[21] = 1 if mode == "vehicle" else 0
        f[22] = 1 if mode == "train" else 0
        f[23] = time_cost * 10
        f[24] = float(self.passanger[destination]) if mode == "train" else 0
        f[25] = edge.get_dist() if mode == "walk" else 0
        f[26] = edge.get_dist() if mode == "vehicle" else 0
        f[27] = edge.get_dist() if mode == "train" else 0
        f[28] = 0
        f[29] = 0
        # disaster
        f[30] = 0
        # print edge, start, destination == start

        return f