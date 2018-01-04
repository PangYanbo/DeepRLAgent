import numpy as np
import tools


class Gridworld(object):
    """
    Gridworld MDP.
    """

    def __init__(self, graph, discount):
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

        self.graph = graph
        self.discount = discount
        self.pop = self.pop_feature()
        self.school = self.school_feature()
        self.office = self.get_business()
        self.passanger = self.passanger_feature()

    def __str__(self):
        return "Gridworld({}, {})".format(self.graph.get_node_number(), self.discount)

    def get_business(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), ('0', '0'))
        with open('C:/Users/PangYanbo/Desktop/Tokyo/OFFICECOUNTPOP/OFFICECOUNTPOP.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[1], tokens[2])

        return mesh_info

    def road_feature(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), 0)
        with open('/home/t-iho/Data/IRL/roadlength.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[1], tokens[2], tokens[3])

        return mesh_info

    def pop_feature(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), 0)
        with open('C:/Users/PangYanbo/Desktop/Tokyo/MESHPOP/MeshPop.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[2])

        return mesh_info

    def passanger_feature(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), 0)
        with open('C:/Users/PangYanbo/Desktop/Tokyo/STOPINFO/dailyrailpassanger.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[1])

        return mesh_info

    def school_feature(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), 0)
        with open('C:/Users/PangYanbo/Desktop/Tokyo/SCHOOL/SchoolCount.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[1])

        return mesh_info

    def landuse_feature(self):
        mesh_info = {}.fromkeys(self.graph.get_nodes().keys(), 0)
        with open('C:/Users/PangYanbo/Desktop/Tokyo/SCHOOL/SchoolCount.csv', 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = line.split(",")
                mesh_info[tokens[0]] = (tokens[1], tokens[5])

        return mesh_info

    def feature_vector(self, edge):
        """
        :param edge:
        :return:
        """
        destination = edge.get_destination()
        mode = edge.get_mode()
        f = np.zeros(11)
        time_cost = 0
        if mode == "stay":
            time_cost = 0
        if mode == "walk":
            time_cost = tools.calculate_edge_distance(edge)/5.0
        if mode == "vehicle":
            time_cost = tools.calculate_edge_distance(edge)/40.0
        if mode == "train":
            time_cost = tools.calculate_edge_distance(edge)/60.0

        f[0] = float(self.office[destination][1])
        f[1] = float(self.office[destination][0])
        f[2] = float(self.school[destination])
        f[3] = float(self.pop[destination])
        f[4] = 1 if mode == "stay" else 0
        f[5] = 0
        f[6] = time_cost
        f[7] = float(self.passanger[destination]) if mode == "train" else 0
        f[8] = edge.get_dist() if mode == "walk" else 0
        f[9] = edge.get_dist() if mode == "vehicle" else 0
        f[10] = edge.get_dist() if mode == "train" else 0

        return f

    def feature_matrix(self, graph):
        features = {}
        for edge in graph.get_edges():
            f = self.feature_vector(edge)
            features[edge] = f
        return features
