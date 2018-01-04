import tools
import math


class Node(object):
    """

    """
    _name = ""
    _edges = []

    def __init__(self, name, edges):
        self._name = name
        self._edges = edges

    def get_name(self):
        return self._name

    def add_edge(self, edge):
        if edge not in self.get_edges():
            self._edges.append(edge)

    def remove_edge(self, destination):
        if destination not in self._edges:
            raise KeyError("Cannot remove non-existent edge from node")
        self._edges.remove(destination)

    def get_edges(self):
        return self._edges

    def get_edge(self, edge):
        if edge in self._edges:
            return edge

    def __str__(self):
        return "<GraphNode>: \"" + self._name + "\" with edges " + str(self._edges)


class Edge(object):
    """

    """
    _origin = ""
    _destination = ""
    _mode = ""
    _dist = 0

    def __init__(self, origin, destination, mode):
        self._origin = origin
        self._destination = destination
        self._mode = mode
        self._dist = 0 if mode == 'stay' else tools.calculate_mesh_distance(origin, destination)*1000

    def get_origin(self):
        return self._origin

    def get_destination(self):
        return self._destination

    def get_mode(self):
        return self._mode

    def get_dist(self):
        return self._dist

    def set_dist(self, distance):
        self._dist = distance

    def __hash__(self):
        return hash(self.get_origin()+self.get_destination()+self.get_mode())

    def __eq__(self, other):
        return self.get_origin() == other.get_origin() \
               and self.get_destination() == other.get_destination() and self.get_mode() == other.get_mode()

    def __repr__(self):
        return "edge({},{},{},{})".format(self._origin, self._destination, self._mode, self._dist)


class Graph(object):
    """

    """
    _nodes = {}
    _edges = []
    _isDirected = False

    def __init__(self, nodes=None, edges=None, directed=False, normalization=True):
        self._isDirected = directed
        if len(nodes) > 0:
            self._nodes = {}
            for node in nodes:
                self.add_node(node)
        if len(edges) > 0:
            self._edges = []
            for edge in edges:
                self.add_edge(edge)
        self._edges = self.get_edges()
        if normalization:
            self.dist_normalization()

    def is_directed(self):
        return self._isDirected

    def add_node(self, node):
        try:
            if node in self._nodes:
                raise KeyError("Cannot add duplicate node `" + node + "` to graph.")
            self._nodes[node] = Node(node, [])
        except:
            print "add_node error"

    def remove_node(self, node):
        if node not in self._nodes:
            raise KeyError()
        del self._nodes[node]

    def get_nodes(self):
        return self._nodes

    def get_node(self, node):
        if node not in self._nodes:
            return None
        return self._nodes[node]

    def get_node_number(self):
        return len(self._nodes)

    def get_edge_number(self):

        return len(self._edges)

    def add_edge(self, edge):
        try:
            if edge.get_origin() not in self._nodes:
                raise KeyError()
            if edge.get_destination() not in self._nodes:
                raise KeyError()
            self._nodes[edge.get_origin()].add_edge(edge)
            if self._isDirected:
                converse_edge = Edge(edge.get_destination(), edge.get_origin(), edge.get_mode())
                self._nodes[edge.get_destination()].add_edge(converse_edge)
        except(AttributeError, ValueError, IndexError, TypeError):
            print "errrrrrrrrrrrrrrrroooooooooooooooooooooooooooorrrrrrrrrrrrrrrrrrrrrrrrrrrr........"

    def get_edge(self, edge):
        count = 0
        for e in self._edges:
            if e == edge:
                count += 1
                return e
        if count == 0:
            raise KeyError

    def get_edges(self):
        edges = []
        for node in self.get_nodes():
            edges.extend(self.get_node(node).get_edges())
        return edges

    def remove_edge(self, origin, destination):
        if origin not in self._nodes:
            raise KeyError()
        if destination not in self._nodes:
            raise KeyError()
        self._nodes[origin].remove(destination)
        if not self._isDirected:
            self._nodes[destination].remove(origin)

    def dist_normalization(self):
        dist_walk = []
        dist_vehicle = []
        dist_train = []

        # for edge in self._edges:
            # if edge.get_mode() == "walk":
            #     dist_walk.append(edge.get_dist())
            # if edge.get_mode() == "vehicle":
            #     dist_vehicle.append(edge.get_dist())
            # if edge.get_mode() == "train":
            #     dist_train.append(edge.get_dist())

        dist = []
        for edge in self._edges:
            dist.append(edge.get_dist())

        max_dist = max(dist)
        min_dist = min(dist)
        print max_dist
        #
        # for edge in self._edges:
        #     if edge.get_mode() != "stay":
        #         edge.set_dist(math.log10(edge.get_dist()) / math.log10(max_dist))

        # max_walk = max(dist_walk)
        # max_vehicle = max(dist_vehicle)
        # max_train = 0 if len(dist_train) == 0 else max(dist_train)
        # mean_walk = np.mean(dist_walk)
        # std_walk = np.std(dist_walk)
        # mean_bike = np.mean(dist_bike)
        # std_bike = np.std(dist_bike)
        # mean_vehicle = np.mean(dist_vehicle)
        # std_vehicle = np.std(dist_vehicle)
        # mean_train = np.mean(dist_train)
        # std_train = np.std(dist_train)
        # print max_value_walk, max_value_bike, max_value_vehicle, max_value_train
        # print min_value_walk, min_value_bike, min_value_vehicle, min_value_train
        for edge in self._edges:
            if edge.get_mode() == "stay":
                edge.set_dist(0.0)
            else:
                edge.set_dist(math.log10(edge.get_dist()) / math.log10(max_dist))
            # if edge.get_mode() == "walk":
            #     edge.set_dist(math.log10(edge.get_dist())/math.log10(max_dist))
            # if edge.get_mode() == "vehicle":
            #     edge.set_dist(math.log10(edge.get_dist()) / math.log10(max_dist))
            # if edge.get_mode() == "train":
            #     edge.set_dist(math.log10(edge.get_dist()) / math.log10(max_dist))
            # if edge.get_mode() == "walk":
            #     edge.set_dist((edge.get_dist()-mean_walk)/std_walk)
            # if edge.get_mode() == "bike":
            #     edge.set_dist((edge.get_dist()-mean_bike)/std_bike)
            # if edge.get_mode() == "vehicle":
            #     edge.set_dist((edge.get_dist()-mean_vehicle)/std_vehicle)
            # if edge.get_mode() == "train":
            #     edge.set_dist((edge.get_dist()-mean_train)/std_train)

    def __repr__(self):
        return "<Graph>:\n  " + ("\n  ".join([str(self.get_node(node)) for node in self._nodes]))
