# -----------------------------2018/11/27--------------------------
import os
import random
import numpy as np
import datetime
import math
import utils.tools as tools
import omnsx as ox

class Env(object):
    """
	import road network from openstreet map
    """
    
    def __init__(self, area, n_demonstrations, path):
	self.graph = ox.graph_from_place(area, network_type='drive')	
	self.path = path
        self.trajectories = list(self.load_trajectory(n_demonstrations).values())
        self.trajectory_length = len(self.trajectories[0])
	
	self.n_nodes = len(self.graph.nodes)
	self.n_edges = len(self.graph.edges)
	
	self.state_space = self.graph.nodes
	self.action_space = self.graph.edges

	self.n_states = len(self.graph.nodes)
	self.n_actions = len(self.graph.edges)

	self.n_features = 30	

    def step(self, s, a):
	# TODO
	return 0

    def reset(self):
	# TODO
	return 0

    def sub_action_space(self, s):
	sub_action_space = []
	
	for e in self.graph.edges(s):
	    sub_action_space.append(e)

	return sub_action_space

    def feature_vector(self, a):
	origin = self.graph.edges[a][0]
	destination = self.graph.edges[a][1]

	f = np.zeros(self.n_features)

	
