# -----------------------------2018/11/27--------------------------
import os
import sys
sys.path.append('/home/ubuntu/PycharmProjects/DeepAgent/')
import random
import numpy as np
import datetime
import math
import networkx as nx
import jismesh.utils as ju
import utils.tools as tools
from collections import deque
from itertools import chain

class State(object):

    def __init__(self, node, graph=None, **kwargs):
        self.node = node
        
        if graph is not None:
           self.graph = graph
           self.actions = self.action_generator()

       # self.node.update(kwargs)
        

    def action_generator(self):
        actions = []
        nodes = random.sample(list(self.graph.nodes), 500)

        for n in nodes:
            action = Action(self.node, n)
            actions.append(action)
      
        return actions

    def __repr__(self):
        return '<node:{}>'.format(self.node)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):

        return self.node == other.node

class Action(object):

    def __init__(self, origin, destination, graph=None):
        self.origin = origin
        self.destination = destination
        
        if graph is not None:
            self.path = []
            count = 0
            for n in nx.shortest_path(G,origin, destination,'length'):
                if count > 0:
                    edge = G.edges[origin][destination]
                    self.path.append(edge)
       

    def __repr__(self):
      
        return '<Origin:{}, Destination:{}>'.format(self.origin, self.destination)

    def __eq__(self, other):
        
        return self.origin == other.origin and self.destination == other.destination    

    def __hash__(self):
        return hash(repr(self))

class Env(object):
    """
	# import road network from openstreet map
        import road network from DRM and rail network
    """
    
    def __init__(self, G):
        self.graph = G
        #self.sub_graph = self.sub_graph(800)

#        self.trajectories = list(self.load_trajectory(n_demonstrations).values())
#        self.trajectory_length = len(self.trajectories[0])
	
        self.n_nodes = len(self.graph.nodes)
        self.n_edges = len(self.graph.edges)
	
       # global network
       # self.state_space, self.node_state = self.initial_state_space()
       # self.action_space, self.action_index = self.initial_action_space()

       # partial network from trajectories
        self.trajectories = list(self.load_trajectories('/home/ubuntu/PycharmProjects/DeepRLAgent/demonstrations/test_sub.csv', 1200).values())
        self.state_space, self.action_space = self.init_space_from_demo() 
        self.initial_state()

        self.n_states = len(self.state_space)
        self.n_actions = len(self.action_space)

        self.n_features = 110
        self.theta = np.zeros([self.n_features])
        self.telepoint_feature = self.load_telepoint_feature('/home/ubuntu/Data/Tokyo/Telepoint/')
        self.weather_feature = self.load_weather_feature('/home/ubuntu/Data/Tokyo/Weather/20150101.csv')

        self.feature_space = self.feature_dict()       


    def step(self, s, a, t):
	# TODO
        # action.destination is a string of node
        s_ = a
        r = self.reward((t, s, a))   
        
        done = False
        
        if t >= 46:
            done = True
        
        return s_, r, done

    def reset(self):
	# TODO
        # randomly return a initial state?
        # reset environment?
        
        return random.choice(self.state_space)

    def initial_state(self):
        initial_state = []

        for traj in self.trajectories:
            initial_state.append(traj[0][1])
        self.initial_state = initial_state
        print("initial state count: ", len(initial_state))


    def sub_graph(self, size):
        nodes = random.sample(self.graph.nodes, size)
        subgraph = self.graph.subgraph(nodes)
        
        return subgraph

    def init_space_from_demo(self):
        state_space = {}
        action_space = []
        for traj in self.trajectories:
            for i in range(len(traj)):
                time, s, a = traj[i]
                if s not in state_space:
                    state_space[s] = {a}
                else:
                    state_space[s].add(a)
                action_space.append(a)
        return state_space, action_space


    def initial_state_space(self):
        state_space = []
        node_state = {}
        #nodes = random.sample(list(self.graph.nodes), 800) 

        for n in self.sub_graph:
            state = State(n, self.sub_graph)
            state_space.append(state)
            node_state[n] = state
        return state_space, node_state

    def initial_action_space(self):
        action_space = []
        
        [action_space.extend(state.actions) for state in self.state_space]

        index_action = dict(zip(action_space, range(len(action_space))))

        return action_space, index_action


    def state_feature(self, s):
        """
        1.start coordinate
        2.end coordinate
        3.time
        4.route length
        5.nearby features?
        """
        origin = s.node
        f = np.zeros(self.n_features)
        if origin in self.graph.nodes:
            f[0] = self.graph.nodes[origin]['x']
            f[1] = self.graph.nodes[origin]['y']
        return f 

    def flatten_features(self, features):
        return np.array(list(chain(*features)))

    def agent_hist_feature(self, agent_hist):
        features = deque(maxlen=len(agent_hist))
        for i in range(len(agent_hist)):
            features.append(self.feature_vector(agent_hist[i]))
        features = self.flatten_features(features)
        features = features[np.newaxis, :]
 
        return features

    def load_telepoint_feature(self, path):
        """
        data is collected from telepoint, zenrin
        office count is sorted by mesh in level 5
        """
        index_mesh_count = dict()

        tele_point_list = os.listdir('/home/ubuntu/Data/Tokyo/Telepoint/')
 
        for i in range(len(tele_point_list)):
            filename =  tele_point_list[i]
            index = filename[:-4]
            index_mesh_count[index] = dict()
            with open(path+filename, 'r') as f:
                for line in f.readlines():
                    tokens = line.strip('\n').split(',')
                    mesh = tokens[0]
                    count = int(tokens[1])
                    index_mesh_count[index][mesh] = count

        return index_mesh_count

    def load_weather_feature(self, path):
        """
        """
        time_column_info = {}
        with open(path, 'r') as f:
            f.readline()
            for line in f.readlines():
                tokens = line.strip('\n').split(',')
                time = int(tokens[0])
                time_column_info[time] = dict()
                for i in range(len(tokens)-1):
                    if i < 9:
                        time_column_info[time][i] = float(tokens[i])  
        return time_column_info


    def episode_feature(self, episode):
        """
        inputs: episode(t, s, a)
        output: vector(np.array([self.n_features]))
        """
        t, s, a = episode
        origin = s
        destination = a
        dest_mesh = ju.to_meshcode(self.graph.nodes[destination]['y'], self.graph.nodes[destination]['x'], 5)

        # path = nx.shortest_path(self.graph, origin, destination)

        f = np.zeros([self.n_features])
        
        i = 0 
        for key in self.telepoint_feature:
            if dest_mesh in self.telepoint_feature[key]:
                f[i] = self.telepoint_feature[key][dest_mesh] 
            i += 1             

        #f[i] = self.graph.nodes[origin]['x']
        #f[i+1] = self.graph.nodes[origin]['y']
        #f[i+2] = self.graph.nodes[destination]['x']
        #f[i+3] = self.graph.nodes[destination]['y']
        # weather data
        for j in range(8):
            f[i+3+j] = self.weather_feature[int(t/2)][j]
        
        return f

    def feature_dict(self):
        episode_features = dict()
        for traj in self.trajectories:
            for episode in traj:
                if episode not in episode_features:
                    episode_features[episode] = self.episode_feature(episode)
        print(episode_features.keys())
        return episode_features


    def update_reward_function(self, theta):
        self.theta = theta

    def reward(self, episode):
        if episode in self.feature_space:
            print("episode in feature space")
            return self.theta.dot(self.feature_space[episode])
        else:
            print("episode not in feature space")
            return self.theta.dot(self.episode_feature(episode))

    def load_episode(self, path):

        episodes = []
        episode = []
        prev_id = ''
        count = 0        

        with open(path, 'r') as f:
            for line in f.readlines():
                if count > 10000:
                    break
                tokens = line.strip('\n').split(',')
                uid = tokens[0]
                timestamp = int(tokens[1])
                origin = tokens[2]
                destination = tokens[3]
                mode = tokens[4]
                if uid != prev_id:
                    if len(episode)>0:
                        episodes.append(episode)
                    episode = []                  
                state = State(origin, self.graph)
                action = Action(origin, destination)
                reward = self.reward(state, action, timestamp)
                episode.append((state, action, reward))
                prev_id = uid
                count += 1
        return episodes

    def trajectory_rewards(self):
        rewards = []
        for traj in self.trajectories:
            reward = sum([self.reward(e) for e in traj])
            rewards.append(reward)

        return np.average(rewards)   

    def load_trajectories(self, path, number):
        id_traj = {}
        count = 0

        with open(path, 'r') as f:
            for line in f.readlines():
                try:
                    if count > number:
                        break
                    count += 1
                    tokens = line.strip('\n').split(',')
                    uid = tokens[0]
                    time = int(tokens[1])
                    origin = tokens[2]
                    dest = tokens[3]
                    # mode = tokens[4]
                   
                    episode = (time, origin, dest)
                   
                    if uid not in id_traj:
                        traj = [episode]
                        id_traj[uid] = traj
                    else:
                        id_traj[uid].append(episode)
                except(AttributeError, ValueError, IndexError, TypeError):
                    print('Load Trajectory Error')
        f.close()
        return id_traj
