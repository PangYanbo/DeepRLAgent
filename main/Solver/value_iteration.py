"""
Find the value function associated with a policy. Based on Sutton & Barto, 1998.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import main.Env.urbanEnv as urbanEnv
# import matplotlib.pyplot as plt


def finite_optimal_value(env, reward, discount, slot):
    """

    :param g: graph
    :param reward:
    :param discount:
    :param slot:
    :return:
    """

    value = []

    temporal_v = {}.fromkeys(range(12, slot+2), 0)

    v = {}.fromkeys(env.graph.get_nodes(), temporal_v)

    for t in reversed(range(12, slot+2)):
        for node in env.graph.get_nodes():
            max_v = float("-inf")
            for edge in env.graph.get_node(node).get_edges():
                max_v = max(max_v, reward[t][edge] + discount * v[node][t])
            v[node][t] = max_v
            value.append(v[node][t])
        # plt.figure()
        # plt.title("Value ")
        # # plt.ylim(0, 0.1)
        # plt.plot(range(len(value)), value)
        # plt.show()
        # plt.savefig(path + "_" + str(t) + "_" + str(learning_rate) + "_" + str(epochs) + ".png")

    return v


def find_temporal_policy(env, reward, slot, v=None, stochastic=True):
    """

    :param g:
    :param reward:
    :param discount:
    :param slot:
    :param v:
    :param stochastic:
    :return:
    """
    if v is None:
        v = finite_optimal_value(env, reward, env.discount, slot)
    if stochastic:
        Q = dict()
        for t in range(12, slot+2):
            Q[t] = dict()
            for node in env.graph.get_nodes():
                Q[t][node] = dict()
                for edge in env.graph.get_node(node).get_edges():
                    Q[t][node][edge] = reward[t][edge] + env.discount * v[edge.get_destination()][t]
                max_value = max(Q[t][node].items(), key=lambda x: x[1])[1]
                temp = 0
                for edge in env.graph.get_node(node).get_edges():
                    Q[t][node][edge] -= max_value
                    temp += np.exp(Q[t][node][edge])

                for edge in env.graph.get_node(node).get_edges():
                    Q[t][node][edge] = np.exp(Q[t][node][edge]) / temp
        return Q

    def _policy(s, step):
        potential = {}
        _node = env.graph.get_node(s)
        for _edge in _node.get_edges():
            potential[_edge] = reward[step][_edge] + env.discount * v[_edge.get_destination()][step]
        return max(potential.items(), key=lambda x: x[1])[0]

    temporal_policy = {}.fromkeys(range(12, slot), 0)
    policy = {}.fromkeys(env.graph.get_nodes(), temporal_policy)

    for t in range(12, slot):
        for node in env.graph.get_nodes():
            policy[node][t] = _policy(node, t)
    return policy


def find_policy(graph, reward, discount, v=None, stochastic=True):
    """
    :param graph:
    :param reward:
    :param discount:
    :param v:
    :param stochastic:
    :return:
    """
    if v is None:
        v = optimal_value(graph, reward, discount)
    if stochastic:
        Q = dict()
        for t in range(12, 48):
            Q[t] = dict()
            for node in graph.get_nodes():
                Q[t][node] = dict()
                for edge in graph.get_node(node).get_edges():
                    Q[t][node][edge] = reward[t][edge] + discount * v[edge.get_destination()][t]
                max_value = max(Q[t][node].items(), key=lambda x: x[1])[1]
                temp = 0
                for edge in graph.get_node(node).get_edges():
                    Q[t][node][edge] -= max_value
                    temp += np.exp(Q[t][node][edge])

                for edge in graph.get_node(node).get_edges():
                    Q[t][node][edge] = np.exp(Q[t][node][edge]) / temp
        return Q


def optimal_value(graph, reward, discount):
    """

    :param graph:
    :param reward:
    :param discount:
    :return:
    """
    value = []

    temporal_v = {}.fromkeys(range(12, 48), 0)

    v = {}.fromkeys(graph.get_nodes(), temporal_v)

    for t in reversed(range(12, 48)):
        for node in graph.get_nodes():
            max_v = float("-inf")
            for edge in graph.get_node(node).get_edges():
                max_v = max(max_v, reward[t][edge] + discount * v[node][t])
            v[node][t] = max_v
            value.append(v[node][t])

    return v



# def optimal_value(graph, reward, discount, threshold=1e-2):
#     """
#     reward is a dict!
#     :param reward:
#     :param discount:
#     :param graph:
#     :param threshold:
#     :return:
#     """
#     v = dict()
#     for node in graph.get_nodes():
#         v[node] = 0
#
#     diff = float("inf")
#     while diff > threshold:
#         diff = 0
#         for node in graph.get_nodes():
#             max_v = float("-inf")
#             for edge in graph.get_node(node).get_edges():
#                 max_v = max(max_v, reward[edge]+discount*v[node])
#
#             new_diff = abs(v[node] - max_v)
#             if new_diff > diff:
#                 diff = new_diff
#             v[node] = max_v
#     return v


# def find_policy(graph, reward, discount, threshold=1e-2, v=None, stochastic=False):
#     """
#     policy is a dict!
#     :return:
#     """
#     if v is None:
#         v = optimal_value(graph, reward, discount, threshold)
#
#     if stochastic:
#         Q = dict()
#         for node in graph.get_nodes():
#             Q[node] = dict()
#             for edge in graph.get_node(node).get_edges():
#                 Q[node][edge] = reward[edge] + discount * v[edge.get_destination()]
#             max_value = max(Q[node].items(), key=lambda x: x[1])[1]
#             temp = 0
#             for edge in graph.get_node(node).get_edges():
#                 Q[node][edge] -= max_value
#                 temp += np.exp(Q[node][edge])
#             for edge in graph.get_node(node).get_edges():
#                 Q[node][edge] = np.exp(Q[node][edge]) / temp
#         return Q
#
#     def _policy(node):
#         potential = {}
#         _node = graph.get_node(node)
#         for _edge in _node.get_edges():
#             potential[_edge] = reward[_edge]+discount*v[_edge.get_destination()]
#         return max(potential.items(), key=lambda x: x[1])[0]
#
#     policy = dict()
#     for node in graph.get_nodes():
#         policy[node] = _policy(node)
#     return policy

def move_policy(graph, reward, discount, threshold=1e-2, v=None):
    if v is None:
        v = optimal_value(graph, reward, discount, threshold)

    def _policy(node):
        potential = {}
        _node = graph.get_node(node)
        for _edge in _node.get_edges():
            if len(_node.get_edges()) > 1:
                if _edge.get_mode() != 'stay' and _edge.get_origin() != _edge.get_destination():
                    potential[_edge] = reward[_edge]+discount*v[_edge.get_destination()]
            else:
                potential[_edge] = reward[_edge] + discount * v[_edge.get_destination()]
        return max(potential.items(), key=lambda x: x[1])[0]

    policy = dict()
    for node in graph.get_nodes():
        policy[node] = _policy(node)
    return policy


# def find_temporal_policy(graph, reward, discount, threshold=1e-2):
#     """
#     policy is a dict!
#     :return:
#     """
#     v = dict()
#     policy = dict().fromkeys(range(12, 47), dict())
#
#     for t in range(12, 47):
#         v[t] = optimal_value(graph, reward[t], discount, threshold)
#
#         def _policy(node):
#             potential = {}
#             _node = graph.get_node(node)
#             for _edge in _node.get_edges():
#                 potential[_edge] = reward[t][_edge]+discount*v[t][_edge.get_destination()]
#             return max(potential.items(), key=lambda x: x[1])[0]
#
#         for node in graph.get_nodes():
#             policy[t][node] = _policy(node)
#
#     return policy

def value_iteration(env, reward, discount, threshold=1e-2, deterministic=True):
    """

    :param env:
    :param reward:
    :param discount:
    :param threshold:
    :param deterministic:
    :return:
    """
    n_states = env.n_state
    n_actions = env.n_action

    v = np.zeros([n_states])

    diff = float("inf")
    while diff > threshold:
        for s in range(n_states):
            max_v = float("-inf")
            for a in env.sub_action_space(env.index_state[s]):
                state = env.index_action[a].get_destination()
                # t = env.index_state[s].get_time()
                # state = urbanEnv.State(env.index_action[a].get_destination(), t+1)
                if state not in env.state_space:
                    continue
                s_ = env.state_index[state]
                print("reward", reward)
                print("reward type", type(reward))
                max_v = max(max_v, reward[s, a] + discount * v[s_])
            print(max_v)

            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v

    if deterministic:
        policy = np.zeros([n_states])

        for s in range(n_states):
            policy[s] = np.argmax([reward[s, a] + discount*v[env.state_index[[env.index_action[a].get_destination()]]]
                                   for a in range(n_actions)])
    else:
        policy = np.zeros([n_states])

    return v, policy

