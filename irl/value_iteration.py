"""
Find the value function associated with a policy. Based on Sutton & Barto, 1998.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np


def finite_optimal_value(g, reward, discount, slot):
    """

    :param g: graph
    :param reward:
    :param discount:
    :param slot:
    :return:
    """

    temporal_v = {}.fromkeys(range(12, slot+2), 0)

    v = {}.fromkeys(g.get_nodes(), temporal_v)

    for t in reversed(range(12, slot+2)):
        for node in g.get_nodes():
            max_v = float("-inf")
            for edge in g.get_node(node).get_edges():
                max_v = max(max_v, reward[t][edge] + discount * v[node][t])
            v[node][t] = max_v

    return v


def find_temporal_policy(g, reward, discount, slot, v=None, stochastic=False):
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
        v = finite_optimal_value(g, reward, discount, slot)
    if stochastic:
        Q = dict()
        for t in range(12, slot+2):
            Q[t] = dict()
            for node in g.get_nodes():
                Q[t][node] = dict()
                for edge in g.get_node(node).get_edges():
                    Q[t][node][edge] = reward[t][edge] + discount * v[edge.get_destination()][t]
                max_value = max(Q[t][node].items(), key=lambda x: x[1])[1]
                temp = 0
                for edge in g.get_node(node).get_edges():
                    Q[t][node][edge] -= max_value
                    temp += np.exp(Q[t][node][edge])

                for edge in g.get_node(node).get_edges():
                    Q[t][node][edge] = np.exp(Q[t][node][edge]) / temp
        return Q

    def _policy(s, step):
        potential = {}
        _node = g.get_node(s)
        for _edge in _node.get_edges():
            potential[_edge] = reward[step][_edge]+discount*v[_edge.get_destination()][step]
        return max(potential.items(), key=lambda x: x[1])[0]

    temporal_policy = {}.fromkeys(range(12, slot), 0)
    policy = {}.fromkeys(g.get_nodes(), temporal_policy)

    for t in range(12, slot):
        for node in g.get_nodes():
            policy[node][t] = _policy(node, t)
    return policy


def optimal_value(graph, reward, discount, threshold=1e-2):
    """
    reward is a dict!
    :param reward:
    :param discount:
    :param graph:
    :param threshold:
    :return:
    """
    v = dict()
    for node in graph.get_nodes():
        v[node] = 0

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for node in graph.get_nodes():
            max_v = float("-inf")
            for edge in graph.get_node(node).get_edges():
                max_v = max(max_v, reward[edge]+discount*v[node])

            new_diff = abs(v[node] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[node] = max_v
    return v


def find_policy(graph, reward, discount, threshold=1e-2, v=None, stochastic=False):
    """
    policy is a dict!
    :return:
    """
    if v is None:
        v = optimal_value(graph, reward, discount, threshold)

    if stochastic:
        Q = dict()
        for node in graph.get_nodes():
            Q[node] = dict()
            for edge in graph.get_node(node).get_edges():
                Q[node][edge] = reward[edge] + discount * v[edge.get_destination()]
            max_value = max(Q[node].items(), key=lambda x: x[1])[1]
            temp = 0
            for edge in graph.get_node(node).get_edges():
                Q[node][edge] -= max_value
                temp += np.exp(Q[node][edge])
            for edge in graph.get_node(node).get_edges():
                Q[node][edge] = np.exp(Q[node][edge]) / temp
        return Q

    def _policy(node):
        potential = {}
        _node = graph.get_node(node)
        for _edge in _node.get_edges():
            potential[_edge] = reward[_edge]+discount*v[_edge.get_destination()]
        return max(potential.items(), key=lambda x: x[1])[0]

    policy = dict()
    for node in graph.get_nodes():
        policy[node] = _policy(node)
    return policy


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
