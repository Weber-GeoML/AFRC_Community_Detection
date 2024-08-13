"""
auxiliary_functions.py

Created on Feb 12 2023

@author: Lukas

This file contains all auxiliary methods used in the remainder of this repo.
"""

# import packages

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import networkx.algorithms.community as nx_comm
import os
import json
import random


def save_data_to_json_file(d, filename):
    """
    Save data to json file. Used for saving positions of network nodes.
    Allows repeated identical visualizations of the same network.

    Parameters
    ----------
    d : dict
        Data to be saved.

    filename : str
        Name of the file to be saved.

    Returns
    -------
    None.
    """
    json_string = json.dumps(d, indent=4)
    json_file = open(filename, "w")
    json_file.write(json_string)
    json_file.close()
    return None


def read_data_from_json_file(fn):
    """
    Read data from json file. Used for reading positions of network nodes.

    Parameters
    ----------
    fn : str
        Name of the file to be read.

    Returns
    -------
    d : dict
        Data read from file.
    """
    f = open(fn)
    d = json.load(f)
    f.close()
    return d


def pos_array_as_list(p):
    """
    Convert pos dict to list.

    Parameters
    ----------
    p : dict
        Dictionary of node arrays.

    Returns
    -------
    d : dict
        Dictionary of positions as lists.
    """
    d = {k: list(a) for k, a in iter(p.items())}
    return d


def pos_list_as_array(p):
    """
    Convert pos dict to array.

    Parameters
    ----------
    p : dict
        Dictionary of node lists.

    Returns
    -------
    d : dict
        Dictionary of positions as arrays.
    """
    d = {k: np.array(a) for k, a in iter(p.items())}
    return d


def get_amf_pos(G, sc, rd, cx, cy, mv, val):
    """
    Get positions of nodes for the AMF network layout.

    Parameters
    ----------
    G : networkx graph
        Graph to be drawn.

    sc : float
        Scale of the layout.

    rd : float
        Radius of the layout.

    cx : float
        x-coordinate of the center of the layout.

    cy : float
        y-coordinate of the center of the layout.

    mv : int
        Maximum value of the node attribute "value".

    val : set
        Set of values of the node attribute "value".

    Returns
    -------
    p : dict
        Dictionary of positions of nodes.
    """
    p = {}
    cnt = np.array([cx, cy])
    for v in val:
        temp = nx.circular_layout(
                    nx.subgraph(G, [n for n, d in iter(G.nodes.items()) if d["value"] == v]),
                    scale=sc,
                    center=cnt + np.array([rd * np.cos(v/(mv+1)*2*np.pi),
                                           rd * np.sin(v/(mv+1)*2*np.pi)])
                    )
        p.update(temp)
    return p


def build_size_list(k, l):
    """
    Build list of number of nodes per community.

    Parameters
    ----------
    k : int
        Number of nodes.

    l : int
        Number of community.

    Returns
    -------
    ll : list
        List of number of nodes per community.
    """
    ll = [k for i in range(l)]
    return ll


def build_prob_list(l, p_in, p_out):
    """
    Build list of probabilities for SBM.

    Parameters
    ----------
    l : int
        Number of communities.

    p_in : float
        Probability of edge within community.

    p_out : float
        Probability of edge between communities.

    Returns
    -------
    ll : list
        List of lists of probabilities for SBM.
        p_in on the main diagonal, p_out elsewhere.
    """
    ll = []
    for i in range(l):
        temp_l = [p_out for j in range(0, i)] + [p_in] + [p_out for j in range(i+2, l+1)]
        ll.append(temp_l)
    return ll


def get_pos_layout(H, fn=""):
    """
    Get positions of nodes for network layout.
    If fn is empty, create new Kamada-Kawai layout.
    Otherwise, read positions from file.

    Parameters
    ----------
    H : networkx graph
        Graph to be drawn.

    fn : str, optional
        Name of file containing positions of nodes.

    Returns
    -------
    pos : dict
        Dictionary of positions of nodes.
    """
    if fn == "":
        pos = nx.kamada_kawai_layout(H)
    else:
        cwd = os.getcwd()
        full_fn = os.path.join(cwd, fn)
        pos = pos_list_as_array(read_data_from_json_file(full_fn))
        pos = {int(k): v for (k, v) in iter(pos.items())}
    return pos


def save_pos_layout(pos, fn=""):
    """
    Save positions of nodes for network layout.
    If fn is empty, do nothing.

    Parameters
    ----------
    pos : dict
        Dictionary of positions of nodes.

    fn : str, optional
        Name of file containing positions of nodes.

    Returns
    -------
    None.
    """
    if fn != "":
        cwd = os.getcwd()
        full_fn = os.path.join(cwd, fn)
        save_data_to_json_file(pos_array_as_list(pos), full_fn)


def save_pos_sbm(p, k, n):
    """
    Save positions of nodes for network layout.
    Only used for SBM.

    Parameters
    ----------
    p : dict
        Dictionary of positions of nodes.

    k : int
        Number of nodes in each community.

    n : int
        Number of communities.

    Returns
    -------
    None.
    """
    cwd = os.getcwd()
    fn = "pos_SBM_graph_" + str(k) + "_nodes_in_" + str(n) + "_communities.json"
    full_fn = os.path.join(cwd, fn)
    save_data_to_json_file(pos_array_as_list(p), full_fn)


def read_pos_sbm(k, n):
    """
    Read positions of nodes for network layout.
    Only used for SBM.

    Parameters
    ----------
    k : int
        Number of nodes in each community.

    n : int
        Number of communities.

    Returns
    -------
    p : dict
        Dictionary of positions of nodes.
    """
    cwd = os.getcwd()
    fn = "pos_SBM_graph_" + str(k) + "_nodes_in_" + str(n) + "_communities.json"
    full_fn = os.path.join(cwd, fn)
    p = pos_list_as_array(read_data_from_json_file(full_fn))
    p = {int(k): v for (k, v) in iter(p.items())}
    return p


def get_bipartite_graph(n=40, m=40, p_high=0.7, p_low=0.2):
    """
    Builds a bipartite graph with groups A (with subgroups A1 and A2) and B (with subgroups B1 and B2),
    adds edges between subgroups A1-B1 and A2-B2 with high probability and
    edges between subgroups A1-B2 and A2-B1 with low probability

    Parameters
    ----------
    n : integer
        number of nodes in A (split evenly to get A1 and A2)
    m : integer
        number of nodes in B (split evenly to get B1 and B2)
    p_high : integer
        probability for B1-A1 and B2-A2
    p_low : integer
        probability for B1-A2 and B2-A1

    Returns
    -------
    B : graph (bipartite)
    """

    # assert n,m are even
    n = (n // 2) * 2
    m = (m // 2) * 2

    # create empty graph
    B = nx.Graph()

    # add nodes subgroupwise
    # definition of subgroups in tuple list
    nodes_struct = [(n//2, 0, "A1"), (n//2, 0, "A2"), (m//2, 1, "B1"), (m//2, 1, "B2")] 
    a, b = 0, 0
    for nd in nodes_struct:
        b += nd[0]
        B.add_nodes_from(list(range(a, b)), bipartite=nd[1], group=nd[2])
        a += nd[0]

    # add edges if binom. distrib. yields 1
    edges_struct = [("A1", "B1", p_high), ("A1", "B2", p_low), ("A2", "B1", p_low), ("A2", "B2", p_high)]
    for ed in edges_struct:
        edge_tupels = []
        gr0 = [n for n, d in B.nodes.data() if d["group"] == ed[0]]
        gr1 = [n for n, d in B.nodes.data() if d["group"] == ed[1]]
        binom_dist = np.random.binomial(1, ed[2], (len(gr0), len(gr1)))
        for i, n0 in enumerate(gr0):
            for j, n1 in enumerate(gr1):
                if binom_dist[i, j] == 1:
                    edge_tupels.append((n0, n1))
        B.add_edges_from(edge_tupels, prob=1 if ed[2] == p_high else 0)

    return B


def get_edges_between_blocks(list_1, list_2, e):
    """
    Get edges between blocks.

    Parameters
    ----------
    list_1 : list
        List of degrees of nodes in first block.

    list_2 : list
        List of degrees of nodes in second block.

    e : int
        Number of edges between blocks.

    Returns
    -------
    edges : list
        List of edges between blocks.

    new_list_1 : list
        List of degrees of nodes in first block.

    new_list_2 : list
        List of degrees of nodes in second block.
    """
    edges = []

    aux_list_1 = []
    aux_list_2 = []

    for i in range(len(list_1)):
        for j in range(list_1[i]):
            aux_list_1.append(i)

    for i in range(len(list_2)):
        for j in range(list_2[i]):
            aux_list_2.append(i)

    for i in range(e):
        stub_1 = random.choice(aux_list_1)
        stub_2 = random.choice(aux_list_2)

        aux_list_1.remove(stub_1)
        aux_list_2.remove(stub_2)

        edges.append((stub_1, stub_2))

    new_list_1 = [0] * len(list_1)
    new_list_2 = [0] * len(list_2)

    for i in range(len(list_1)):
        new_list_1[i] = aux_list_1.count(i)

    for i in range(len(list_2)):
        new_list_2[i] = aux_list_2.count(i)

    return edges, new_list_1, new_list_2


def assign_edges(G, compare_key):
    """
    Allocate the edges of a given graph to within or between community edges.
    Node communities can be distinguished by the attribute "compare_key".

    Parameters
    ----------
    G : graph
        The input graph.

    compare_key : string
        The attribute key to compare.

    Returns
    -------
    G : graph
        The input graph with the attribute "group" added to each edge.
    """

    for e in G.edges:
        if G.nodes[e[0]][compare_key] == G.nodes[e[1]][compare_key]:
            G.edges[e]["group"] = "within"
        else:
            G.edges[e]["group"] = "between"
    return G