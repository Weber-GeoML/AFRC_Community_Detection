"""
community_detection.py

Created on Feb 12 2023

@author: Lukas

This file contains all methods for community detection.
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


# community detection algorithm

def detect_communities(G, curvature, threshold):
    """
    Sequential deletion algorithm for detecting communities in a graph G

    Parameters
    ----------
    G : graph
        A networkx graph

    curvature : str
        The curvature to be used for community detection.

    threshold : float
        The threshold value for the curvature
        to be used for community detection.

    Returns
    -------
    G : graph
        A networkx graph with the detected communities as node labels.
    """
    # create a copy of the graph
    G_copy = deepcopy(G)

    # set graph attributes and calculate initial AFRC values
    curv_min, curv_max = get_min_max_curv_values(G_copy, curvature)

    # collect edges with extremal negative curvature
    if curvature == "afrc":
        threshold_list = [edge for edge in G_copy.edges.data()
                          if (edge[2][curvature] > threshold)]
        val_list = [edge for edge in threshold_list
                    if (edge[2][curvature] == curv_max)]

    else:
        threshold_list = [edge for edge in G_copy.edges.data()
                          if (edge[2][curvature] < threshold)]
        val_list = [edge for edge in threshold_list
                    if (edge[2][curvature] == curv_min)]

    while len(val_list) > 0:
        if len(val_list) == 1:
            # edge is the only element in the list
            extremum = val_list[0]
        else:
            # edge is randomly chosen from the list
            extremum = select_an_edge(val_list)

        (u, v) = extremum[:2]
        threshold_list.remove(extremum)

        # remove chosen edge
        G_copy.remove_edge(u, v)
        affecteds = list(G_copy.edges([u, v]))
        threshold_edges = [(u, v) for u, v, d in threshold_list]

        # delete all the cycles involving (u, v) from G_copy.cycles['triangles']
        G_copy.cycles['triangles'] = [cycle for cycle in G_copy.cycles['triangles']
                                      if (u, v) not in cycle]
        
        G_copy.cycles['quadrangles'] = [cycle for cycle in G_copy.cycles['quadrangles']
                                        if (u, v) not in cycle]

        # recompute curvature values
        if curvature == "frc":
            G_copy.compute_frc(affected_edges=affecteds + threshold_edges)

        elif curvature == "afrc":
            G_copy.compute_afrc(affected_edges=affecteds + threshold_edges)

        elif curvature == "orc":
            G_copy.compute_orc(affected_edges=affecteds + threshold_edges)

        elif curvature == "afrc_3":
            G_copy.compute_afrc_3(affected_edges=affecteds + threshold_edges)

        elif curvature == "afrc_4":
            G_copy.compute_afrc_4(affected_edges=affecteds + threshold_edges)

        if affecteds + threshold_edges != []:
            curv_min, curv_max = get_min_max_curv_values(G_copy, curvature,
                                                         affecteds + threshold_edges)

        # collect edges with extremal curvature
        if curvature == "afrc":
            threshold_list = [edge for edge in G_copy.edges.data()
                              if (edge[2][curvature] > threshold)]
            val_list = [edge for edge in threshold_list
                        if (edge[2][curvature] == curv_max)]

        else:
            threshold_list = [edge for edge in G_copy.edges.data()
                              if (edge[2][curvature] < threshold)]
            val_list = [edge for edge in threshold_list
                        if (edge[2][curvature] == curv_min)]

    # determine connected components of remaining graph
    C = [c for c in sorted(
        nx.connected_components(G_copy), key=len, reverse=True)]

    # Create list of tupels with node names and cluster labels
    set_node_labels(G, C, curvature)
    # print("removed edges: ", removed_edges)


# helper functions

cyc_names = {3: "triangles", 4: "quadrangles", 5: "pentagons"}


def set_edge_attributes_2(G, ll, i):
    """
    Set edge attributes triangles and curvature

    Parameters
    ----------
    G : graph
        A networkx graph

    ll : list
        List of lists of nodes

    i : int
        Number of nodes in each list

    Returns
    -------
    None.
    """
    for l in ll:
        for e1 in range(0, i):
            if e1 == i-1:
                e2 = 0
            else:
                e2 = e1 + 1
            u = l[e1]
            v = l[e2]
            G.edges[u, v][cyc_names[i]].append(l)


def get_min_max_curv_values(G, curvature, affected_edges=None):
    """
    Get minimum and maximum values of the curvature

    Parameters
    ----------
    G : graph
        A networkx graph

    curvature : str
        The curvature to be used for community detection.

    affected_edges : list
        List of edges to be considered for calculation of edge attributes

    Returns
    -------
    minimum : int
        Minimum value of curvature

    maximum : int
        Maximum value of curvature
    """
    if affected_edges is None:
        affected_edges = list(G.edges())

    affected_curvatures = [G.edges[edge][curvature] for edge in affected_edges]
    minimum = min(affected_curvatures)
    maximum = max(affected_curvatures)

    return minimum, maximum


def select_an_edge(edge_list):
    """
    Select an edge from a list of edges with uniform probability distribution

    Parameters
    ----------
    edge_list : list
        List of edges

    Returns
    -------
    edge : tuple
        A randomly chosen edge from the list

    """
    # randomly choose an edge from the list of edges
    edge = random.choice(edge_list)

    return edge


def set_node_labels(G, C, curvature):
    """
    Set node labels according to connected component labels

    Parameters
    ----------
    G : graph
        A networkx graph

    C : list
        List of clusters

    curvature : str
        The curvature to be used for community detection.

    Returns
    -------
    None.
    """
    for i, c in enumerate(C):
        for u in c:
            G.nodes[u][curvature + "_community"] = i


def get_clustering_accuracy(network_type, network, curvature, threshold=0):
    """
    Get clustering accuracy for a given network

    Parameters
    ----------
    network_type : str
        The type of network

    network : nx.Graph
        A networkx graph

    curvature : str
        The curvature to be used for community detection.

    threshold : int, optional
        The threshold for the curvature. The default is 0.

    Returns
    -------
    accuracy : float
        Clustering accuracy
    """
    # assert that network_type is either "sbm" or "hbg"
    assert network_type in ["sbm", "hbg"], "network_type must be either 'sbm' or 'hbg'"

    # get ground truth labels
    if network_type == "sbm":
        ground_truth = {}

        for node in network.nodes:
            block = network.nodes[node]["block"]

            if block not in ground_truth:
                ground_truth[block] = [node]

            else:
                ground_truth[block].append(node)

    elif network_type == "hbg":
        ground_truth = {0: [], 1: []}

        for node in network.nodes:
            if network.nodes[node]["group"] == "A1" or network.nodes[node]["group"] == "B1":
                ground_truth[0].append(node)

            else:
                ground_truth[1].append(node)

    # run community detection
    network.detect_communities(curvature, threshold)

    # get predicted labels
    predicted = {}

    for node in network.nodes:
        community = network.nodes[node][curvature + "_community"]

        if community not in predicted:
            predicted[community] = [node]

        else:
            predicted[community].append(node)

    # calculate clustering accuracy as the percentage of correctly detected communities
    accuracy = 0

    for community in ground_truth:
        if ground_truth[community] in predicted.values():
            accuracy += 1

    accuracy = accuracy / len(ground_truth)

    return accuracy