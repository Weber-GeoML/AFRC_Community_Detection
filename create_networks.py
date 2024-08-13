"""
create_networks.py

Created on Feb 12 2023

@author: Lukas

This file contains all methods used to
create artificial and real-world networks.
"""

# import packages

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import networkx.algorithms.community as nx_comm
import os
import json


# auxiliary function for creating a degree-corrected Stochastic Block Model

def determine_between_edges(node_affiliations,
                            node_degrees, num_blocks, param):
    """
    Determine the number of edges between communities in a degree-corrected
    Stochastic Block Model.

    Parameters
    ----------
    node_affiliations : list
        List of node affiliations.

    node_degrees : list
        List of node degrees.

    num_blocks : int
        Number of blocks.

    param : float
        The parameter for the degree-corrected Stochastic Block Model.

    Returns
    -------
    between_edges : dict
        A dictionary of the number of edges between the blocks
        with sets as keys.
    """

    # create a dictionary of the number of edges between communities
    between_edges = {}
    for i in range(num_blocks):
        for j in range(i, num_blocks):
            between_edges[(i, j)] = 0

    # for each block, sum the degrees of the nodes in the block
    block_degrees = {}
    for i in range(num_blocks):
        block_degrees[i] = 0

    for i in range(len(node_affiliations)):
        block_degrees[node_affiliations[i]] += node_degrees[i]

    # for each pair of blocks {i, j}, choose a random integer between 0 and the
    # minimum of block_degrees[i] and block_degrees[j] and update
    # block_degrees[i] and block_degrees[j] accordingly
    for i in range(num_blocks):
        for j in range(i, num_blocks):
            if i == j:
                between_edges[(i, j)] = 0
            else:
                try:
                    between_edges[(i, j)] = np.random.randint(
                        0, param * min(block_degrees[i], block_degrees[j]))
                    block_degrees[i] -= between_edges[(i, j)]
                    block_degrees[j] -= between_edges[(i, j)]

                except ValueError:
                    between_edges[(i, j)] = 0

    return between_edges


# function for creating a Tree-Stochastic Block Model (T_SBM)

def create_t_sbm(num_blocks, num_nodes, p, q):
    """
    Create a Tree-Stochastic Block Model (T_SBM).

    Parameters
    ----------
    num_blocks : int
        Number of blocks.

    num_nodes : int
        Number of nodes per block.

    p : float
        Probability of an edge within a block.

    q : float
        Probability of an edge between nodes in different blocks.

    Returns
    -------
    G : NetworkX graph
        A T_SBM graph.
    """
    # create num_blocks trees with num_nodes nodes each
    G = nx.empty_graph(num_blocks * num_nodes)
    for i in range(num_blocks):
        H = nx.random_tree(num_nodes)

        # add the nodes and edges of H to G
        for j in range(num_nodes):
            G.add_node(i * num_nodes + j)
            for k in range(j + 1, num_nodes):
                if H.has_edge(j, k):
                    G.add_edge(i * num_nodes + j, i * num_nodes + k)

        # allocate the nodes to the blocks
        for j in range(num_nodes):
            G.nodes[i * num_nodes + j]['block'] = i

    # add edges within the blocks
    if p > 0:
        for i in range(num_blocks):
            for j in range(i * num_nodes, (i + 1) * num_nodes):
                for k in range(j + 1, (i + 1) * num_nodes):
                    if np.random.random() < p:
                        G.add_edge(j, k)

    # add edges between the blocks
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            for k in range(i * num_nodes, (i + 1) * num_nodes):
                for l in range(j * num_nodes, (j + 1) * num_nodes):
                    if np.random.random() < q:
                        G.add_edge(k, l)

    return G