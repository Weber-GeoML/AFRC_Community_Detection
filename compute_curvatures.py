"""
compute_curvatures.py

Created on Feb 12 2023

@author: Lukas

This file contains all methods to compute the curvature of a given graph.
"""

# import packages

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import networkx.algorithms.community as nx_comm
import os
import json
import scipy.sparse as sc

from GraphRicciCurvature.OllivierRicci import OllivierRicci


cyc_names = {3: "triangles", 4: "quadrangles", 5: "pentagons"}


def simple_cycles(G, limit):
    """
    Find simple cycles (elementary circuits) of a graph up to a given length.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    limit : int
        Maximum length of cycles to find plus one.

    Returns
    -------
    cycles : generator
        A generator that produces lists of nodes, one for each cycle.
    """
    subG = type(G)(G.edges())
    sccs = list(nx.strongly_connected_components(subG))
    while sccs:
        scc = sccs.pop()
        startnode = scc.pop()
        path = [startnode]
        blocked = set()
        blocked.add(startnode)
        stack = [(startnode, list(subG[startnode]))]

        while stack:
            thisnode, nbrs = stack[-1]

            if nbrs and len(path) < limit:
                nextnode = nbrs.pop()
                if nextnode == startnode:
                    yield path[:]
                elif nextnode not in blocked:
                    path.append(nextnode)
                    stack.append((nextnode, list(subG[nextnode])))
                    blocked.add(nextnode)
                    continue
            if not nbrs or len(path) >= limit:
                blocked.remove(thisnode)
                stack.pop()
                path.pop()
        subG.remove_node(startnode)
        H = subG.subgraph(scc)
        sccs.extend(list(nx.strongly_connected_components(H)))


def fr_curvature(G, ni, nj):
    """
    computes the Forman-Ricci curvature of a given edge

    Parameters
    ----------
    G : Graph
    ni : node i
    nj : node j

    Returns
    -------
    frc : int
        Forman Ricci curvature of the edge connecting nodes i and j

    """
    frc = 4 - G.degree(ni) - G.degree(nj)
    return frc


def afrc_3_curvature(G, ni, nj, t_num, t_coeff=3):
    """
    computes the Augmented Forman-Ricci curvature of a given edge
    includes 3-cycles in calculation

    Parameters
    ----------
    G : Graph

    ni : int
        node i

    nj : int
        node j

    m : int
        number of triangles containing the edge between node i and j

    Returns
    -------
    afrc : int
        Forman Ricci curvature of the edge connecting nodes i and j
    """
    afrc = 4 - G.degree(ni) - G.degree(nj) + t_coeff * t_num
    return afrc


def afrc_4_curvature(G, ni, nj, t_num, q_num, t_coeff=3, q_coeff=2):
    """
    computes the Augmented Forman-Ricci curvature of a given edge,
    includes 3- and 4-cycles in calculation

    Parameters
    ----------
    G : Graph

    ni : int
        node i

    nj : int
        node j

    t : int
        number of triangles containing the edge between node i and j

    q : int
        number of quadrangles containing the edge between node i and j

    Returns
    -------
    afrc4 : int
        enhanced Forman Ricci curvature of the edge connecting nodes i and j
    """
    afrc4 = 4 - G.degree(ni) - G.degree(nj) + t_coeff * t_num + q_coeff * q_num
    return afrc4


def afrc_5_curvature(G, ni, nj, t_num, q_num, p_num,
                     t_coeff=3, q_coeff=2, p_coeff=1):
    """
    computes the Augmented Forman-Ricci curvature of a given edge
    includes 3-, 4- and 5-cycles in calculation

    Parameters
    ----------
    G : Graph

    ni : int
        node i

    nj : int
        node j

    t : int
        number of triangles containing the edge between node i and j

    q : int
        number of quadrangles containing the edge between node i and j

    p : int
        number of pentagons containing the edge between node i and j

    Returns
    -------
    afrc5 : int
        enhanced Forman Ricci curvature of the edge connecting nodes i and j   
    """
    afrc5 = 4 - G.degree(ni) - G.degree(nj) + t_coeff * t_num + q_coeff * q_num + p_coeff * p_num
    return afrc5


def allocate_cycles_to_edges(G, ll, i):
    """
    allocate the number of cycles of length i to the edges of the graph G

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    ll : list
        list of cycles of length i

    i : int
        length of cycles

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

            G.edges[u, v][cyc_names[i]] += 1


def get_orc_edge_curvatures(G):
    """
    computes the Ollivier-Ricci curvature of all edges of a given graph G

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    None.
    """
    orc = OllivierRicci(G, alpha=0, verbose="ERROR")
    orc.compute_ricci_curvature()

    for (u, v) in list(orc.G.edges()):
        G.edges[u, v]["orc"] = orc.G.edges[u, v]["ricciCurvature"]


def AugFormanSq_semisparse(e, G):
    """
    Computes the augmented Forman-Ricci curvature of an edge e in a graph G

    Parameters
    ----------
    e : tuple
        edge of the graph G

    G : NetworkX graph
        An undirected graph.

    Returns
    -------
    FR : int
        augmented Forman-Ricci curvature of the edge e
    """
    E = np.zeros([len(G), len(G)]) # Matrix of edge contributions
    FR = 0

    # Add a -1 to the contribution of all edges sharing a node with e
    for i in (set(G[e[0]]) - {e[1]}):
        E[min(e[0], i)][max(e[0], i)] = -1
    for i in (set(G[e[1]]) - {e[0]}):
        E[min(e[1], i)][max(e[1], i)] = -1

    # Count triangles, and add +1 to the contribution of edges contained in a triangle with e
    T = len(set(G[e[0]]) & set(G[e[1]]))
    for i in (set(G[e[0]]) & set(G[e[1]])):
        E[min(e[0], i)][max(e[0], i)] += 1
        E[min(e[1], i)][max(e[1], i)] += 1

    # Count squares,
    # Add +1 to each edge neighbour to e contained in a square with it
    # Add +1 or -1 for edges not touching e contained in a square with it (the matrix lets us keep track of both orientations separately)
    Sq = 0
    neigh_0 = [i for i in G[e[0]] if i != e[1]]
    for i in neigh_0:         # for all neighbors of first node
        for j in set(G[i]) & set(G[e[1]]) - {e[0]}:   # for all nodes forming a square with first node, neighbors of first node, and second node
            Sq += 1
            E[min(e[0], i)][max(e[0], i)] += 1
            E[min(e[1], j)][max(e[1], j)] += 1
            E[i][j] += 1

    # Convert to sparse matrix for matrix multiplications
    csr = sc.csr_matrix(E)
    csr.eliminate_zeros()   # eliminate zeros

    csr_t = csr.transpose()     # transpose sparse matrix
    csr_res = csr - csr_t       # subtract tranposed from original matrix
    csr_sign = csr_res.sign()   # get sign for absolute value
    csr_res = csr_res.multiply(csr_sign)    # calculate absolute value by multiplication with sign of each cell

    FR -= csr_res.sum()         # summarize sparse matrix

    FR = int(FR / 2)            # divide by two, as both triangular matrices have been added above, we only need one
    FR += 2 + T + Sq

    return FR


""" NEED TO REPLACE THE CODE BELOW """

def AugFormanPent(e,G): # Returns pentagon-augmented curvature. Arguments: an edge e of a graph G
    
    E=np.zeros([len(G), len(G)]) #Matrix of edge contributions
    FR=0
   
    #Add a -1 to the contribution of all edges sharing a node with e
    for i in (set(G[e[0]]) - {e[1]}):
         E[min(e[0],i)][max(e[0],i)] = -1
    
    for i in (set(G[e[1]]) - {e[0]}):
         E[min(e[1],i)][max(e[1],i)] = -1
    
    #Count triangles, and add +1 to the contribution of edges contained in a triangle with e
    T=len(set(G[e[0]]) & set(G[e[1]]))

    
    for i in (set(G[e[0]]) & set(G[e[1]])):
        E[min(e[0],i)][max(e[0],i)] += 1
        E[min(e[1],i)][max(e[1],i)] += 1
    
    #Count squares,
    #Add +1 to each edge neighbour to e contained in a square with it
    #Add +1 or -1 for edges not touching e contained in a square with it (the matrix lets us keep track of both orientations separately)
    Sq=0
    neigh_0= [i for i in G[e[0]] if i!=e[1]]
    for i in neigh_0:
        for j in (set(G[i]) & set(G[e[1]]) - {e[0]}):
            Sq +=1
            E[min(e[0],i)][max(e[0],i)] += 1
            E[min(e[1],j)][max(e[1],j)] += 1
            E[i][j] += 1
    

    #Same as with squares, but for pentagons 
    Pent=0
    neigh_1= [i for i in G[e[1]] if i!=e[0]]
    for i in neigh_0:
        for j in neigh_1:
            if i != j:
                for k in (set(G[i]) & set(G[j]) - {e[0],e[1]}):
                    Pent += 1
                    E[min(e[0],i)][max(e[0],i)] += 1
                    E[min(e[1],j)][max(e[1],j)] += 1
                    E[i][k] += 1
                    E[k][j] += 1
    
    FR += 2 + T + Sq + Pent

    for i in range(len(G)):
        for j in range(i,len(G)):
            FR += -abs(E[i][j]-E[j][i])
            
    return FR
