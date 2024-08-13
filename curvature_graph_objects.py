"""
curvature_graph_objects.py

Created on Feb 13 2023

@author: Lukas

This file contains the classes for the curvature graph objects.
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


# import functions from other scripts in the repository

import compute_curvatures as cc
import auxiliary_functions as af
import curvature_gaps as cg
import visualizations as vis
import community_detection as cd
import create_networks as cn


# define abstract class for curvature graphs

class CurvatureGraph(nx.Graph):
    """
    An abstract class that inherits from the networkx Graph class
    and adds additional attributes and methods related to graph curvature.
    Used to create subclasses for artificial and real graphs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cycles = {}
        self.curvature_gap = {}
        self.threshold = {}
        for edge in self.edges:
            self.edges[edge]["weight"] = 1
        self.pos = None


    def read_pos_from_json(self, filename):
        """
        Read positions from json file.

        Parameters
        ----------
        filename : str
            Name of the file to be read.

        Returns
        -------
        None.
            Attaches the positions to the attribute self.pos as a dictionary.
        """
        dict_pos = af.read_data_from_json_file(filename)
        dict_pos_array = af.pos_list_as_array(dict_pos)

        self.pos = {int(k): v  for (k, v) in iter(dict_pos_array.items())}


    def save_pos_to_json(self, filename):
        """
        Save positions to json file.

        Parameters
        ----------
        filename : str
            Name of the file to be saved.

        Returns
        -------
        None.
        """
        cwd = os.getcwd()
        full_filename = os.path.join(cwd, "pos_karate_club_graph.json")

        af.save_data_to_json_file(
            af.pos_array_as_list(self.pos), full_filename)


    def plot_curvature_graph(self, pos=None,
                            node_col="white", edge_lst=[],
                            edge_col="lightgrey", edge_lab={},
                            bbox=None, color_map="Set3",
                            alpha=1.0):
        """
        Plot the curvature graph.
        """
        vis.plot_my_graph(self, pos,
                          node_col, edge_lst,
                          edge_col, edge_lab,
                          bbox, color_map,
                          alpha)


    def plot_curvature_differences(self, curvature_difference):
        """
        Plot the graph with curvature differences on the edges.
        """
        vis.plot_curvature_differences(self, curvature_difference)


    def detect_louvain_communities(self):
        """
        Detect communities using the Louvain algorithm
        from networkx and store the detected communities
        in the attribute self.nodes[node]["louvain-community"].
        """
        # detect communities using the Louvain algorithm from networkx
        communities = nx_comm.greedy_modularity_communities(self)

        # store the detected communities in the attribute
        # self.nodes[node]["louvain-community"]
        for i, community in enumerate(communities):
            for node in community:
                self.nodes[node]["louvain_community"] = i


    def get_cycles(self):
        """
        Get the cycles of the graph.
        """
        all_cycles = []
        for cycle in cc.simple_cycles(self.to_directed(), 5): # need to change this back to 6 later
            all_cycles.append(cycle)

        self.cycles["triangles"] = [cycle for cycle in all_cycles
                                    if len(cycle) == 3]
        self.cycles["quadrangles"] = [cycle for cycle in all_cycles
                                      if len(cycle) == 4]
        self.cycles["pentagons"] = [cycle for cycle in all_cycles
                                    if len(cycle) == 5]


    def compute_frc(self, affected_edges=None):
        """
        Compute the Forman-Ricci curvature of the graph.
        """
        if affected_edges is None:
            affected_edges = self.edges()

        for edge in list(affected_edges):
            u, v = edge
            self.edges[edge]['frc'] = cc.fr_curvature(self, u, v)


    def compute_orc(self, affected_edges=None):
        """
        Compute the Ollivier-Ricci curvature of the graph.
        """
        # compute curvatures for all edges
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)

        # compute Ollivier-Ricci curvature
        cc.get_orc_edge_curvatures(G)
        for edge in self.edges:
            self.edges[edge]["orc"] = G.edges[edge]["orc"]


    def compute_afrc(self, affected_edges=None):
        """
        Compute the correct augmented Forman-Ricci curvature of the graph.
        """
        if affected_edges is None:
            affected_edges = self.edges()

        for edge in list(affected_edges):
            self.edges[edge]['afrc'] = cc.AugFormanSq_semisparse(edge, self)


    def compute_afrc_pentagons(self, affected_edges=None):
        """
        Compute the 5-cycle-augmented Forman-Ricci curvature of the graph.
        """
        if affected_edges is None:
            affected_edges = self.edges()

        for edge in list(affected_edges):
            self.edges[edge]['afrc_pent'] = cc.AugFormanPent(edge, self)



    def compute_afrc_3(self, affected_edges=None):
        """
        Compute the augmented Forman-Ricci curvature of the graph.
        """
        try:
            if affected_edges is None:
                affected_edges = self.edges()

            for edge in list(affected_edges):
                u, v = edge

                # compute curvature
                self.edges[edge]['afrc_3'] = cc.afrc_3_curvature(
                    self, u, v, t_num=self.edges[edge]["triangles"]
                    )

        except KeyError:
            print("Need to compute the number of triangles first.")
            self.count_triangles()
            self.compute_afrc_3()


    def compute_afrc_4(self, affected_edges=None):
        """
        Compute the augmented Forman-Ricci curvature of the graph.
        """
        try:
            if affected_edges is None:
                affected_edges = self.edges()

            for edge in list(affected_edges):
                u, v = edge

                # compute curvature
                self.edges[edge]['afrc_4'] = cc.afrc_4_curvature(
                    self, u, v, t_num=self.edges[edge]["triangles"],
                    q_num=self.edges[edge]["quadrangles"]
                    )

        except KeyError:
            print("Need to compute the number of triangles and quadrangles first.")
            self.count_triangles()
            self.count_quadrangles()
            self.compute_afrc_4()


    def compute_afrc_5(self, affected_edges=None):
        """
        Compute the augmented Forman-Ricci curvature of the graph.
        """
        try:
            if affected_edges is None:
                affected_edges = self.edges()

            for edge in list(affected_edges):
                u, v = edge

                # compute curvature
                self.edges[edge]['afrc_5'] = cc.afrc_5_curvature(
                    self, u, v, t_num=self.edges[edge]["triangles"],
                    q_num=self.edges[edge]["quadrangles"],
                    p_num=self.edges[edge]["pentagons"]
                    )

        except KeyError:
            print("Need to compute the number of triangles, quadrangles and pentagons first.")
            self.count_triangles()
            self.count_quadrangles()
            self.count_pentagons()
            self.compute_afrc_5()


    def count_triangles(self):
        """
        Count the number of triangles for each edge in the graph.
        """
        try:
            for edge in list(self.edges()):
                u, v = edge
                self.edges[edge]["triangles"] = len([cycle for cycle
                                                     in self.cycles["triangles"]
                                                     if u in cycle and v in cycle])/2

        except KeyError:
            print("Need to compute the cycles first.")
            self.get_cycles()
            self.count_triangles()


    def count_quadrangles(self):
        """
        Count the number of quadrangles for each edge in the graph.
        """
        try:
            for edge in list(self.edges()):
                u, v = edge
                self.edges[edge]["quadrangles"] = len([cycle for cycle
                                                       in self.cycles["quadrangles"]
                                                       if u in cycle and v in cycle])/2

        except KeyError:
            print("Need to compute the cycles first.")
            self.get_cycles()
            self.count_quadrangles()


    def count_pentagons(self):
        """
        Count the number of pentagons for each edge in the graph.
        """
        try:
            for edge in list(self.edges()):
                u, v = edge
                self.edges[edge]["pentagons"] = len([cycle for cycle
                                                     in self.cycles["pentagons"]
                                                     if u in cycle and v in cycle])/2

        except KeyError:
            print("Need to compute the cycles first.")
            self.get_cycles()
            self.count_pentagons()


    def compute_curvature_difference(self, curvature1, curvature2):
        """
        Compute the difference between two curvatures.
        """
        try:
            for edge in list(self.edges()):
                self.edges[edge]["diff_" + curvature1 + "_" + curvature2] = self.edges[edge][curvature1] - self.edges[edge][curvature2]

        except KeyError as error:
            if error.args[0] == "frc":
                print("Forman-Ricci curvature not found. Computing it now.")
                self.compute_frc()
                self.compute_curvature_difference(curvature1, curvature2)

            elif error.args[0] == "orc":
                print("Ollivier-Ricci curvature not found. Computing it now.")
                self.compute_orc()
                self.compute_curvature_difference(curvature1, curvature2)

            elif error.args[0] == "afrc":
                print("Augmented Forman-Ricci curvature not found. Computing it now.")
                self.compute_afrc()
                self.compute_curvature_difference(curvature1, curvature2)

            elif error.args[0] == "afrc_3":
                print("Augmented Forman-Ricci curvature not found. Computing it now.")
                self.compute_afrc_3()
                self.compute_curvature_difference(curvature1, curvature2)

            elif error.args[0] == "afrc_4":
                print("Augmented Forman-Ricci curvature not found. Computing it now.")
                self.compute_afrc_4()
                self.compute_curvature_difference(curvature1, curvature2)

            elif error.args[0] == "afrc_5":
                print("Augmented Forman-Ricci curvature not found. Computing it now.")
                self.compute_afrc_5()
                self.compute_curvature_difference(curvature1, curvature2)


    def compute_correlation(self, curvature1, curvature2):
        """
        Compute the correlation between two curvatures.
        """
        try:
            corr_coeff = np.corrcoef(
                [self.edges[edge][curvature1] for edge in self.edges],
                [self.edges[edge][curvature2] for edge in self.edges]
                )[0, 1]

            return corr_coeff

        except KeyError as error:
            if error.args[0] == "frc":
                print("Forman-Ricci curvature not found. Computing it now.")
                self.compute_frc()
                self.compute_correlation(curvature1, curvature2)

            elif error.args[0] == "orc":
                print("Ollivier-Ricci curvature not found. Computing it now.")
                self.compute_orc()
                self.compute_correlation(curvature1, curvature2)

            elif error.args[0] == "afrc":
                print("Augmented Forman-Ricci curvature not found. Computing it now.")
                self.compute_afrc()
                self.compute_correlation(curvature1, curvature2)


    def plot_curvature_histogram(self, curvature,
                                 title='No title', x_axis='x-axis',
                                 y_axis='y-axis', colors=False,
                                 font_size=30):
        """
        Plot a histogram of the values of a curvature.

        Parameters
        ----------
        curvature : str
            The curvature to plot. Can be "frc", "orc" or "afrc".

        title : str, optional
            The title of the plot. The default is ''.

        colors : bool, optional
            Whether to color the histogram by the edge affiliation (between or within communities). 
            The default is False.

        Returns
        -------
        None.
            Plots the histogram.
        """

        try:
            if colors:
                try:
                    vis.plot_curvature_hist_colors([
                        [self.edges[edge][curvature]
                         for edge in self.edges if self.edges[edge]["group"] == "within"],
                        [self.edges[edge][curvature]
                         for edge in self.edges if self.edges[edge]["group"] == "between"]],
                        title_str=title,
                        x_axis_str=x_axis,
                        y_axis_str=y_axis,
                        font_size=font_size
                    )

                except KeyError:
                    print("Need to compute the communities first.")

            else:
                vis.plot_curvature_hist(
                    [self.edges[edge][curvature] for edge in self.edges],
                    title=title,
                    x_axis_str=x_axis,
                    y_axis_str=y_axis,
                    font_size=font_size
                    )

        except KeyError as error:
            if error.args[0] == "frc":
                print("Forman-Ricci curvature not found. Computing it now.")
                self.compute_frc()
                self.plot_curvature_histogram(curvature, colors)

            elif error.args[0] == "orc":
                print("Ollivier-Ricci curvature not found. Computing it now.")
                self.compute_orc()
                self.plot_curvature_histogram(curvature, colors)

            elif error.args[0] == "afrc":
                print("Augmented Forman-Ricci curvature not found. Computing it now.")
                self.compute_afrc()
                self.plot_curvature_histogram(curvature, colors)


    def find_threshold(self, curv_name, cmp_key="block"):
        """
        Find the threshold for a given curvature distribution.
        Used in the community detection algorithms.

        Parameters
        ----------
        curv_name : str
            The curvature to use for the thresholding. Can be "frc", "orc" or "afrc".

        cmp_key : str, optional
            The key to use for comparison. The default is "block".

        Returns
        -------
        float
            The threshold for the curvature.
        """
        try:
            self.threshold[curv_name] = float(cg.find_threshold(self, curv_name, cmp_key))

        except KeyError as error:
            if error.args[0] == "frc":
                print("Forman-Ricci curvature not found. Computing it now.")
                self.compute_frc()
                self.find_threshold(curv_name, cmp_key)

            elif error.args[0] == "orc":
                print("Ollivier-Ricci curvature not found. Computing it now.")
                self.compute_orc()
                self.find_threshold(curv_name, cmp_key)

            elif error.args[0] == "afrc":
                print("Augmented Forman-Ricci curvature not found. Computing it now.")
                self.compute_afrc()
                self.find_threshold(curv_name, cmp_key)


    def detect_communities(self, curvature, threshold=0):
        """
        Detect communities using a curvature.

        Parameters
        ----------
        curvature : str
            The curvature to use for community detection. Can be "frc", "orc" or "afrc".

        threshold : float, optional
            The threshold for the curvature. The default is 0.

        Returns
        -------
        None.
            Adds the communities to the graph.
        """
        try:
            self = cd.detect_communities(self, curvature, threshold)

        except KeyError as error:
            if error.args[0] == "frc":
                print("Forman-Ricci curvature not found. Computing it now.")
                self.compute_frc()
                self.detect_communities(curvature, threshold)

            elif error.args[0] == "orc":
                print("Ollivier-Ricci curvature not found. Computing it now.")
                self.compute_orc()
                self.detect_communities(curvature, threshold)

            elif error.args[0] == "afrc":
                print("Augmented Forman-Ricci curvature not found. Computing it now.")
                self.compute_afrc()
                self.detect_communities(curvature, threshold)

            elif error.args[0] == "afrc_3":
                print("Augmented Forman-Ricci curvature (3) not found. Computing it now.")
                self.compute_afrc_3()
                self.detect_communities(curvature, threshold)

            elif error.args[0] == "afrc_4":
                print("Augmented Forman-Ricci curvature (4) not found. Computing it now.")
                self.compute_afrc_4()
                self.detect_communities(curvature, threshold)

            elif error.args[0] == "afrc_5":
                print("Augmented Forman-Ricci curvature (5) not found. Computing it now.")
                self.compute_afrc_5()
                self.detect_communities(curvature, threshold)


# define subclasses for artificial graphs

class CurvatureSBM(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for stochastic block models.
    """
    def __init__(self, l, k, p_in, p_out):
        sizes = af.build_size_list(k, l)
        probs = af.build_prob_list(l, p_in, p_out)

        super().__init__(nx.stochastic_block_model(sizes, probs, seed = np.random.randint(0, 1000)))


    def compute_curvature_gap(self, curv_name, key = "block"):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name, key)


    def assign_edges(self):
        """
        Assign edges to be between or within communities.
        """
        self = af.assign_edges(self, "block")


    def plot_curvature_graph(self, pos=None, node_col="white",
                             edge_lst=[], edge_col="lightgrey",
                             edge_lab={}, bbox=None, color_map="Set3", alpha=1):
        """
        Plot the graph with the nodes colored by their block affiliation.
        """
        if node_col == "white":
            node_col = [self.nodes[node]["block"] for node in self.nodes]

        super().plot_curvature_graph(pos, node_col,
                                     edge_lst, edge_col,
                                     edge_lab, bbox,
                                     color_map, alpha)


class CurvatureDC_SBM(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for degree-corrected stochastic block models.
    """
    def __init__(self, N, B, alpha, beta):
        """
        Initialize a degree-corrected stochastic block model.

        Parameters
        ----------
        N : int
            The number of nodes in the model.

        B : int
            The number of blocks in the model.

        alpha : float
            The parameter of the Poisson distribution from which the degrees are drawn.
            Must be non-negative.

        beta : float
            Parameter to regulate the number of edges between blocks, between 0 and 1.

        Returns
        -------
        G : nx.Graph
            A degree-corrected stochastic block model.
        """
        # assign N nodes randomly to B blocks in a list
        b = list(np.random.randint(0, B, N))

        # draw the degrees of the nodes from a Poisson distribution with parameter alpha
        k = list(np.random.poisson(alpha, N))
        print(k)

        # get the number of edges between blocks
        E = cn.determine_between_edges(b, k, B, beta)

        assert len(b) == len(k), "The length of b and k must be the same."

        # create a dictionary with blocks as keys, and lists of nodes and degrees as values
        block_dict = {block: [[], []] for block in range(B)}

        for node in range(len(b)):
            block_dict[b[node]][0].append(node)
            block_dict[b[node]][1].append(k[node])

        # initialize the adjacency matrix
        A = np.zeros((len(b), len(b)))

        for block_1 in range(B):
            for block_2 in range(block_1 + 1, B):
                assert block_1 < block_2, "The block_1 must be less than block_2."

                # get the degrees of the nodes in node_1 and node_2, store them in two lists
                k_1, k_2 = block_dict[block_1][1], block_dict[block_2][1]

                # get the number of edges between the two blocks
                num_edges = E[(block_1, block_2)]

                # draw the edges between the two blocks, update the degree lists
                edges_between, new_list_1, new_list_2 = af.get_edges_between_blocks(k_1, k_2, num_edges)

                # update the adjacency matrix
                for edge in edges_between:
                    A[block_dict[block_1][0][edge[0]], block_dict[block_2][0][edge[1]]] = 1
                    A[block_dict[block_2][0][edge[1]], block_dict[block_1][0][edge[0]]] = 1

                # update the degree lists in the dictionary
                block_dict[block_1][1] = new_list_1
                block_dict[block_2][1] = new_list_2

        # create the graph
        G = nx.from_numpy_array(A, parallel_edges=True, create_using=nx.MultiGraph())

        # for each degree sequence in the dictionary, create a graph according to the configuration model
        for block in range(B):
            # check if the sum of the degrees is odd
            print("Block:" + str(block))
            print("Degree distribution: " + str(block_dict[block][1]))

            if sum(block_dict[block][1]) % 2 == 1:
                # if so, increase the degree of a random node by 1
                block_dict[block][1][np.random.randint(0, len(block_dict[block][1]))] += 1

            # compose the graph with the configuration model
            H = nx.configuration_model(block_dict[block][1])
            print("Edges in the current block: " + str(H.edges))

            G = nx.compose(G, H)

        super().__init__(G)

        # assign the nodes to blocks
        for node in range(len(b)):
            self.nodes[node]["block"] = b[node]


    def plot_curvature_graph(self, pos=None,
                             node_col="white", edge_lst=[],
                             edge_col="lightgrey", edge_lab={},
                             bbox=None, color_map="Set3", alpha=1):
        """
        Plot the graph with the nodes colored by their block affiliation.
        """
        if node_col == "white":
            node_col = [self.nodes[node]["block"] for node in self.nodes]

        super().plot_curvature_graph(pos, node_col,
                                     edge_lst, edge_col,
                                     edge_lab, bbox, color_map, alpha)


    def compute_curvature_gap(self, curv_name, cmp_key = "community"):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name, cmp_key)


    def assign_edges(self):
        """
        Assign edges to be between or within communities.
        """
        self = af.assign_edges(self, "block")


class CurvatureER(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for Erdos-Renyi graphs.
    """
    def __init__(self, n, p):
        super().__init__(nx.erdos_renyi_graph(n, p))


    def assign_edges(self):
        """
        Assign edges to be between or within communities.
        """
        try:
            self = af.assign_edges(self, "louvain_community")

        except KeyError:
            print("No Louvain communities detected. Computing them now.")
            self.detect_louvain_communities()
            self.assign_edges()


    def compute_curvature_gap(self, curv_name):
        """
        Compute the curvature gap for the graph.
        """
        try:
            self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name, cmp_key = "louvain_community")

        except KeyError:
            print("No Louvain communities detected. Computing them now.")
            self.detect_louvain_communities()
            self.compute_curvature_gap(curv_name)


class CurvatureBG(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for bipartite graphs.
    """
    def __init__(self, n, p):
        super().__init__(nx.bipartite.random_graph(n, n, p, seed=0))


class CurvatureHBG(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for hierarchical bipartite graphs.
    """
    def __init__(self, n, m, p, q):
        super().__init__(af.get_bipartite_graph(n, m, p, q))


    def compute_curvature_gap(self, curv_name):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name, cmp_key = "community")


    def assign_edges(self):
        """
        Assign edges to be between or within communities.
        """
        for node in self.nodes:
            if self.nodes[node]["group"] == "A1" or self.nodes[node]["group"] == "B1":
                self.nodes[node]["community"] = 0
            else:
                self.nodes[node]["community"] = 1

        self = af.assign_edges(self, "community")


    def plot_curvature_graph(self, node_col=None):
        """
        Plot the graph with the nodes colored by their block affiliation.
        """
        top_nodes = {n for n, d in self.nodes(data=True) if d["bipartite"] == 0}
        bottom_nodes = set(self) - top_nodes

        pos = nx.bipartite_layout(self, top_nodes)

        if node_col is None:
            node_col=[["A1", "A2", "B1", "B2"].index(d["group"])
                                    for n, d in self.nodes.data()]

        vis.plot_my_graph(self, pos,
                          node_col=node_col, edge_lst=[],
                          edge_col=[["pink", "lightgrey"][d["prob"]]
                                    for u, v, d in self.edges.data()],
                          edge_lab={}, bbox=None,
                          color_map="tab20", alpha=0.7)
        

class CurvatureT_SBM(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for SBMs whose blocks are trees.
    """
    def __init__(self, l, k, p, q):
        super().__init__(cn.create_t_sbm(l, k, p, q))


    def compute_curvature_gap(self, curv_name):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name)


    def assign_edges(self):
        """
        Assign edges to be between or within communities.
        """
        self = af.assign_edges(self, "block")


    def plot_curvature_graph(self, pos=None, node_col="white",
                             edge_lst=[], edge_col="lightgrey",
                             edge_lab={}, bbox=None, color_map="Set3", alpha=1):
        """
        Plot the graph with the nodes colored by their block affiliation.
        """
        if node_col == "white":
            node_col = [self.nodes[node]["block"] for node in self.nodes]

        super().plot_curvature_graph(pos, node_col,
                                     edge_lst, edge_col,
                                     edge_lab, bbox,
                                     color_map, alpha)


# define subclasses for real graphs

class CurvatureKarate(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for the karate club graph.
    """
    def __init__(self):
        super().__init__(nx.karate_club_graph())


    def compute_curvature_gap(self, curv_name):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name, cmp_key="club")


    def assign_edges(self):
        """
        Assign edges to be between or within communities.
        """
        self = af.assign_edges(self, "club")


class CurvatureAMF(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for the American football graph.
    """
    def __init__(self):
        super().__init__(nx.read_gml("Network Models/football.gml"))

        # relabel node names with integers using nx.relabel_nodes
        mapping = dict(zip(self, range(len(self.nodes))))
        self = nx.relabel_nodes(self, mapping, copy=False)


    def compute_curvature_gap(self, curv_name):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name, cmp_key = "value")


    def assign_edges(self):
        """
        Assign edges to be between or within communities.
        """
        self = af.assign_edges(self, "value")


    def plot_curvature_graph(self, pos=None,
                             node_col="white", edge_lst=[],
                             edge_col="lightgrey", edge_lab={},
                             bbox=None, color_map="Set3", alpha=1):
        """
        Plot the AMF graph using the cycle-of-cycles layout.
        """
        if pos is None:
            values = set([d["value"]  for n,d in iter(self.nodes.items())])
            max_value = list(values)[-1]

            pos = af.get_amf_pos(self, 0.1, 0.6, 0.5, 0.5, max_value, values)

        vis.plot_my_graph(self, pos,
                          node_col=node_col, edge_lst=edge_lst,
                          edge_col=edge_col, edge_lab=edge_lab,
                          bbox=bbox, color_map=color_map, alpha=alpha)


class CurvatureDolphins(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for the dolphins graph.
    """
    def __init__(self):
        super().__init__(nx.read_gml("Network Models/dolphins.gml"))

        # relabel node names with integers using nx.relabel_nodes
        mapping = dict(zip(self, range(len(self.nodes))))
        self = nx.relabel_nodes(self, mapping, copy=False)


    def assign_edges(self):
        """
        Assign edges to be between or within communities.
        """
        try:
            self = af.assign_edges(self, "louvain_community")

        except KeyError:
            print("No Louvain communities detected. Computing them now.")
            self.detect_louvain_communities()
            self.assign_edges()


    def compute_curvature_gap(self, curv_name):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name, cmp_key = "louvain_community")


class CurvatureUSPowerGrid(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for the US power grid graph.
    """
    def __init__(self):
        super().__init__(nx.read_gml("Network Models/power.gml", label='id'))


    def assign_edges(self):
        """
        Assign edges to be between or within communities.
        """
        try:
            self = af.assign_edges(self, "louvain_community")

        except KeyError:
            print("No Louvain communities detected. Computing them now.")
            self.detect_louvain_communities()
            self.assign_edges()


    def compute_curvature_gap(self, curv_name):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name, cmp_key = "louvain_community")


class CurvatureWordAdjacency(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for the word adjacency graph.
    """
    def __init__(self):
        super().__init__(nx.read_gml("Network Models/adjnoun.gml"))

        # relabel node names with integers using nx.relabel_nodes
        mapping = dict(zip(self, range(len(self.nodes))))
        self = nx.relabel_nodes(self, mapping, copy=False)


    def assign_edges(self):
        """
        Assign edges to be between or within communities.
        """
        try:
            self = af.assign_edges(self, "louvain_community")

        except KeyError:
            print("No Louvain communities detected. Computing them now.")
            self.detect_louvain_communities()
            self.assign_edges()


    def compute_curvature_gap(self, curv_name):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name, cmp_key = "louvain_community")


class CurvatureScotland(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for
    the Corporate interlocks in Scotland graph.
    """
    def __init__(self):
        txt = open("Network Models/Scotland.net").readlines()
        G = nx.parse_pajek(txt)
        super().__init__(G)

        # relabel node names with integers using nx.relabel_nodes
        mapping = dict(zip(self, range(len(self.nodes))))
        self = nx.relabel_nodes(self, mapping, copy=False)


    def plot_curvature_graph(self, node_col="blue", ):
        """
        Plot the Southern Women network.
        """
        top_nodes = {n for n in self.nodes() if n <= 108}
        bottom_nodes = set(self) - top_nodes

        pos = nx.bipartite_layout(self, top_nodes)

        vis.plot_my_graph(self, pos,
                          node_col, edge_lst=[],
                          edge_col="lightgrey", edge_lab={},
                          bbox=None, color_map="tab20",
                          alpha=0.7)


    def assign_edges(self):
        """
        Assign edges to be between or within communities.
        """
        try:
            self = af.assign_edges(self, "louvain_community")

        except KeyError:
            print("No community information found. Assigning node communities using Louvain.")
            self.detect_louvain_communities()
            self = af.assign_edges(self, "louvain_community")


    def compute_curvature_gap(self, curv_name):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name, cmp_key = "louvain_community")


class CurvatureSouthernWomen(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for the Southern Women graph
    """
    def __init__(self):

        # read in the data
        filename = "Network Models/out.opsahl-southernwomen"
        cwd = os.getcwd()
        full_filename = os.path.join(cwd, filename)

        fobj = open(full_filename)
        lines = []
        for line in fobj:
            lines.append(line)

        fobj.close()

        lines = lines[2:]

        # create the graph
        edge_nodes = [[int(s) for s in line.split()] for line in lines]
        edge_list = [(edge[0] - 1, edge[1] + 17) for edge in edge_nodes]

        G = nx.Graph()
        G.add_edges_from(edge_list)

        # initialize the graph object
        super().__init__(G)


    def plot_curvature_graph(self, node_col="blue", ):
        """
        Plot the Southern Women network.
        """
        top_nodes = {n for n in self.nodes() if n <= 18}
        bottom_nodes = set(self) - top_nodes

        pos = nx.bipartite_layout(self, top_nodes)

        vis.plot_my_graph(self, pos,
                          node_col, edge_lst=[],
                          edge_col="lightgrey", edge_lab={},
                          bbox=None, color_map="tab20",
                          alpha=0.7)
        

    def assign_edges(self):
        """
        Assign edges to be between or within communities.
        """
        try:
            self = af.assign_edges(self, "louvain_community")

        except KeyError:
            print("No community information found. Assigning node communities using Louvain.")
            self.detect_louvain_communities()
            self = af.assign_edges(self, "louvain_community")


    def compute_curvature_gap(self, curv_name):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name, cmp_key = "louvain_community")


class CurvatureSchoolDay1(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for the School Day 1 graph
    """
    def __init__(self):
        super().__init__(nx.read_gexf("Network Models/school_day_1.gexf"))

        # relabel node names with integers using nx.relabel_nodes
        mapping = dict(zip(self, range(len(self.nodes))))
        self = nx.relabel_nodes(self, mapping, copy=False)


    def assign_edges(self):
        """
        Assign edges to be between or within communities.
        """
        self = af.assign_edges(self, "classname")


    def compute_curvature_gap(self, curv_name):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name, cmp_key = "classname")


class CurvatureSchoolDay2(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for the School Day 2 graph
    """
    def __init__(self):
        super().__init__(nx.read_gexf("Network Models/school_day_2.gexf"))

        # relabel node names with integers using nx.relabel_nodes
        mapping = dict(zip(self, range(len(self.nodes))))
        self = nx.relabel_nodes(self, mapping, copy=False)


    def assign_edges(self):
        """
        Assign edges to be between or within communities.
        """
        self = af.assign_edges(self, "classname")


    def compute_curvature_gap(self, curv_name):
        """
        Compute the curvature gap for the graph.
        """
        self.curvature_gap[curv_name] = cg.compute_curvature_gap(self, curv_name, cmp_key = "classname")


class PyGCurvatureGraph(CurvatureGraph):
    """
    A subclass of CurvatureGraph specifically for graphs
    from the Pytorch Geometric package.
    """
    def __init__(self, graph):
        """
        Initialize the graph object.
        """
        super().__init__(graph)

        # relabel node names with integers using nx.relabel_nodes
        mapping = dict(zip(self, range(len(self.nodes))))
        self = nx.relabel_nodes(self, mapping, copy=False)