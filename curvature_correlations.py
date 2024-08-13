"""
curvature_correlations.py

Created on Feb 12 2023

@author: Lukas

This file contains all methods used to compute correlations between curvatures.
"""

# import packages

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import networkx.algorithms.community as nx_comm
import os
import json


def show_correlation_coeffs (h_data):
    """
    Show the correlation coefficients between the curvatures of a graph.

    Parameters
    ----------
    h_data : dict
        A dictionary containing the curvatures of a graph.

    Returns
    -------
    None.
    """
    def merged_list(ll):
        l = []
        for i in range(len(ll)):
            l.extend(ll[i])
        return l
    
    print("\nCorrelation coefficients:")
    curv_names = ["orc", "frc", "afrc", "afrc4", "afrc5"] 
    for i,cn in enumerate(curv_names):
        for j in range(i+1, len(curv_names)):
            s = h_data[cn]["title"] + " / " + h_data[curv_names[j]]["title"]
            c = np.corrcoef( merged_list(h_data[cn]["curv"]),  merged_list(h_data[curv_names[j]]["curv"]) ) [1][0]
            print(s.ljust(55,"."), f"{c:8.5f}")
        print()