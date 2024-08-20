""" *Nset, nset=ALL_MAIN_SIDE_NODES, instance=PART-1-1 """ #MAIN
""" *Nset, nset=ALL_MATE_SIDE_NODES, instance=PART-1-1 """ #MATE
""" *Nset, nset=INNER_SURF, instance=PART-1-1 """ #inner 
""" *Nset, nset=OUTER_SURF, instance=PART-1-1 """ #outer

# from features_to_df import *
import os
import pandas as pd
import json
import numpy as np
import math as m
from initiation_sites import *

""" sorts the dynamic steps by first step and then uci, and returns the last one"""
def get_max_dynamic_step(folder_path, simulation):
    steps_and_ucis = []
    for root, dirs, files in os.walk(folder_path+simulation):
        # select file name
            for file in files:
                # check the extension of files
                if (file.endswith("Dynamic.inp")):
                    step = int(file.split("_")[0].split("p")[1])
                    uci = int(file.split("_")[-2])
                    steps_and_ucis.append([step, uci])
    sorted_list = sorted(steps_and_ucis, key=lambda x: (x[0], x[1]))
    return sorted_list[-1]


""" gets the locations and nodes from the inner/outer surface of the skull as well as the main side of the crack. """
def get_main_side_outer_and_inner_surface_nodes(folder_path, simulation, max_step_uci):
    main_side_nodes = []
    outer_surface_nodes = []
    inner_surface_nodes = []
    node_locations = {}
    main = False
    outer = False
    inner = False
    locations = False

    f = open(folder_path + simulation + f"\\Step{max_step_uci[0]}_UCI_{max_step_uci[1]}_Dynamic.inp",'r')
    for line in f.readlines(): 
        #""" getting all node locations """
        if(locations and line.startswith("*Element")):
            locations = False
        if(locations):
            node_and_location = line.replace(" ", "").replace("\n", "").split(",")
            node_locations[node_and_location[0]] = [float(node_and_location[1]), float(node_and_location[2]), float(node_and_location[3])]        
        if(line.startswith("*Node") and len(node_locations.keys()) == 0):
            locations = True

        #""" getting the nodes on the mate side """
        if(main and line.startswith("*")):
            main = False
        if(main):
            nodes = line.replace(" ", "").replace("\n", "").split(",")
            main_side_nodes.extend(nodes)
        if(line.startswith("*Nset, nset=ALL_MAIN_SIDE_NODES, instance=PART-1-1")):
            main = True


        #""" getting the nodes on the outer side """
        if(outer and line.startswith("*")):
            outer = False
        if(outer):
            nodes = line.replace(" ", "").replace("\n", "").split(",")
            outer_surface_nodes.extend(nodes)
        if(line.startswith("*Nset, nset=OUTER_SURF, instance=PART-1-1")):
            outer = True

        #""" getting the nodes on the inner side """
        if(inner and line.startswith("*")):
            inner = False
        if(inner):
            nodes = line.replace(" ", "").replace("\n", "").split(",")
            inner_surface_nodes.extend(nodes)
        if(line.startswith("*Nset, nset=INNER_SURF, instance=PART-1-1")):
            inner = True
        if(main_side_nodes.__contains__("")):main_side_nodes.remove("")
        if(outer_surface_nodes.__contains__("")):outer_surface_nodes.remove("")
        if(inner_surface_nodes.__contains__("")):inner_surface_nodes.remove("")
    return inner_surface_nodes, outer_surface_nodes, main_side_nodes, node_locations

""" gets the thickness values along the crack based on the inner/outer surface and main side crack nodes"""
def get_thickness_along_crack(inner_surface_nodes, outer_surface_nodes, main_side_nodes, node_locations):
    inner_main_nodes = {}
    outer_main_nodes = {}
    opposite_nodes_and_distances = {}
    for main_node in main_side_nodes:
        if(inner_surface_nodes.__contains__(main_node)):
            inner_main_nodes[main_node] = node_locations[main_node]
        elif(outer_surface_nodes.__contains__(main_node)):
            outer_main_nodes[main_node] = node_locations[main_node]
    
    for inner_key in inner_main_nodes.keys():
        min_distance = m.inf
        min_distance_outer_node = ""
        for outer_key in outer_main_nodes.keys():
            d = get_euclidean_distance(inner_main_nodes[inner_key], outer_main_nodes[outer_key])
            if(d < min_distance):
                min_distance = d
                min_distance_outer_node = outer_key
        opposite_nodes_and_distances[inner_key + "_" + min_distance_outer_node] = min_distance
    
    return opposite_nodes_and_distances

""" gets the euclidean distance between two xyz values """
def get_euclidean_distance(A, B):
    return m.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2 + (A[2] - B[2])**2)

""" gets the mean of all the thicknesses """
def get_mean_thickness(opposite_nodes_and_distances):
    distances = np.fromiter(opposite_nodes_and_distances.values(), dtype=float)
    return np.mean(distances)

""" gets the max of all the thicknesses. also returns the two nodes that correspond to the max thickness"""
def get_max_thickness(opposite_nodes_and_distances):
    distances = np.fromiter(opposite_nodes_and_distances.values(), dtype=float)
    max = np.max(distances)
    max_nodes = list(opposite_nodes_and_distances.keys())[list(opposite_nodes_and_distances.values()).index(max)]
    return max, max_nodes

""" gets the max of all the thicknesses. also returns the two nodes that correspond to the max thickness"""
def get_median_thickness(opposite_nodes_and_distances):
    distances = np.fromiter(opposite_nodes_and_distances.values(), dtype=float)
    return np.median(distances)

""" gets the max of all the thicknesses. also returns the two nodes that correspond to the max thickness"""
def get_var_thickness(opposite_nodes_and_distances):
    distances = np.fromiter(opposite_nodes_and_distances.values(), dtype=float)
    return np.var(distances)

""" gets the max of all the thicknesses. also returns the two nodes that correspond to the max thickness"""
def get_std_thickness(opposite_nodes_and_distances):
    distances = np.fromiter(opposite_nodes_and_distances.values(), dtype=float)
    return np.std(distances)

""" gets the max and mean thickness along the crack """
def get_max_and_mean_thickness(folder_path, simulation):
    max_step_uci = get_max_dynamic_step(folder_path, simulation)
    inner_surface_nodes, outer_surface_nodes, main_side_nodes, node_locations = get_main_side_outer_and_inner_surface_nodes(folder_path, simulation, max_step_uci)
    opposite_nodes_and_distances = get_thickness_along_crack(inner_surface_nodes, outer_surface_nodes, main_side_nodes, node_locations)
    mean = get_mean_thickness(opposite_nodes_and_distances)
    max, max_nodes = get_max_thickness(opposite_nodes_and_distances)
    median = get_median_thickness(opposite_nodes_and_distances)
    var = get_var_thickness(opposite_nodes_and_distances)
    std = get_std_thickness(opposite_nodes_and_distances)

    return max, mean, median, var, std

'''gets the average thickness of the skull at the 10 closest nodes at a location'''
def get_thickness_at_location(location, outer_surface_nodes_df, inner_surface_nodes_df):
    '''efficiently gets the euclidean distance'''
    def euclidean_dist(row, point):
        return np.linalg.norm(np.array(row) - point)
    '''add the distances between the location and the nodes as a column'''
    new_outer_surface_nodes_df = outer_surface_nodes_df.copy()
    new_outer_surface_nodes_df['Distance_from_location'] = outer_surface_nodes_df['Coordinates'].apply(euclidean_dist, point=location)
    new_outer_surface_nodes_df_sorted = new_outer_surface_nodes_df.sort_values(by='Distance_from_location')
    '''Get top 10 closest nodes on the outer surface to the location'''
    top_10_min_distance_rows = new_outer_surface_nodes_df_sorted.head(10)
    '''gets the thicknesses of the skull at those nodes'''
    closest_inner_nodes = []
    closest_distances = []
    for index, row in top_10_min_distance_rows.iterrows():
        outer_coord = np.array(row['Coordinates'])
        inner_surface_nodes_df['Distance_to_outer_node'] = inner_surface_nodes_df['Coordinates'].apply(euclidean_dist, point=outer_coord)
        closest_inner_node = inner_surface_nodes_df.loc[inner_surface_nodes_df['Distance_to_outer_node'].idxmin()]
        closest_inner_nodes.append(int(closest_inner_node['Node']))
        closest_distances.append(closest_inner_node['Distance_to_outer_node'])
    '''get the average thickness of those nodes'''
    return np.average(closest_distances)

