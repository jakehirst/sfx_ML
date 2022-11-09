""" This contains functions that enable you to get the final crack front locations of a sfx simulation """
import os
import pandas as pd
import json
import numpy as np
import math as m
from initiation_sites import *



                    

""" gets the final front locations of a single simulation
path_to_folder example: C:\\Users\\u1056\\sfx\\simulation_results
file example: Para_1-5ft_PHI_30_THETA_230
max_step example: '16'
max_uci example: '8'
"""
def get_final_front_locations(path_to_folder, file, max_step, max_uci):
    #path_to_folder = "C:\\Users\\u1056\\sfx\\simulation_results\\"
    #max_s = get_max_step_and_max_UCIs()
    front_locations = {}
    filepath = path_to_folder + str(file) + "\\" + str(file) + "_Stp" + max_step + "_UCI_" + max_uci + "_original.frt"
    front_locations = get_average_node_locations(filepath)
    return front_locations




#TODO: this code needs to change if the multiple initiation site code is fixed
""" gets the average node locations of a front in cartesian coordinates 
filepath example = C:\\Users\\u1056\\sfx\\simulation_resultsPara_1-5ft_PHI_30_THETA_230\\Para_1-5ft_PHI_30_THETA_230_Stp16_UCI_8_original.frt
"""
def get_average_node_locations(filepath):
    reading_locations = False
    avg_locations = {}
    with open(filepath, 'r') as file:
        for row in file: 
            #print(row)
            if(row.startswith("OLD_FRT_PTS:")):
                frontnum = row.split(" ")[1] #defining the front 
                node_locations_x = []
                node_locations_y = []
                node_locations_z = []
                reading_locations = True
            elif(reading_locations == True and row.startswith(" ")):
                node_locations_x.append(float(row.split("          ")[1].split(" ")[-1]))
                node_locations_y.append(float(row.split("          ")[2]))
                node_locations_z.append(float(row.split("          ")[3].split("\\")[0]))
            elif(row.startswith("}")):
                avg_locations["front " + frontnum] = [sum(node_locations_x)/len(node_locations_x), sum(node_locations_y)/len(node_locations_y), sum(node_locations_z)/len(node_locations_z)]
                reading_locations = False
    #this needs to be changed if we ever get multiple initiation sites going
    if(len(avg_locations) == 1): #if there is only 1 front,(its an edge crack) 
        #then we can just use the initiation site as the location of the other front
        filename = filepath.split("\\")[-1]
        folder = filepath.removesuffix(filename)
        f = open(folder + folder.split("\\")[-2] + ".odb_initiation_info.json",'r')
        data = json.load(f)
        f.close()
        initiation_cite = data[1][0]
        avg_locations["front 1"] = initiation_cite
                
    return avg_locations


""" gets all of the front locations of all fronts at all time steps and UCI's in a single simulation
path_to_simulations example: "C:\\Users\\u1056\\sfx\\good_simies\\"
simulation example: "Para_3ft_PHI_55_THETA_250"
"""
def get_all_front_locations(path_to_simulations, simulation):
    simulation_folder = path_to_simulations + simulation
    front_locations = []
    for root, dirs, files in os.walk(simulation_folder):
        # select file name
            for file in files:
                # check the extension of files
                #print(file)
                if (file.startswith('Para') and file.__contains__("original") and file.endswith(".frt")):
                    location = get_average_node_locations(simulation_folder + "\\" + file)
                    step = int(file.split("Stp")[1].split("_")[0])
                    uci = int(file.split("Stp")[1].split("_")[2])
                    front_locations.append([step, uci, location["front 0"], location["front 1"]])

    sorted_list = sorted(front_locations, key=lambda x: (x[0], x[1])) #sorts the list of front locations by step, then by uci... soooo nice  
    return sorted_list


""" gets euclidean distance between two 3d cartesian points """
def get_euclidean_distance(pt_a, pt_b):
    return m.sqrt((pt_a[0] - pt_b[0])**2 + (pt_a[1] - pt_b[1])**2 + (pt_a[2] - pt_b[2])**2)




""" gets the total crack length based on euclidean distance between each front location """
def get_crack_len(simulation_folder, simulation):
    prev_front_0 = get_initiation_cite(simulation_folder, simulation)
    prev_front_1 = get_initiation_cite(simulation_folder, simulation)
    front_locations = get_all_front_locations(simulation_folder, simulation)
    crack_len_total = 0.0

    for locations in front_locations:
        dist_frt_0 = get_euclidean_distance(prev_front_0, locations[2])
        dist_frt_1 = get_euclidean_distance(prev_front_1, locations[3])
        crack_len_total += (dist_frt_0 + dist_frt_1)
        prev_front_0 = locations[2]
        prev_front_1 = locations[3]

    return crack_len_total



# simulation_folder = "C:\\Users\\u1056\\sfx\\good_simies\\"
# simulation = "Para_2-5ft_PHI_35_THETA_15"
# get_crack_len(simulation_folder, simulation)
