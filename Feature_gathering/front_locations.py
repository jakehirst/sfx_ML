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
# def get_final_front_locations(path_to_folder, file, max_step, max_uci):
#     #path_to_folder = "C:\\Users\\u1056\\sfx\\simulation_results\\"
#     #max_s = get_max_step_and_max_UCIs()
#     front_locations = {}
#     filepath = path_to_folder + str(file) + "\\" + str(file) + "_Stp" + max_step + "_UCI_" + max_uci + "_original.frt"
#     front_locations = get_average_node_locations(filepath)
#     return front_locations




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
    # df = pd.DataFrame(sorted_list)
    # df.columns = ["step", "uci", "front 0 location", "front 1 location"]
    return sorted_list


""" gets euclidean distance between two 3d cartesian points """
def get_euclidean_distance(pt_a, pt_b):
    return m.sqrt((pt_a[0] - pt_b[0])**2 + (pt_a[1] - pt_b[1])**2 + (pt_a[2] - pt_b[2])**2)


""" gets approximate final front location for both fronts. cannot get exact due to the front locations resetting to the intiation site """
def get_unique_front_locations(simulation_folder, simulation):
    front_locations = get_all_front_locations(simulation_folder, simulation)
    front_0 = np.empty(0)
    front_1 = np.empty(0)

    front_0 = np.array(front_locations[0][2])
    front_1 = np.array(front_locations[0][3])
    for location in front_locations:
            front_0 = np.vstack([front_0, location[2]])
            front_1 = np.vstack([front_1, location[3]])
    
    #getting the unique front locations in the order that they are propogated
    indexes = np.unique(front_0, axis=0, return_index=True)[1]
    unique_front_0 = np.array([front_0[index] for index in sorted(indexes)])
    indexes = np.unique(front_1, axis=0, return_index=True)[1]
    unique_front_1 = np.array([front_1[index] for index in sorted(indexes)])


    #removing any huge jumps to different parts of the skull that are not supposed to be there
    temp = unique_front_0[0]
    i = 0
    for loc in unique_front_0:
        x = get_euclidean_distance(temp, loc)
        #print(x)
        if(x > 6):
            unique_front_0 = np.delete(unique_front_0, i, axis=0)
            continue
        temp = loc
        i+=1
    
    temp = unique_front_1[0]
    i = 0
    for loc in unique_front_1:
        x = get_euclidean_distance(temp, loc)
        #print(x)
        if(x > 6):
            unique_front_1 = np.delete(unique_front_1, i, axis=0)
            continue
        temp = loc
        i+=1

    return unique_front_0, unique_front_1

""" gets the total crack length based on euclidean distance between each front location """
def get_crack_len(simulation_folder, simulation):
    f = open(simulation_folder + simulation + "\\" + simulation + "_history.log")
    
    #replaces "crack_len_line" with the line with crack lengths in the log file all the way until the 
    # end of the log file to capture the final crack length line
    crack_len_line = ""
    for line in f.readlines():
        if(line.startswith("INFO:root:crack_length :")):
            crack_len_line = line

    crack_len = float(crack_len_line.split(",")[-1].replace("]]\n", "").replace("INFO:root:crack_length :  [[", ""))

    return crack_len



# simulation_folder = "C:\\Users\\u1056\\sfx\\good_simies\\"
# simulation = "Para_2-5ft_PHI_35_THETA_15"
# get_crack_len(simulation_folder, simulation)
