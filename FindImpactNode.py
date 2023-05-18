import pandas as pd
import math as m
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import json


#gets the maximum steps and UCIs from the simulation_results folder
def get_min_step_and_min_UCIs(folder_path):
    #folder_path = "C:\\Users\\u1056\\sfx\\good_simies\\"
    #folder_path = "F:\\Jake\\good_simies\\"
    dic = {}
    for root, dirs, files in os.walk(folder_path):
        # select file name
        for file in files:
            #print(file)
            # check the extension of files
            if (file.startswith('Para') and file.endswith(".inp")):
                if(file.__contains__("LOCAL") or file.__contains__("GLOBAL") or file.__contains__("full")):
                    continue
                if(not file.__contains__("Stp")):
                    continue
                simulation = file.split("_Stp")[0]
                step = file.split("_")[-3].split("p")[1]
                uci = file.split("_")[-1].split(".")[0]
                if(dic.keys().__contains__(simulation)):
                    if((int(step) < int(dic[simulation][0])) or (int(uci) < int(dic[simulation][1]))):
                        dic[simulation] = [step, uci]
                else:
                    dic[simulation] = [step, uci]
    return dic

def get_distance(N1, N2):
    #print(N1)
    #print(N2)
    distance = ((N1[1] - N2[1])**2 + (N1[2] - N2[2])**2 + (N1[3] - N2[3])**2)**0.5
    return distance

def get_impact_node(file, min_steps_and_ucis):
    print("\nCHECKING "+ file)
    step = min_steps_and_ucis[file][0]
    uci = min_steps_and_ucis[file][1]

    Parietal_filename = folder_name + file + "\\"+ file + ".inp"
    Plate_filename = folder_name+file+f"\\"+file+f"_Stp{step}_UCI_{uci}_full.inp"

    data = []
    Skull_nodes = {}
    Plate_nodes = []
    current = []
    Parietal = False
    Plate = False
    part = False

    with open(Parietal_filename, 'r') as f1:
    #print(f.read()) # to read the entire file
        while(True):  
            line = str(f1.readline())
            if(line.startswith("*Part, name=")):
                current = []
                Part_name = line.split("=")[1]
                #print(Part_name)
                Skull_nodes[Part_name] = []
                part = True
                continue
            elif(part == True and line.startswith("*Element")):
                part = False
            
            if(part == True):
                if(line.startswith("*Node")):
                    continue
                
                #appending []
                #print(list(map(float, line.split(","))))
                Skull_nodes[Part_name].append(list(map(float, line.split(","))))

            if not line:
                break



    with open(Plate_filename, 'r') as f2:
    #print(f.read()) # to read the entire file
        while(True):  
            line = str(f2.readline())

            if(line.startswith("*Node")):
                Plate = True
                continue
            elif(Plate == True and list(map(float, line.split(",")))[0] == 129.0):
                Plate = False
                break

            if(Plate == True):
                # if(list(map(float, line.split(",")))[0] == 129.0):
                #     continue
                Plate_nodes.append(list(map(float, line.split(","))))
            if not line:
                break

    min_distance = [-1]
    node_combos = []
    for key in Skull_nodes.keys():
        if(key == "PL\n"):continue
        #print(key)
        for Plate_node in Plate_nodes:
            for Skull_node in Skull_nodes[key]:
                d = get_distance(Skull_node, Plate_node)
                if(min_distance[0] == -1 or min_distance[0] > d):
                    min_distance = [d, key, Skull_node, Plate_node, 0]
                elif(min_distance[0] != -1 and min_distance[0] == d):
                    if(not key == 'RPA1_5#PART-1\n'):
                        num_ties = min_distance[4]
                        min_distance = [d, key, Skull_node, Plate_node, 0]
                    min_distance[4] = num_ties + 1 #adding one to the number of ties for min distance
                    print(Skull_node)
                    print("OOOP THERES A TIE")
                

    part_of_skull = min_distance[1].split("\n")[0]


    return min_distance, part_of_skull, Skull_nodes


def get_all_impact_node_sites(folder_name):
    min_steps_and_ucis = get_min_step_and_min_UCIs(folder_name)
    min_distance_locations = {}
    impact_bones = {}

    for file in min_steps_and_ucis.keys():

        min_distance, part_of_skull,Skull_nodes = get_impact_node(file, min_steps_and_ucis)
        impact_bones[file] = part_of_skull
        min_distance_locations[file] = min_distance[2][1:]
        if(min_distance[1] != "RPA1_5#PART-1\n"):
            print("*****************  RED ALERT  *****************")
            print("check " + file)
            print(f"min distance = {min_distance[0]}")
            print(f"part of skull = {part_of_skull}")
            print(f"node num = {int(min_distance[2][0])}")
            print(f"num ties for min distance = {min_distance[4]}")
            print("moving on...")

        print(f"min distance = {min_distance[0]}")
        print(f"part of skull = {part_of_skull}")
        print(f"node num = {int(min_distance[2][0])}")
        print(f"num ties for min distance = {min_distance[4]}")
        print("moving on...")

    return min_distance_locations, Skull_nodes['RPA1_5#PART-1\n'], impact_bones

def plot_impact_sites(parietal_nodes, min_distance_locations):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the parietal points lightly
    ax.scatter(parietal_node_locations[:,0], parietal_node_locations[:,1], parietal_node_locations[:,2], color='blue', alpha=0.1, label='parietal_nodes')

    # Plot the impact points more heavily
    ax.scatter(min_distance_locations[:,0], min_distance_locations[:,1], min_distance_locations[:,2], color='red', alpha=0.3, label='impact_sites')

    # Show the plot
    plt.legend()
    # plt.show()
    plt.savefig('C:\\Users\\u1056\\sfx\\sfx_ML\\sfx_ML\\Feature_gathering\\impact_sites_image.png')
    plt.show()
    plt.close()


def save_impact_sites(min_distance_locations):
    folder = 'C:\\Users\\u1056\\sfx\\sfx_ML\\sfx_ML\\Feature_gathering\\'
    with open(folder + 'impact_sites.json', 'w') as f:
        json.dump(min_distance_locations, f)

def load_impact_sites(folder, filename):
    # open the file in read mode and load the JSON data into a dictionary
    with open(folder + filename, 'r') as f:
        my_dict = json.load(f)
    return my_dict

def add_impact_sites_to_df(df_folder, df_filename, impact_folder, impact_filename):
    df = pd.read_csv(df_folder + df_filename)
    impact_sites = load_impact_sites(impact_folder, impact_filename)
    impact_simulations = list(impact_sites.keys())
    # impact_location_arr = np.zeros((len(df),3))
    impact_location_arr = [''] * len(df)

    for simulation in impact_simulations:
        height = float(simulation.split('_')[1].replace('-','.').replace('ft',''))
        phi = float(simulation.split('_')[3])
        theta = float(simulation.split('_')[-1])

        index = df[(df['height'] == height) & (df['phi'] == phi) & (df['theta'] == theta)]
        impact_location_arr[index.index[0]] = str(impact_sites[simulation])
    df['impact_sites'] = impact_location_arr
    df.to_csv(df_folder + df_filename.replace('.csv', '_with_impact_sites.csv'))
    return df


folder_name = "F:\\Jake\\good_simies_coats\\"
folder_name = "C:\\Users\\u1056\\OneDrive\\Desktop\\Loyd_42_0_case\\delta_k_code\\"
folder_name = "C:\\Users\\u1056\\sfx\\impact_node_check\\"
folder_name = "F:\\Jake\\good_simies\\"

min_distance_locations, parietal_nodes, impact_bones = get_all_impact_node_sites(folder_name)
parietal_nodes = np.asarray(parietal_nodes)
parietal_node_locations = parietal_nodes[:,1:]
impact_nodes = np.array(list(min_distance_locations.values()))
plot_impact_sites(parietal_nodes, impact_nodes)

df = add_impact_sites_to_df('C:\\Users\\u1056\\sfx\\sfx_ML\\sfx_ML\\Feature_gathering\\', 'FULL_OG_dataframe.csv','C:\\Users\\u1056\\sfx\\sfx_ML\\sfx_ML\\Feature_gathering\\','impact_sites.json')

print(df)
print("done")