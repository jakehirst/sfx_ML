import sys
sys.path.append("C:\\Users\\u1056\\Code\\abafrank3\\")
sys.path.append('C:\\Users\\u1056\\Code')
sys.path.append('C:\\Users\\u1056\\Code\\abafrank3\\lib\\abafrank_lib')
import os
import numpy as np
import math as m

def get_max_dynamic(folder_path, key):
    sim_path = folder_path + key
    max_step = 0
    max_uci = 0
    for root, dirs, files in os.walk(sim_path):
        for file in files:
            if(file.startswith("Step") and file.__contains__("Dynamic")):
                uci = int(file.split("Step")[1].split("_")[2])
                step = int(file.split("Step")[1].split("_")[0])
                if(step > max_step):
                    max_step=step
                    max_uci = uci
                elif(step == max_step and max_uci < uci):
                    max_uci = uci
    return max_step, max_uci
        

def get_crack_locations(folder_path, key, max_step, max_uci):
    max_step, max_uci = get_max_dynamic(folder_path, key)
    filepath = folder_path + key + "\\" + key + "_Stp" + str(max_step) + "_UCI_" + str(max_uci) + "_full.inp"
    main_nodes = np.empty(0)
    mate_nodes = np.empty(0)
    main = False
    mate = False
    with open(filepath, 'r') as file:
        for row in file:
            if(row.startswith("*Nset,Nset=main_side_a")):
                main = True
                mate = False
                continue
            elif(row.startswith("*Nset,Nset=mate_side_a")):
                mate = True
                main = False
                continue
            elif(row.startswith("*Nset,Nset=front_nodes_0,unsorted")):
                main = False
                mate = False
                continue
            
            if(main == True):
                row = row.split(",")
                row.remove("\n")
                main_nodes = np.append(main_nodes, np.array(row))
            elif(mate == True):
                row = row.split(",")
                row.remove("\n")
                mate_nodes = np.append(mate_nodes, np.array(row))  

    locations = False
    i = 0
    j = 0
    main = []
    mate = []
    with open(filepath, 'r') as file:
        for row in file:
            if(row.startswith("*Node")):
                locations = True
                continue
            elif(row.startswith("*Element")):
                break

            if(locations):
                row = row.replace(" ","").split(",")
                if(main_nodes.__contains__(row[0])):
                    main.append(row)
                    i+=1
                elif(mate_nodes.__contains__(row[0])):
                    mate.append(row)
                    j+=1
    return main, mate
                
def max_width(main, mate):
    mins = get_min_distances(main, mate)
    max_width = max(mins)

    return max_width

def get_min_distances(main, mate):
    mins = []
    for node in main:
        min = []
        for other_node in mate:
            x1 = float(node[1]); y1 = float(node[2]); z1 = float(node[3])
            x2 = float(other_node[1]); y2 = float(other_node[2]); z2 = float(other_node[3])
            distance = m.sqrt((x2-x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            if(node[0] == '383360'):
                print(node)
                print(other_node)
                print("\n")
            if(min == []):
                min = [node, other_node, distance]
            elif(min[2] > distance):
                #print(distance)
                min = [node, other_node, distance]
        mins.append(min)
    return mins

def max_crack_width(folder_path, key, max_step, max_uci):
    main, mate = get_crack_locations(folder_path, key, max_step, max_uci)
    return max_width(main, mate)
            


def mean_crack_width(folder_path, key, max_step, max_uci):
    print("not implemented yet")


max_crack_width('C:\\Users\\u1056\\sfx\\good_simies\\','Para_1ft_PHI_50_THETA_290', '100', '0')