import pandas as pd
import math as m
import os


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


folder_name = "F:\\Jake\\good_simies_coats\\"

min_steps_and_ucis = get_min_step_and_min_UCIs(folder_name)


for file in min_steps_and_ucis.keys():
    print("\nCHECKING "+ file)
    step = min_steps_and_ucis[file][0]
    uci = min_steps_and_ucis[file][1]

    Parietal_filename = folder_name + file + "\\"+ file + ".inp"
    Plate_filename = folder_name+file+f"\\"+file+f"_Stp{step}_UCI_{uci}_full.inp"


# ft = "1-5"
# phi = 0
# theta = 0
# step = 20 #the lowest possible step outside of step 0 of INP files

# Parietal_filename = f"F:\\Jake\\good_simies\\Para_{ft}ft_PHI_{phi}_THETA_{theta}\\Para_{ft}ft_PHI_{phi}_THETA_{theta}.inp"

# Plate_filename = f"F:\\Jake\\good_simies\\Para_{ft}ft_PHI_{phi}_THETA_{theta}\\Para_{ft}ft_PHI_{phi}_THETA_{theta}_Stp{step}_UCI_0_full.inp"

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


