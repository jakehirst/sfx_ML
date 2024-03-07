""" located in : ~simulation~_history.txt """
import numpy as np
import math as m
import glob
import os

def get_all_kink_angles(folder_path, key):
    kinks = {"front 0":{}, "front 1":{}}
    """ F:\Jake\good_simies\Para_3-0625ft_PHI_39_THETA_45\Para_3-0625ft_PHI_39_THETA_45_history.log """
    filename = folder_path + key + "\\" + key + "_history.log"
    f = open(filename,'r')

    kink_line = False
    for line in f.readlines():
        if(kink_line):
            angle = float(line.split(" ")[-2])
            step = line.split(" ")[-6]
            UcI = line.split(" ")[-4]
            kinks["front "+front]["step_" + step + "_uci_" + UcI] = angle

            kink_line = False

        elif(line.startswith("INFO:root:##############################                 FRANC 3D: On initiation site")):
            #print(line)
            if(line.endswith(" no crack growth occurred in this step.\n")):
                continue
            else:
                kink_line = True
                front = line.split(" ")[-1].replace(":\n", "")

    '''only keep the kink angles whose dynamic odb's are still in the simulation.'''
    pattern = os.path.join(folder_path + key, '*_Dynamic.odb')
    files = glob.glob(pattern)
    relevant_kinks = []
    for file in files:
        newstring = file.removesuffix('_Dynamic.odb').removeprefix(folder_path + key + "\\").lower()
        relevant_kinks.append(newstring.replace('step', 'step_'))
    kinks['front 0'] = {k: kinks['front 0'][k] for k in relevant_kinks if k in kinks['front 0']}
    kinks['front 1'] = {k: kinks['front 1'][k] for k in relevant_kinks if k in kinks['front 1']}
    
    return kinks


def get_max_kink(kinks):
    if(len(kinks["front 1"]) == 0 and len(kinks['front 0']) == 0):
        return 0
    if(len(kinks["front 1"]) == 0):
        max0 = np.max(np.abs(np.fromiter(kinks["front 0"].values(), dtype=float)))
        return max0
    if(len(kinks["front 0"]) == 0):
        max1 = np.max(np.abs(np.fromiter(kinks["front 1"].values(), dtype=float)))
        return max1

    else:
        max0 = np.max(np.abs(np.fromiter(kinks["front 0"].values(), dtype=float)))
        max1 = np.max(np.abs(np.fromiter(kinks["front 1"].values(), dtype=float)))
        themax = np.max([max0, max1])
        return themax


def get_abs_value_mean_kink(kinks):
    if(len(kinks["front 1"]) == 0 and len(kinks['front 0']) == 0):
        return 0
    if(len(kinks["front 1"]) == 0):
        mean0 = np.mean(np.abs(np.fromiter(kinks["front 0"].values(), dtype=float)))
        return mean0
    if(len(kinks['front 0']) == 0):
        mean1 = np.mean(np.abs(np.fromiter(kinks["front 1"].values(), dtype=float)))
        return mean1
    else:
        mean0 = np.mean(np.abs(np.fromiter(kinks["front 0"].values(), dtype=float)))
        mean1 = np.mean(np.abs(np.fromiter(kinks["front 1"].values(), dtype=float)))
        return np.mean([mean0, mean1])


def get_mean_kink(kinks):
    if(len(kinks["front 1"]) == 0 and len(kinks['front 0']) == 0):
        return 0
    if(len(kinks["front 1"]) == 0):
        mean0 = np.mean(np.fromiter(kinks["front 0"].values(), dtype=float))
        return mean0
    if(len(kinks['front 0']) == 0):
        mean1 = np.mean(np.fromiter(kinks["front 1"].values(), dtype=float))
        return mean1
    else:
        mean0 = np.mean(np.fromiter(kinks["front 0"].values(), dtype=float))
        mean1 = np.mean(np.fromiter(kinks["front 1"].values(), dtype=float))
        return np.mean([mean0, mean1])


def get_sum_of_abs_kink(kinks):
    if(len(kinks["front 1"]) == 0 and len(kinks['front 0']) == 0):
        return 0
    if(len(kinks["front 1"]) == 0):
        sum0 = np.sum(np.abs(np.fromiter(kinks["front 0"].values(), dtype=float)))
        return sum0
    if(len(kinks["front 0"]) == 0):
        sum1 = np.sum(np.abs(np.fromiter(kinks["front 1"].values(), dtype=float)))
        return sum1
    else:
        sum0 = np.sum(np.abs(np.fromiter(kinks["front 0"].values(), dtype=float)))
        sum1 = np.sum(np.abs(np.fromiter(kinks["front 1"].values(), dtype=float)))
        return sum0 + sum1


def get_sum_of_kink(kinks):
    if(len(kinks["front 1"]) == 0 and len(kinks['front 0']) == 0):
        return 0
    if(len(kinks["front 1"]) == 0):
        sum0 = np.sum(np.fromiter(kinks["front 0"].values(), dtype=float))
        return sum0
    if(len(kinks["front 0"]) == 0):
        sum1 = np.sum(np.fromiter(kinks["front 1"].values(), dtype=float))
        return sum1
    else:
        sum0 = np.sum(np.fromiter(kinks["front 0"].values(), dtype=float))
        sum1 = np.sum(np.fromiter(kinks["front 1"].values(), dtype=float))
        return sum0 + sum1
    
def get_var_of_kinks(kinks):
    front_0_kinks = np.fromiter(kinks["front 0"].values(), dtype=float)
    front_1_kinks = np.fromiter(kinks["front 1"].values(), dtype=float)
    all_kinks = np.append(front_0_kinks, front_1_kinks)
    var = np.var(all_kinks)
    if(m.isnan(var)):
        return 0
    else:
        return var
    
def get_std_of_kinks(kinks):
    front_0_kinks = np.fromiter(kinks["front 0"].values(), dtype=float)
    front_1_kinks = np.fromiter(kinks["front 1"].values(), dtype=float)
    all_kinks = np.append(front_0_kinks, front_1_kinks)
    std = np.std(all_kinks)
    if(m.isnan(std)):
        return 0
    else:
        return std

def get_median_of_kinks(kinks):
    front_0_kinks = np.fromiter(kinks["front 0"].values(), dtype=float)
    front_1_kinks = np.fromiter(kinks["front 1"].values(), dtype=float)
    all_kinks = np.abs(np.append(front_0_kinks, front_1_kinks))
    median = np.median(all_kinks)
    if(m.isnan(median)):
        return 0
    else:
        return median


def kink_angle_call(folder_path, key):
    kinks = get_all_kink_angles(folder_path, key)

    max = get_max_kink(kinks)
    abs_val_mean = get_abs_value_mean_kink(kinks)
    mean = get_mean_kink(kinks)
    sum = get_sum_of_kink(kinks)
    abs_val_sum = get_sum_of_abs_kink(kinks)
    median_kink = get_median_of_kinks(kinks)
    std_kinks = get_std_of_kinks(kinks)
    var_kinks = get_var_of_kinks(kinks)

    return max, abs_val_mean, mean, sum, abs_val_sum, median_kink, std_kinks, var_kinks

