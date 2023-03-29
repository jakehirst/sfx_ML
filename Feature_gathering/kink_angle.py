""" located in : ~simulation~_history.txt """
import numpy as np
import math as m

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
            if(line.endswith(" no crack growth occurred in this step.\n")):
                continue
            else:
                kink_line = True
                front = line.split(" ")[-1].replace(":\n", "")

    
    return kinks


def get_max_kink(kinks):
    if(len(kinks["front 1"]) == 0):
        return 0
    max0 = np.max(np.abs(np.fromiter(kinks["front 0"].values(), dtype=float)))
    if(len(kinks["front 1"]) == 0):
        return max0
    else:
        max1 = np.max(np.abs(np.fromiter(kinks["front 1"].values(), dtype=float)))
        themax = np.max([max0, max1])
        return themax


def get_abs_value_mean_kink(kinks):
    if(len(kinks["front 1"]) == 0):
        return 0
    mean0 = np.mean(np.abs(np.fromiter(kinks["front 0"].values(), dtype=float)))
    if(len(kinks["front 1"]) == 0):
        return mean0
    else:
        mean1 = np.mean(np.abs(np.fromiter(kinks["front 1"].values(), dtype=float)))
        return np.mean([mean0, mean1])


def get_mean_kink(kinks):
    if (len(kinks["front 1"]) == 0):
        return 0
    mean0 = np.mean(np.fromiter(kinks["front 0"].values(), dtype=float))
    if(len(kinks["front 1"]) == 0):
        return mean0
    else:
        mean1 = np.mean(np.fromiter(kinks["front 1"].values(), dtype=float))
        return np.mean([mean0, mean1])


def get_sum_of_abs_kink(kinks):
    if(len(kinks["front 1"]) == 0):
        return 0
    sum0 = np.sum(np.abs(np.fromiter(kinks["front 0"].values(), dtype=float)))
    if(len(kinks["front 1"]) == 0):
        return sum0
    else:
        sum1 = np.sum(np.abs(np.fromiter(kinks["front 1"].values(), dtype=float)))
        return sum0 + sum1


def get_sum_of_kink(kinks):
    if(len(kinks["front 1"]) == 0):
        return 0
    sum0 = np.sum(np.fromiter(kinks["front 0"].values(), dtype=float))
    if(len(kinks["front 1"]) == 0):
        return sum0
    else:
        sum1 = np.sum(np.fromiter(kinks["front 1"].values(), dtype=float))
        return sum0 + sum1



def kink_angle_call(folder_path, key):
    kinks = get_all_kink_angles(folder_path, key)

    max = get_max_kink(kinks)
    abs_val_mean = get_abs_value_mean_kink(kinks)
    mean = get_mean_kink(kinks)
    sum = get_sum_of_kink(kinks)
    abs_val_sum = get_sum_of_abs_kink(kinks)

    return max, abs_val_mean, mean, sum, abs_val_sum

