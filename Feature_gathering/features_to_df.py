import pandas as pd
from ast import literal_eval
import json
from front_locations import *
from initiation_sites import *
from linearity import *
from width_of_crack import *
# import tensorflow_probability as tfp
import scipy.stats
from thickness import *
from orientation import *
from kink_angle import *
import random 

FOLDER_PATH = "F:\\Jake\\good_simies\\"
# FOLDER_PATH = "Z:\\bjornssimies\\correcthistory\\"
# FOLDER_PATH = "Z:\\Brian_simies\\k_diff_simmies\\"
# FOLDER_PATH = "F:\\Jake\\new_good_simies\\"
# FOLDER_PATH = "Z:\\bjornssimies\\delta_k\\"
# FOLDER_PATH = "Z:\\Brian_simies\\k_diff_simmies\\"
# FOLDER_PATH = "C:\\Users\\u1056\\sfx\\simulation_results\\"

#gets the maximum steps and UCIs from the simulation_results folder
def get_max_step_and_max_UCIs(folder_path):
    # folder_path = FOLDER_PATH

    dic = {}
    for root, dirs, files in os.walk(folder_path):
        # select file name
        for file in files:
            # check the extension of files
            if (file.startswith('Para') and file.__contains__("original") and file.endswith(".frt")):
                simulation = file.split("_Stp")[0]
                step = file.split("_")[-4].split("p")[1]
                uci = file.split("_")[-2]

                if(dic.keys().__contains__(simulation)):
                    if((int(step) > int(dic[simulation][0])) or (int(uci) > int(dic[simulation][1]))):
                        dic[simulation] = [step, uci]
                else:
                    dic[simulation] = [step, uci]

    return dic

#turns a filename like Para_1-5ft_PHI_30_THETA_230 into [height, phi, theta]
def turn_filename_to_labels(filename):
    labels = []
    height = float(filename.split("_")[1].split("f")[0].replace("-","."))
    labels.append(height) #height
    labels.append(float(filename.split("_")[-3])) #phi
    labels.append(float(filename.split("_")[-1])) #theta
    return labels

""" finds the correlation between a given feature and a label, using Pearson Correlation. """
def Pearson_Correlation(feature, label, df):
    y = df[label]
    x = df[feature]

    #stuff from pearson correlation from before
    new_df = pd.concat([x, y], axis=1)
    r = new_df.corr()[feature][label]
    degfreedom = len(df) - 2
    t = r / (m.sqrt((1-r**2) / degfreedom ) )
    #stuff from pearson correlations from before

    return scipy.stats.pearsonr(x, y) #returns (pearson correlation, 2-tailed p value) #p value is supposed to show the significance
    # of a pearson correlation. A p value below 0.05 means the results is statistically significant

""" finds the pearson correlation of each feature and prints it out"""
def Pearson_Correlations_for_df(df, label):
    Correlations = dict()
    print("\n\n***** CORRELATION FOR " + label + " *****")
    for feature in df.columns.to_list():
        if(feature == "height" or feature == "phi" or feature =="theta" or feature =="x" or feature =="y" or feature =="z"):
            continue
        pc = Pearson_Correlation(feature, label, df)
        Correlations[feature + "/" + label] = pc
        print("\n"+ feature + "/" + label+" = " + str(pc))
    return Correlations

""" just turns phi and theta into cartesian points (with r being assumed to be 1) """
def PhiTheta_to_cartesian(df):
    phis = df["phi"]
    thetas = df["theta"]
    x = []
    y = []
    z = []
    for i in range(len(phis)):
        x.append(m.sin(phis[i]) * m.cos(thetas[i]))
        y.append(m.sin(phis[i]) * m.sin(thetas[i]))
        z.append(m.cos(phis[i]))
    
    df=df.drop("phi", axis=1)
    df=df.drop("theta", axis=1)
    cart = pd.DataFrame({'x': x,
                        'y': y,
                        'z': z})
    df = pd.concat([df, cart], axis=1)
    return df

""" saves the dataframe to the specified filepath"""
def save_df(df, filepath):
    df.to_csv(filepath)

def check_history_output(folder_path, key):
    file = open(folder_path + key + "\\" + key + "_history.log")
    for line in file.readlines():
        if(line.startswith("INFO:root:NAME :  Para")):
            if(not line.endswith(key + "\n")):
                print("BAD HISTORY OUTPUT FOR " + key)
                return
    return 


def create_df():
    maxes = get_max_step_and_max_UCIs(FOLDER_PATH)
    #TODO: add columns to dataframe as necessary
    columns = ["front 0 x", 
            "front 0 y", 
            "front 0 z", 
            "front 1 x", 
            "front 1 y", 
            "front 1 z",
            "init x", 
            "init y", 
            "init z",
            "dist btw frts", 
            "crack len",
            "linearity",
            "max thickness",
            "mean thickness",
            "max_kink",
            "abs_val_mean_kink",
            "mean_kink",
            "sum_kink",
            "abs_val_sum_kink",
            "avg_ori",
            "angle_btw",
            "height", 
            "phi", 
            "theta"]
    df = pd.DataFrame(columns=columns)
    folder_path = FOLDER_PATH

    front_0_array_x = []
    front_0_array_y = []
    front_0_array_z = []
    front_1_array_x = []
    front_1_array_y = []
    front_1_array_z = []
    initiation_cite_x = []
    initiation_cite_y = []
    initiation_cite_z = []
    distance_between_fronts = []
    crack_lengths = []
    linearity_arr = []
    height_array = []
    phi_array = []
    theta_array = []
    max_thickness_arr = []
    mean_thickness_arr = []
    max_kink_arr = []
    abs_val_mean_kink_arr = []
    mean_kink_arr = []
    sum_kink_arr = []
    abs_val_sum_kink_arr = []
    average_orientation_arr = []
    angle_between_cracks_arr = []

    i = 0
    #goes through each of the simulations and gathers features for the dataframe
    for key in maxes.keys():
        print(i)
        # print(key)
        #TODO: add feature gathering functions as necessary here

        #TODO: delete this below
        #max_crack_width(folder_path, key, maxes[key][0], maxes[key][1])
        #mean_crack_width(folder_path, key)
        #TODO: delete this above
        # if(key == 'Para_1-03ft_PHI_42_THETA_334'):
        #     continue
        # if(key == 'Para_1-1725ft_PHI_8_THETA_162'):
        #     print("here")
        # if(key == 'Para_1-2775ft_PHI_10_THETA_74'):
        #     print("here")

        good_history = check_history_output(folder_path, key)
 
        labels = turn_filename_to_labels(key)
        final_front_locations = get_final_front_locations(folder_path, key)
        crack_type = get_crack_type(folder_path, key)
        # final_front_locations = get_final_front_locations(folder_path, key, maxes[key][0], maxes[key][1]) #TODO delete this
        initiation_cite = get_initiation_cite(folder_path, key)
        average_orientation, angle_between_cracks = find_orientation(folder_path, key, final_front_locations, initiation_cite, crack_type)
        
        d = get_euclidean_distance(final_front_locations[0], final_front_locations[1])

        inner_surface_nodes, outer_surface_nodes, main_side_nodes, node_locations = get_main_side_outer_and_inner_surface_nodes(folder_path, key, get_max_dynamic_step(folder_path, key))
        len = get_crack_len(outer_surface_nodes, main_side_nodes, node_locations, final_front_locations)
        linearity = get_linearity(folder_path, key)
        max_kink, abs_val_mean_kink, mean_kink, sum_kink, abs_val_sum_kink = kink_angle_call(folder_path, key)
        max_thickness, mean_thickness = get_max_and_mean_thickness(folder_path, key)
        

        height_array.append(labels[0])
        phi_array.append(labels[1])
        theta_array.append(labels[2])
        front_0_array_x.append(final_front_locations[0][0])
        front_0_array_y.append(final_front_locations[0][1])
        front_0_array_z.append(final_front_locations[0][2])
        front_1_array_x.append(final_front_locations[1][0])
        front_1_array_y.append(final_front_locations[1][1])
        front_1_array_z.append(final_front_locations[1][2])
        initiation_cite_x.append(initiation_cite[0])
        initiation_cite_y.append(initiation_cite[1])
        initiation_cite_z.append(initiation_cite[2])
        distance_between_fronts.append(d)
        crack_lengths.append(len)
        linearity_arr.append(linearity)
        max_thickness_arr.append(max_thickness)
        mean_thickness_arr.append(mean_thickness)
        max_kink_arr.append(max_kink)
        abs_val_mean_kink_arr.append(abs_val_mean_kink)
        mean_kink_arr.append(mean_kink)
        sum_kink_arr.append(sum_kink)
        abs_val_sum_kink_arr.append(abs_val_sum_kink)
        average_orientation_arr.append(average_orientation)
        angle_between_cracks_arr.append(angle_between_cracks)

        i += 1
                


    df = {"front 0 x": front_0_array_x ,
            "front 0 y": front_0_array_y, 
            "front 0 z": front_0_array_z, 
            "front 1 x": front_1_array_x, 
            "front 1 y": front_1_array_y, 
            "front 1 z": front_1_array_z,
            "init x": initiation_cite_x, 
            "init y": initiation_cite_y, 
            "init z": initiation_cite_z, 
            "dist btw frts": distance_between_fronts, 
            "crack len": crack_lengths,
            "linearity": linearity_arr,
            "max thickness": max_thickness_arr,
            "mean thickness": mean_thickness_arr,
            "max_kink": max_kink_arr,
            "abs_val_mean_kink": abs_val_mean_kink_arr,
            "mean_kink": mean_kink_arr,
            "sum_kink": sum_kink_arr,
            "abs_val_sum_kink": abs_val_sum_kink_arr,
            "avg_ori": average_orientation_arr,
            "angle_btw": angle_between_cracks_arr,
            "height": height_array, 
            "phi": phi_array, 
            "theta": theta_array
            }
    df = pd.DataFrame(df,columns=columns)
    return df






df = create_df()
random_indicies = random.sample(range(1, len(df)), 30)
test_df = df.iloc[random_indicies]
train_df = df.drop(random_indicies, axis=0)
print(df)
Pearson_Correlations_for_df(df, "height")
Pearson_Correlations_for_df(df, "phi")
Pearson_Correlations_for_df(df, "theta")
save_df(test_df, "C:\\Users\\u1056\\sfx\\sfx_ML\\sfx_ML\\Feature_gathering\\TEST_OG_dataframe.csv")
save_df(train_df, "C:\\Users\\u1056\\sfx\\sfx_ML\\sfx_ML\\Feature_gathering\\TRAIN_OG_dataframe.csv")
save_df(df, "C:\\Users\\u1056\\sfx\\sfx_ML\\sfx_ML\\Feature_gathering\\FULL_OG_dataframe.csv")
df = PhiTheta_to_cartesian(df)
save_df(df, "C:\\Users\\u1056\\sfx\\sfx_ML\\sfx_ML\\Feature_gathering\\OG_dataframe_cartesian.csv")
Pearson_Correlations_for_df(df, "x")
Pearson_Correlations_for_df(df, "y")
Pearson_Correlations_for_df(df, "z")
print("done")



