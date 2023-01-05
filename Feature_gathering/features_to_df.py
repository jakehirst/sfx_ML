import pandas as pd
from ast import literal_eval
import json
from front_locations import *
from initiation_sites import *
from linearity import *
from width_of_crack import *
import tensorflow_probability as tfp
import scipy.stats

#gets the maximum steps and UCIs from the simulation_results folder
def get_max_step_and_max_UCIs():
    #folder_path = "C:\\Users\\u1056\\sfx\\good_simies\\"
    folder_path = "F:\\Jake\\good_simies\\"
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

def create_df():
    maxes = get_max_step_and_max_UCIs()
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
            "height", 
            "phi", 
            "theta"]
    df = pd.DataFrame(columns=columns)
    # folder_path = "C:\\Users\\u1056\\sfx\\good_simies\\"
    folder_path = "F:\\Jake\\good_simies\\"
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

    #goes through each of the simulations and gathers features for the dataframe
    for key in maxes.keys():
        #TODO: add feature gathering functions as necessary here

        #TODO: delete this below
        #max_crack_width(folder_path, key, maxes[key][0], maxes[key][1])
        #mean_crack_width(folder_path, key)
        #TODO: delete this above

        labels = turn_filename_to_labels(key)
        final_front_locations = get_final_front_locations(folder_path, key)
        #final_front_locations = get_final_front_locations(folder_path, key, maxes[key][0], maxes[key][1])
        initiation_cite = get_initiation_cite(folder_path, key)
        d = get_euclidean_distance(final_front_locations[0], final_front_locations[1])
        len = get_crack_len(folder_path, key)
        linearity = get_linearity(folder_path, key)

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
            "height": height_array, 
            "phi": phi_array, 
            "theta": theta_array}
    df = pd.DataFrame(df,columns=columns)
    return df






df = create_df()
print(df)
Pearson_Correlations_for_df(df, "height")
Pearson_Correlations_for_df(df, "phi")
Pearson_Correlations_for_df(df, "theta")
save_df(df, "C:\\Users\\u1056\\sfx\\ML\\Feature_gathering\\OG_dataframe.csv")
df = PhiTheta_to_cartesian(df)
save_df(df, "C:\\Users\\u1056\\sfx\\ML\\Feature_gathering\\OG_dataframe_cartesian.csv")
# Pearson_Correlations_for_df(df, "x")
# Pearson_Correlations_for_df(df, "y")
# Pearson_Correlations_for_df(df, "z")

