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
from FindImpactNode import *
import ast
from datetime import date
import re
import fnmatch
import json


current_date = date.today()
current_date_string = current_date.strftime("%Y_%m_%d")


#material basis vectors for RPA bone
Material_X = np.array([-0.87491124, -0.44839274,  0.18295974])
Material_Y = np.array([ 0.23213791, -0.71986519, -0.65414532])
Material_Z = np.array([ 0.42502036, -0.5298472,   0.7339071 ])
#Center of mass of the RPA bone in abaqus basis
CM = np.array([106.55,72.79,56.64])
#Ossification center of the RPA bone in abaqus basis
OC = np.array([130.395996,46.6063,98.649696])

# FOLDER_PATH = "F:\\Jake\\good_simies\\"
# FOLDER_PATH = "F:\\Jake\\good_simies_new_R_curve_j_3.5_q_2.5\\"
# FOLDER_PATH = "F:\\Jake\\good_simies_old_R_curve\\" #making data from old simulations so that we can delete them
# FOLDER_PATH = "F:\\Jake\\toy_dataset\\"

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

'''
Inserts a column of data (new_column_data) into the df after the column_before column and it 
calls the new column 'new_column_name'
Returns the new dataframe.
'''
def insert_column_into_df(df, column_before, new_column_name, new_column_data):
    df.insert(loc=df.columns.get_loc(column_before) , column=new_column_name, value=new_column_data)
    return df

''' 
Converts the xs ys and zs from the ABAQUS basis into the basis that is centered at the center of mass of the skull CM, and 
the z axis goes through the the ossification site of the RPA bone. These new basis vectors are defined above as Material X, Y, and Z.
Returns the x, y and z values in the new basis.
'''
def convert_coordinates_to_new_basis(Material_X, Material_Y, Material_Z, CM, xs, ys, zs):
    Transform = np.linalg.inv( np.matrix(np.transpose([Material_X,Material_Y,Material_Z])) )
    x2 = []
    y2 = []
    z2 = []
    for i in range(len(xs)):
        #the old point is in reference to the center of mass of the skull, defined as CM
        old_point = np.array([xs[i], ys[i], zs[i]]) - CM 
        #using transformation matrix to go from abaqus basis to the material basis
        new_point = np.array( Transform * np.matrix(old_point.reshape((3,1))) ).reshape((1,3)) 
        x2.append(new_point[0][0])
        y2.append(new_point[0][1])
        z2.append(new_point[0][2])

    print("Coordinates in the target basis:")
    print(f"x2: {x2}, y2: {y2}, z2: {z2}")
        
    return np.array(x2), np.array(y2), np.array(z2)

'''
Converts x, y and z values into sphereical coordinates r, phi, and theta.
Returns r, phi and theta values 
'''
def convert_cartesian_to_spherical(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arccos(z / r)
    theta = np.arctan2(y, x)
    theta = np.degrees(theta)
    return r, np.degrees(phi) , (theta % 360 + 360) % 360


def check_history_output(folder_path, key):
    file = open(folder_path + key + "\\" + key + "_history.log")
    for line in file.readlines():
        if(line.startswith("INFO:root:NAME :  Para")):
            if(not line.endswith(key + "\n")):
                print("BAD HISTORY OUTPUT FOR " + key)
                return
    return 

''' converts all of the node locations and the node numbers into a single dataframe that only contains the nodes in the node_numbers and their locations. '''
def convert_node_locations_to_df(node_locations, node_numbers):
    df = {k: v for k, v in node_locations.items() if k in node_numbers}
    df = pd.DataFrame(list(df.items()), columns=['Node', 'Coordinates'])
    df[['X', 'Y', 'Z']] = pd.DataFrame(df['Coordinates'].to_list(), index=df.index)
    return df

''' This orders the file names that have a step and UCI first by the step, and then by the UCI. find_Simsetting_json_files below shows how it is used.'''
def extract_numbers(filename):
    # This regular expression looks for two groups of numbers in the filename
    match = re.search(r'Stp(\d+).*_(\d+)\.json', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        return (float('inf'), float('inf'))  # Return a tuple of infinity if no match is found

''' finds the Json files that look like this in a given simulation folder: SimSettings_Para_3-67ft_PHI_21_THETA_226_Stp4_UCI_1
    it also orders them by the step, and then by the UCI. '''
def find_Simsetting_json_files(directory, simulation):
    pattern = f'SimSettings_{simulation}*.json'
    matching_files = []
    stp_numbers = set()
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, pattern):
            stp_num, _ = extract_numbers(filename)
            stp_numbers.add(stp_num)
            matching_files.append(os.path.join(root, filename))
    # Sort the list based on the extracted numbers
    matching_files.sort(key=lambda x: extract_numbers(os.path.basename(x)))
    return matching_files, sorted(stp_numbers)

def get_propagation_speed_features(path_to_simulations, simulation):
    max_propagation_speed, avg_propagation_speed = 0,0
    simulation_folder = path_to_simulations + simulation
    simsettings_files, nothin = find_Simsetting_json_files(simulation_folder, simulation)

    '''only include files up until the last timestep of crack growth. the files after that should not be considered in the time since the crack doesnt grow anymore.'''
    index = next((i for i, filename in reversed(list(enumerate(simsettings_files))) if re.search(r'_UCI_([1-9]\d*)\.json$', os.path.basename(filename))), None)
    if index is not None:
        simsettings_files = simsettings_files[:index + 1]

    '''getting all of the time step numbers'''
    stp_numbers = set()
    for filename in simsettings_files:
        match = re.search(r'_Stp(\d+)_UCI_', os.path.basename(filename))
        if match:
            stp_numbers.add(int(match.group(1)))
    '''sorting them and putting them in a list'''
    sorted_timesteps = sorted(list(stp_numbers))

    propagation_speeds = []
    intitation_timestep = sorted_timesteps[0]
    last_propagation_timestep = sorted_timesteps[-1]
    '''go through each timestep and collect the total propagation at the timestep.'''
    for timestep in sorted_timesteps:
        ucis_in_this_timestep = [filename for filename in simsettings_files if f'Stp{timestep}_UCI' in os.path.basename(filename)]
        if(timestep == intitation_timestep): #get the first crack length
            file = open(ucis_in_this_timestep[0], 'r')
            data = json.load(file)
            initial_crack_len = data['crack_length'][0][0]
            current_crack_len = initial_crack_len
            file.close() 
        
        file = open(ucis_in_this_timestep[-1], 'r')
        data = json.load(file)
        last_crack_len_at_this_timestep = data['crack_length'][0][-1]
        propagation_speed_in_this_timestep = last_crack_len_at_this_timestep - current_crack_len
        propagation_speeds.append(propagation_speed_in_this_timestep)
        current_crack_len = last_crack_len_at_this_timestep
        file.close() 

    max_propagation_speed = np.max(propagation_speeds)
    avg_propagation_speed = np.average(propagation_speeds)
    return max_propagation_speed, avg_propagation_speed, intitation_timestep

def create_df():
    maxes = get_max_step_and_max_UCIs(FOLDER_PATH)
    #TODO: add columns to dataframe as necessary
    columns = [
            #"thickness at impact",
            "init x", 
            "init y", 
            "init z",
            "timestep_init",
            "max_prop_speed",
            "avg_prop_speed",
            "dist btw frts", 
            "crack len",
            "linearity",
            "max thickness",
            "mean thickness",
            "median_thickness", 
            "var_thickness", 
            "std_thickness",
            "thickness_at_init",
            "max_kink",
            "abs_val_mean_kink",
            "mean_kink",
            "sum_kink",
            "abs_val_sum_kink",
            "median_kink",
            "std_kink",
            "var_kink",
            "avg_ori",
            "angle_btw",
            "height", 
            "phi", 
            "theta"]
    df = pd.DataFrame(columns=columns)
    folder_path = FOLDER_PATH

    thickness_at_init = []
    initiation_cite_x = []
    initiation_cite_y = []
    initiation_cite_z = []
    timestep_inits = []
    max_prop_speeds = []
    avg_prop_speeds = []
    distance_between_fronts = []
    crack_lengths = []
    linearity_arr = []
    height_array = []
    phi_array = []
    theta_array = []
    max_thickness_arr = []
    mean_thickness_arr = []
    median_thickness_arr = []
    var_thickness_arr = []
    std_thickness_arr = []
    max_kink_arr = []
    abs_val_mean_kink_arr = []
    mean_kink_arr = []
    sum_kink_arr = []
    abs_val_sum_kink_arr = []
    average_orientation_arr = []
    angle_between_cracks_arr = []
    all_median_kinks = []
    all_std_kinks = []
    all_var_kinks = []

    i = 0
    #goes through each of the simulations and gathers features for the dataframe
    ''' this for loop goes through and gets all of the features that we can get from the end of the simulation '''
    for key in maxes.keys():
        print(i)
        print(key)
        # if(key != 'Para_2-0275ft_PHI_24_THETA_222'): continue

        good_history = check_history_output(folder_path, key)
 
        labels = turn_filename_to_labels(key)
        final_front_locations = get_final_front_locations(folder_path, key)
        crack_type = get_crack_type(folder_path, key)
        # final_front_locations = get_final_front_locations(folder_path, key, maxes[key][0], maxes[key][1]) #TODO delete this
        initiation_cite = get_initiation_cite(folder_path, key)
        average_orientation, angle_between_cracks = find_orientation(folder_path, key, final_front_locations, initiation_cite, crack_type)
        
        d = get_euclidean_distance(final_front_locations[0], final_front_locations[1])

        '''gets the node locations of the inner, outer, and main side nodes at the maximum dynamic step of the simulation'''
        inner_surface_nodes, outer_surface_nodes, main_side_nodes, node_locations = get_main_side_outer_and_inner_surface_nodes(folder_path, key, get_max_dynamic_step(folder_path, key))
        ''' turning outer_surface, inner_surface, and main_side nodes into easy to work with pandas dataframes '''
        inner_surface_nodes_df = convert_node_locations_to_df(node_locations, inner_surface_nodes)
        outer_surface_nodes_df = convert_node_locations_to_df(node_locations, outer_surface_nodes)
        main_side_nodes_df = convert_node_locations_to_df(node_locations, main_side_nodes)

        len = get_crack_len(outer_surface_nodes, main_side_nodes, node_locations, final_front_locations)
        # thickness_at_init = get_thickness_at_init(initiation_cite, inner_surface_nodes, outer_surface_nodes)
        # thickness_at_impact = get_thickness_at_impact(impa)
        linearity = get_linearity(folder_path, key)
        max_kink, abs_val_mean_kink, mean_kink, sum_kink, abs_val_sum_kink, median_kink, std_kinks, var_kinks = kink_angle_call(folder_path, key)
        max_thickness, mean_thickness, median_thickness, var_thickness, std_thickness = get_max_and_mean_thickness(folder_path, key)

        current_thickness_at_init = get_thickness_at_location(initiation_cite, outer_surface_nodes_df, inner_surface_nodes_df)
        max_propagation_speed, avg_propagation_speed, intitation_timestep = get_propagation_speed_features(folder_path, key)

        height_array.append(labels[0])
        phi_array.append(labels[1])
        theta_array.append(labels[2])
        thickness_at_init.append(current_thickness_at_init)
        initiation_cite_x.append(initiation_cite[0])
        initiation_cite_y.append(initiation_cite[1])
        initiation_cite_z.append(initiation_cite[2])
        timestep_inits.append(intitation_timestep)
        max_prop_speeds.append(max_propagation_speed)
        avg_prop_speeds.append(avg_propagation_speed)
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
        all_median_kinks.append(median_kink)
        all_std_kinks.append(std_kinks)
        all_var_kinks.append(var_kinks)
        median_thickness_arr.append(median_thickness)
        var_thickness_arr.append(var_thickness)
        std_thickness_arr.append(std_thickness)

        i += 1
                


    df = {
        "init x": initiation_cite_x, 
        "init y": initiation_cite_y, 
        "init z": initiation_cite_z, 
        "timestep_init":timestep_inits,
        "max_prop_speed":max_prop_speeds,
        "avg_prop_speed":avg_prop_speeds,
        "dist btw frts": distance_between_fronts, 
        "crack len": crack_lengths,
        "linearity": linearity_arr,
        "max thickness": max_thickness_arr,
        "mean thickness": mean_thickness_arr,
        "median_thickness": median_thickness_arr, 
        "var_thickness": var_thickness_arr, 
        "std_thickness": std_thickness_arr,
        "thickness_at_init":thickness_at_init,
        "max_kink": max_kink_arr,
        "abs_val_mean_kink": abs_val_mean_kink_arr,
        "mean_kink": mean_kink_arr,
        "sum_kink": sum_kink_arr,
        "abs_val_sum_kink": abs_val_sum_kink_arr,
        "median_kink": all_median_kinks,
        "std_kink": all_std_kinks,
        "var_kink": all_var_kinks,
        "avg_ori": average_orientation_arr,
        "angle_btw": angle_between_cracks_arr,
        "height": height_array, 
        "phi": phi_array, 
        "theta": theta_array
        }
    df = pd.DataFrame(df,columns=columns)

    ''' add impact sites to the df '''
    min_distance_locations, parietal_nodes, impact_bones = get_all_impact_node_sites(FOLDER_PATH)
    parietal_nodes = np.asarray(parietal_nodes)
    parietal_node_locations = parietal_nodes[:,1:] #parietal_node_locations doesnt include the node number, parietal_nodes does include the node number
    impact_nodes = np.array(list(min_distance_locations.values()))
    df = add_impact_sites_to_existing_df(df, 'C:\\Users\\u1056\\sfx\\sfx_ML\\sfx_ML\\Feature_gathering\\','impact_sites.json')


    ''' convert all locational features and labels into Jimmys reference frame. (z axis goes from the center of mass through the ossification site)'''
    impact_sites_list = df['impact_sites'].to_numpy()
    # Transform each list element into a numpy array
    impact_sites = np.array([np.array(ast.literal_eval(string)) for string in impact_sites_list])
    df = insert_column_into_df(df, 'impact_sites', 'impact site x', impact_sites[:,0])
    df = insert_column_into_df(df, 'impact_sites', 'impact site y', impact_sites[:,1])
    df = insert_column_into_df(df, 'impact_sites', 'impact site z', impact_sites[:,2])

    Jimmy_impact_sites_x, Jimmy_impact_sites_y, Jimmy_impact_sites_z = convert_coordinates_to_new_basis(Material_X, Material_Y, Material_Z, CM, impact_sites[:,0], impact_sites[:,1], impact_sites[:,2])
    Jimmy_init_x, Jimmy_init_y, Jimmy_init_z = convert_coordinates_to_new_basis(Material_X, Material_Y, Material_Z, CM, df['init x'].to_numpy(), df['init y'].to_numpy(), df['init z'].to_numpy())
    # Jimmy_front_0_x, Jimmy_front_0_y, Jimmy_front_0_z = convert_coordinates_to_new_basis(Material_X, Material_Y, Material_Z, CM, df['front 0 x'].to_numpy(), df['front 0 y'].to_numpy(), df['front 0 z'].to_numpy())
    # Jimmy_front_1_x, Jimmy_front_1_y, Jimmy_front_1_z = convert_coordinates_to_new_basis(Material_X, Material_Y, Material_Z, CM, df['front 1 x'].to_numpy(), df['front 1 y'].to_numpy(), df['front 1 z'].to_numpy())

    Jimmy_impact_sites_r, Jimmy_impact_sites_phi, Jimmy_impact_sites_theta = convert_cartesian_to_spherical(Jimmy_impact_sites_x, Jimmy_impact_sites_y, Jimmy_impact_sites_z)
    Jimmy_init_r, Jimmy_init_phi, Jimmy_init_theta = convert_cartesian_to_spherical(Jimmy_init_x, Jimmy_init_y, Jimmy_init_z)
    # Jimmy_front0_r, Jimmy_front0_phi, Jimmy_front0_theta = convert_cartesian_to_spherical(Jimmy_front_0_x, Jimmy_front_0_y, Jimmy_front_0_z)
    # Jimmy_front1_r, Jimmy_front1_phi, Jimmy_front1_theta = convert_cartesian_to_spherical(Jimmy_front_1_x, Jimmy_front_1_y, Jimmy_front_1_z)

    # df = insert_column_into_df(df, 'init z', 'Jimmy_init theta', Jimmy_init_theta)
    # df = insert_column_into_df(df, 'init z', 'Jimmy_init phi', Jimmy_init_phi)
    # df = insert_column_into_df(df, 'init z', 'Jimmy_init r', Jimmy_init_r)

    # df = insert_column_into_df(df, 'front 0 z', 'Jimmy_front 0 theta', Jimmy_front0_theta)
    # df = insert_column_into_df(df, 'front 0 z', 'Jimmy_front 0 phi', Jimmy_front0_phi)
    # df = insert_column_into_df(df, 'front 0 z', 'Jimmy_front 0 r', Jimmy_front0_r)

    # df = insert_column_into_df(df, 'front 1 z', 'Jimmy_front 1 theta', Jimmy_front1_theta)
    # df = insert_column_into_df(df, 'front 1 z', 'Jimmy_front 1 phi', Jimmy_front1_phi)
    # df = insert_column_into_df(df, 'front 1 z', 'Jimmy_front 1 r', Jimmy_front1_r)

    df = insert_column_into_df(df, 'impact_sites', 'Jimmy_impact site theta', Jimmy_impact_sites_theta)
    df = insert_column_into_df(df, 'impact_sites', 'Jimmy_impact site phi', Jimmy_impact_sites_phi)
    df = insert_column_into_df(df, 'impact_sites', 'Jimmy_impact site r', Jimmy_impact_sites_r)

    df = insert_column_into_df(df, 'impact_sites', 'Jimmy_impact site z', Jimmy_impact_sites_z)
    df = insert_column_into_df(df, 'impact_sites', 'Jimmy_impact site y', Jimmy_impact_sites_y)
    df = insert_column_into_df(df, 'impact_sites', 'Jimmy_impact site x', Jimmy_impact_sites_x)

    df = insert_column_into_df(df, 'init z', 'Jimmy_init z', Jimmy_init_z)
    df = insert_column_into_df(df, 'init z', 'Jimmy_init y', Jimmy_init_y)
    df = insert_column_into_df(df, 'init z', 'Jimmy_init x', Jimmy_init_x)

    # df = insert_column_into_df(df, 'front 0 z', 'Jimmy_front_0_z', Jimmy_front_0_z)
    # df = insert_column_into_df(df, 'front 0 z', 'Jimmy_front_0_y', Jimmy_front_0_y)
    # df = insert_column_into_df(df, 'front 0 z', 'Jimmy_front_0_x', Jimmy_front_0_x)

    # df = insert_column_into_df(df, 'front 1 z', 'Jimmy_front_1_z', Jimmy_front_1_z)
    # df = insert_column_into_df(df, 'front 1 z', 'Jimmy_front_1_y', Jimmy_front_1_y)
    # df = insert_column_into_df(df, 'front 1 z', 'Jimmy_front_1_x', Jimmy_front_1_x)

    df = df.drop(columns=['impact site x', 'impact site y', 'impact site z',
                          'init x', 'init y', 'init z',
                          'impact site x', 'impact site y', 'impact site z', 'impact_sites'])
    
    column_name_changes = {
    'Jimmy_front 0 theta': 'front 0 theta',
    'Jimmy_front 0 phi': 'front 0 phi',
    'Jimmy_front 0 r': 'front 0 r',
    'Jimmy_front_0_z': 'front_0_z', 
    'Jimmy_front_0_y': 'front_0_y', 
    'Jimmy_front_0_x': 'front_0_x',
    'Jimmy_front 1 theta': 'front 1 theta', 
    'Jimmy_front 1 phi': 'front 1 phi', 
    'Jimmy_front 1 r': 'front 1 r',
    'Jimmy_front_1_z': 'front 1 z', 
    'Jimmy_front_1_y': 'front 1 y', 
    'Jimmy_front_1_x': 'front 1 x',
    'Jimmy_init theta': 'init theta', 
    'Jimmy_init phi': 'init phi', 
    'Jimmy_init r': 'init r', 
    'Jimmy_init z': 'init z',
    'Jimmy_init y': 'init y', 
    'Jimmy_init x': 'init x',
    'Jimmy_impact site theta': 'impact site theta', 
    'Jimmy_impact site phi': 'impact site phi',
    'Jimmy_impact site r': 'impact site r',
    'Jimmy_impact site z': 'impact site z', 
    'Jimmy_impact site y': 'impact site y',
    'Jimmy_impact site x': 'impact site x'
    }

    # Rename specific columns of the DataFrame
    df = df.rename(columns=column_name_changes)

    
    return df



# FOLDER_PATH = "F:\\Jake\\good_simies\\"
FOLDER_PATH = "F:\\Jake\\good_simies_new_R_curve_j_3.5_q_2.5\\"
# FOLDER_PATH = "F:\\Jake\\good_simies_old_R_curve\\" #making data from old simulations so that we can delete them

df = create_df()
random_indicies = random.sample(range(1, len(df)), 30)
test_df = df.iloc[random_indicies]
train_df = df.drop(random_indicies, axis=0)
print(df)
Pearson_Correlations_for_df(df, "height")
Pearson_Correlations_for_df(df, "phi")
Pearson_Correlations_for_df(df, "theta")
# save_df(test_df, "C:\\Users\\u1056\\sfx\\sfx_ML\\sfx_ML\\Feature_gathering\\New_Crack_Len_TEST_OG_dataframe.csv")
# save_df(train_df, "C:\\Users\\u1056\\sfx\\sfx_ML\\sfx_ML\\Feature_gathering\\New_Crack_Len_TRAIN_OG_dataframe.csv")
# save_df(df, "F:\\Jake\\good_simies_old_R_curve\\Old_R_curve_data_"+ current_date_string +".csv")
save_df(df, "C:\\Users\\u1056\\sfx\\sfx_ML_data\\feature_data\\New_Crack_Len_FULL_OG_dataframe_" + current_date_string + ".csv")
df = PhiTheta_to_cartesian(df)
# save_df(df, "F:\\Jake\\good_simies_old_R_curve\\Old_R_curve_data_cartesian_"+ current_date_string +".csv")
save_df(df, "C:\\Users\\u1056\\sfx\\sfx_ML_data\\feature_data\\OG_dataframe_cartesian_" + current_date_string + ".csv")
Pearson_Correlations_for_df(df, "x")
Pearson_Correlations_for_df(df, "y")
Pearson_Correlations_for_df(df, "z")
print("done")



