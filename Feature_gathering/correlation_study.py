import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import ast



#material basis vectors for RPA bone
Material_X = np.array([-0.87491124, -0.44839274,  0.18295974])
Material_Y = np.array([ 0.23213791, -0.71986519, -0.65414532])
Material_Z = np.array([ 0.42502036, -0.5298472,   0.7339071 ])
#Center of mass of the RPA bone in abaqus basis
CM = np.array([106.55,72.79,56.64])
#Ossification center of the RPA bone in abaqus basis
OC = np.array([130.395996,46.6063,98.649696])

""" Returns the correlation matrix, p matrix, and the features that have a p value less than the minimum_p_value """
def Pearson_correlation(df, label_to_predict, maximum_p_value):
    corr_matrix, p_matrix = df.corr(method=lambda x, y: pearsonr(x, y)[0]), df.corr(method=lambda x, y: pearsonr(x, y)[1])
    important_features = p_matrix[p_matrix[label_to_predict] < maximum_p_value].index
    return corr_matrix, p_matrix, list(important_features)

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

'''
Inserts a column of data (new_column_data) into the df after the column_before column and it 
calls the new column 'new_column_name'
Returns the new dataframe.
'''
def insert_column_into_df(df, column_before, new_column_name, new_column_data):
    df.insert(loc=df.columns.get_loc(column_before) , column=new_column_name, value=new_column_data)
    return df

full_df = pd.read_csv("/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/FULL_OG_dataframe_with_impact_sites.csv")
full_df = full_df.drop("Unnamed: 0.1", axis=1)
full_df = full_df.drop("Unnamed: 0", axis=1)

impact_sites_list = full_df['impact_sites'].to_numpy()
# Transform each list element into a numpy array
impact_sites = np.array([np.array(ast.literal_eval(string)) for string in impact_sites_list])

full_df = insert_column_into_df(full_df, 'impact_sites', 'impact site x', impact_sites[:,0])
full_df = insert_column_into_df(full_df, 'impact_sites', 'impact site y', impact_sites[:,1])
full_df = insert_column_into_df(full_df, 'impact_sites', 'impact site z', impact_sites[:,2])
labels = ['height', 'phi', 'theta', 'impact site x', 'impact site y', 'impact site z']

""" correlation study before everything is converted to Jimmy's reference frame """
corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'height', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['height'][important_features])
corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'phi', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['phi'][important_features])
corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'theta', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['theta'][important_features])
corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'impact site x', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['impact site x'][important_features])
corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'impact site y', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['impact site y'][important_features])
corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'impact site z', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['impact site z'][important_features])




Jimmy_impact_sites_x, Jimmy_impact_sites_y, Jimmy_impact_sites_z = convert_coordinates_to_new_basis(Material_X, Material_Y, Material_Z, CM, impact_sites[:,0], impact_sites[:,1], impact_sites[:,2])
Jimmy_init_x, Jimmy_init_y, Jimmy_init_z = convert_coordinates_to_new_basis(Material_X, Material_Y, Material_Z, CM, full_df['init x'].to_numpy(), full_df['init y'].to_numpy(), full_df['init z'].to_numpy())
Jimmy_front_0_x, Jimmy_front_0_y, Jimmy_front_0_z = convert_coordinates_to_new_basis(Material_X, Material_Y, Material_Z, CM, full_df['front 0 x'].to_numpy(), full_df['front 0 y'].to_numpy(), full_df['front 0 z'].to_numpy())
Jimmy_front_1_x, Jimmy_front_1_y, Jimmy_front_1_z = convert_coordinates_to_new_basis(Material_X, Material_Y, Material_Z, CM, full_df['front 1 x'].to_numpy(), full_df['front 1 y'].to_numpy(), full_df['front 1 z'].to_numpy())

Jimmy_impact_sites_r, Jimmy_impact_sites_phi, Jimmy_impact_sites_theta = convert_cartesian_to_spherical(Jimmy_impact_sites_x, Jimmy_impact_sites_y, Jimmy_impact_sites_z)
Jimmy_init_r, Jimmy_init_phi, Jimmy_init_theta = convert_cartesian_to_spherical(Jimmy_init_x, Jimmy_init_y, Jimmy_init_z)
Jimmy_front0_r, Jimmy_front0_phi, Jimmy_front0_theta = convert_cartesian_to_spherical(Jimmy_front_0_x, Jimmy_front_0_y, Jimmy_front_0_z)
Jimmy_front1_r, Jimmy_front1_phi, Jimmy_front1_theta = convert_cartesian_to_spherical(Jimmy_front_1_x, Jimmy_front_1_y, Jimmy_front_1_z)

full_df = insert_column_into_df(full_df, 'init z', 'Jimmy_init theta', Jimmy_init_theta)
full_df = insert_column_into_df(full_df, 'init z', 'Jimmy_init phi', Jimmy_init_phi)
full_df = insert_column_into_df(full_df, 'init z', 'Jimmy_init r', Jimmy_init_r)

full_df = insert_column_into_df(full_df, 'front 0 z', 'Jimmy_front 0 theta', Jimmy_front0_theta)
full_df = insert_column_into_df(full_df, 'front 0 z', 'Jimmy_front 0 phi', Jimmy_front0_phi)
full_df = insert_column_into_df(full_df, 'front 0 z', 'Jimmy_front 0 r', Jimmy_front0_r)

full_df = insert_column_into_df(full_df, 'front 1 z', 'Jimmy_front 1 theta', Jimmy_front1_theta)
full_df = insert_column_into_df(full_df, 'front 1 z', 'Jimmy_front 1 phi', Jimmy_front1_phi)
full_df = insert_column_into_df(full_df, 'front 1 z', 'Jimmy_front 1 r', Jimmy_front1_r)

full_df = insert_column_into_df(full_df, 'impact_sites', 'Jimmy_impact site theta', Jimmy_impact_sites_theta)
full_df = insert_column_into_df(full_df, 'impact_sites', 'Jimmy_impact site phi', Jimmy_impact_sites_phi)
full_df = insert_column_into_df(full_df, 'impact_sites', 'Jimmy_impact site r', Jimmy_impact_sites_r)

full_df = insert_column_into_df(full_df, 'impact_sites', 'Jimmy_impact site z', Jimmy_impact_sites_z)
full_df = insert_column_into_df(full_df, 'impact_sites', 'Jimmy_impact site y', Jimmy_impact_sites_y)
full_df = insert_column_into_df(full_df, 'impact_sites', 'Jimmy_impact site x', Jimmy_impact_sites_x)

full_df = insert_column_into_df(full_df, 'init z', 'Jimmy_init z', Jimmy_init_z)
full_df = insert_column_into_df(full_df, 'init z', 'Jimmy_init y', Jimmy_init_y)
full_df = insert_column_into_df(full_df, 'init z', 'Jimmy_init x', Jimmy_init_x)

full_df = insert_column_into_df(full_df, 'front 0 z', 'Jimmy_front_0_z', Jimmy_front_0_z)
full_df = insert_column_into_df(full_df, 'front 0 z', 'Jimmy_front_0_y', Jimmy_front_0_y)
full_df = insert_column_into_df(full_df, 'front 0 z', 'Jimmy_front_0_x', Jimmy_front_0_x)

full_df = insert_column_into_df(full_df, 'front 1 z', 'Jimmy_front_1_z', Jimmy_front_1_z)
full_df = insert_column_into_df(full_df, 'front 1 z', 'Jimmy_front_1_y', Jimmy_front_1_y)
full_df = insert_column_into_df(full_df, 'front 1 z', 'Jimmy_front_1_x', Jimmy_front_1_x)

labels = ['height', 'phi', 'theta', 'impact site x', 'impact site y', 'impact site z', 'Jimmy_impact site x', 'Jimmy_impact site y', 
          'Jimmy_impact site z', 'Jimmy_impact site r', 'Jimmy_impact site phi', 'Jimmy_impact site theta']

""" correlation study after everything is converted to Jimmy's reference frame """
corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'height', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['height'][important_features])
corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'phi', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['phi'][important_features])
corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'theta', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['theta'][important_features])

corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'impact site x', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['impact site x'][important_features])
corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'impact site y', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['impact site y'][important_features])
corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'impact site z', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['impact site z'][important_features])

corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'Jimmy_impact site x', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['Jimmy_impact site x'][important_features])
corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'Jimmy_impact site y', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['Jimmy_impact site y'][important_features])
corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'Jimmy_impact site z', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['Jimmy_impact site z'][important_features])

corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'Jimmy_impact site r', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['Jimmy_impact site r'][important_features])
corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'Jimmy_impact site phi', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['Jimmy_impact site phi'][important_features])
corr_matrix, p_matrix, important_features = Pearson_correlation(full_df, 'Jimmy_impact site theta', maximum_p_value=0.05)
important_features  = [x for x in important_features if x not in labels]
print(corr_matrix['Jimmy_impact site theta'][important_features])

full_df.to_csv("/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/FULL_OG_dataframe_with_impact_sites_and_Jimmy_RF.csv")



# impact_r, impact_phi, impact_theta = 

print("here")