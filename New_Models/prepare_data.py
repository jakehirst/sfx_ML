import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from sklearn.cluster import *
from sklearn.neighbors import *
from scipy.stats import pearsonr
import glob
from PIL import Image
import imageio


""" 
reads the dataset as a csv and deletes the columns that start with "Unnamed" before returning the dataset as a pd dataframe 
it also turns the 'impact_sites' column from strings into the actual arrays.
"""
def read_dataset(dataset_filepath):
    dataset = pd.read_csv(dataset_filepath)
    dataset = dataset[dataset.filter(regex='^(?!Unnamed)').columns] #filtering out the "Unnamed" columns that pandas adds to the dataframes when it is read from a csv.
    # impact_site_strings = dataset['impact_sites'].to_numpy()
    # impact_points = []
    # for point in impact_site_strings: impact_points.append(eval(point)) #turning the string that looks like an array into an actual array 
    # dataset['impact_sites'] = impact_points
    return dataset

''' 
Discretely binning based on height label
'''
def Discretely_bin(df, num_bins, column_name, label_to_predict, saving_folder=None):
    # Compute the bin edges based on equal spacing
    bin_edges = np.linspace(1.0, 4.0, num_bins+1)
    # Bin the data using numpy.histogram()
    counts, bin_edges = np.histogram(df[label_to_predict], bins=bin_edges)
    # Create a new column in the DataFrame with the bin labels
    df[column_name] = pd.cut(df[label_to_predict], bins=bin_edges, labels=False)
    return df, bin_edges, counts

"""
clusters the column "column_name" by cartesian coordinates using k-means clustering and adds it to the dataframe df. k clusters are used.
returns the new dataframe and the centroids of the clusters
"""
def Kmeans_cluster_cartesian_coordinates(df, k, column_name, new_column_name, saving_folder=None):
    points_to_cluster = np.array(df[column_name].apply(lambda x: np.array(x)).tolist()) #turning the column of arrays into a numpy array with n rows, 3 columns
    kmeans = KMeans(n_clusters=k) #using sklearn's kmeans clustrering methods... super fast
    kmeans.fit(points_to_cluster)
    
    cluster_assignments = kmeans.labels_ 
    centroids = kmeans.cluster_centers_
    
    df[new_column_name] = cluster_assignments
    plot_points_and_clusters(df, new_column_name, column_name, centroids, saving_folder)
    return df, centroids

"""
plotting the clusters and their centroids (if any) with the number of examples per cluster in the legend. This is done after the clusters are assigned.
"""
def plot_points_and_clusters(df, cluster_column_name, coordinate_column_name, centroids=None, saving_folder=None):
    # Create a 3D plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    cluster_num = 0
    if(not centroids.all() == None):
        ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2], s = 200, marker='*', c = 'r',label="centroids")
        
    for cluster in range(0, max(df[cluster_column_name])+1):
        current_cluster_df = df.loc[df[cluster_column_name] == cluster]
        current_cluster_examples = np.array(current_cluster_df[coordinate_column_name].apply(lambda x: np.array(x)).tolist()) #turning the column of arrays into a numpy array with n rows, 3 columns
        x = current_cluster_examples[:,0]
        y = current_cluster_examples[:,1]
        z = current_cluster_examples[:,2]
        ax.scatter(x, y, z, alpha = 0.3, label=f'cluster number {cluster} | total = {len(current_cluster_examples)}')
        cluster_num += 1
    plt.legend()
    if(saving_folder == None):
        plt.show()
        plt.close()
    else:
        plt.savefig(saving_folder + 'clusters.png')
        plt.close()

'''
performs pearson correlation on all features with and labels. 
returns the whole correlation matrix, p-value matrix, and the features that have a p-value less than the threshold
'''    
def Pearson_correlation(df, label_to_predict, maximum_p_value):
    corr_matrix, p_matrix = df.corr(method=lambda x, y: pearsonr(x, y)[0]), df.corr(method=lambda x, y: pearsonr(x, y)[1])
    important_features = p_matrix[p_matrix[label_to_predict] < maximum_p_value].index
    return corr_matrix, p_matrix, list(important_features)

'''
Removes all columns from the 'labels' array out of the dataset that is not the label_to_predict
'''
def remove_unwanted_labels(dataset, label_to_predict, labels):
    for label in labels:
        if(not label == label_to_predict):
            dataset = dataset.drop(label, axis=1) 
    return dataset

'''
gets the list of images in their raw form and adds it to the dataframe under 'raw_images'
'''
def get_images_from_dataset(df, image_folder):
    images = []
    for row in df.iterrows():
        height = str(row[1]['height']).replace('.', '-')
        phi = str(row[1]['phi']).replace('.0', '')
        theta = str(row[1]['theta']).replace('.0', '')
        max_step_uci = find_max_step_and_uci(image_folder, height, phi, theta)
        image_path = image_folder + f'/OG/Para_{height}ft_PHI_{phi}_THETA_{theta}/Step{max_step_uci[0]}_UCI_{max_step_uci[1]}_Dynamic.png' 
        img_arr = imageio.v2.imread(image_path)[:,:,0:3] / 255.0
        images.append(img_arr)
    return np.asarray(images)

'''
finds the max step and uci for a single simulation given the image folder, returns it as a tuple (step,uci)
'''
def find_max_step_and_uci(image_folder, height, phi, theta):
    specific_folder = image_folder + f'/OG/Para_{height}ft_PHI_{phi}_THETA_{theta}' 
    files = glob.glob(specific_folder + "/*Dynamic*") #finds all of the files that have 'Dynamic' in the name
    Step_list = [int(s.split('/Step')[-1].split('_')[0]) for s in files]
    uci_list = [int(s.split('_UCI_')[-1].split('_')[0]) for s in files]
    Step_UCI_list = [(x, y) for x, y in zip(Step_list, uci_list)]
    sorted_list = sorted(Step_UCI_list, key=lambda x: (x[0], x[1])) #sorts the Step/UCI list by step and then uci 
    return sorted_list[-1]

'''
prepares the full dataset for regression with a single output from the given pathname, returning...
    - Correlated features from 1D dataset
    - raw images from image datset
    - the labels in the same order as the dataset
'''
def prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.05):
    dataset = read_dataset(full_dataset_pathname)
    #adding images to the dataset
    # raw_images = get_images_from_dataset(dataset, image_folder)
    corr_matrix, p_matrix, important_features = Pearson_correlation(dataset, label_to_predict, maximum_p_value=maximum_p_value) #changed this from .05 to .01 on 5/22/23
    dataset = remove_unwanted_labels(dataset, label_to_predict, all_labels)
    full_dataset_labels = dataset[label_to_predict].to_numpy()
    for label in all_labels: 
        if(important_features.__contains__(label)): important_features.remove(label)

    correlated_featureset = dataset[important_features]
    print(corr_matrix[important_features])
    print(p_matrix[important_features])
    # return correlated_featureset, raw_images, full_dataset_labels, important_features
    return correlated_featureset, full_dataset_labels, important_features

'''
prepares the full dataset for classification using k-means clustering from the given pathname, returning...
    - Correlated features from 1D dataset
    - raw images from image datset
    - the labels in the same order as the dataset (not one hot vectors, but rather bin 1 - num_clusters)
'''
def prepare_dataset_Kmeans_cluster(full_dataset_pathname, image_folder, label_to_predict, cluster='impact_sites', num_clusters=None, saving_folder=None):
    dataset = read_dataset(full_dataset_pathname)
    #adding images to the dataset
    raw_images = get_images_from_dataset(dataset, image_folder)
    print(dataset)
    # label_to_predict = 'height'
    if(not num_clusters == None):
        dataset, cluster_centroids = Kmeans_cluster_cartesian_coordinates(dataset, num_clusters, cluster, label_to_predict, saving_folder=saving_folder)
    # removing any labels that are not the label we are predicting
    labels = ['height', 'phi', 'theta', 'impact_sites']
    dataset = remove_unwanted_labels(dataset, label_to_predict, labels)
    full_dataset_labels = dataset[label_to_predict].to_numpy()
    # only using the well correlated features
    corr_matrix, p_matrix, important_features = Pearson_correlation(dataset, label_to_predict, maximum_p_value=0.01)
    correlated_featureset = dataset[important_features]
    print(corr_matrix[important_features])
    print(p_matrix[important_features])
        
    return correlated_featureset, raw_images, full_dataset_labels

'''
prepares the full dataset for classification using discrete binning for a single label from the given pathname, returning...
    - Correlated features from 1D dataset
    - raw images from image datset
    - the labels in the same order as the dataset (not one hot vectors, but rather bin 1 - num_bins)
'''
def prepare_dataset_discrete_Binning(full_dataset_pathname, image_folder, label_to_predict, num_bins=2, saving_folder=None):
    dataset = read_dataset(full_dataset_pathname)
    #adding images to the dataset
    raw_images = get_images_from_dataset(dataset, image_folder)
    new_label_to_predict = 'binned_' + label_to_predict
    dataset, bin_edges, counts = Discretely_bin(dataset, num_bins, new_label_to_predict, label_to_predict, saving_folder=None)
    corr_matrix, p_matrix, important_features = Pearson_correlation(dataset, label_to_predict, maximum_p_value=0.01)
    labels = ['height', 'phi', 'theta', 'impact_sites', new_label_to_predict]
    dataset = remove_unwanted_labels(dataset, new_label_to_predict, labels)
    full_dataset_labels = dataset[new_label_to_predict].to_numpy()
    for label in labels: 
        if(important_features.__contains__(label)): important_features.remove(label)

    correlated_featureset = dataset[important_features]
    print(corr_matrix[important_features])
    print(p_matrix[important_features])
    return correlated_featureset, raw_images, full_dataset_labels

''' 
removes all of the spacial features that are in the ABAQUS reference frame
'''
def remove_ABAQUS_features(df):
    features_to_remove = ['init x', 'init y', 'init z', 'front 0 x', 'front 0 y', 'front 0 z', 'front 1 x', 'front 1 y', 'front 1 z']
    for feature in features_to_remove:
        if(df.columns.__contains__(feature)): df = df.drop(feature, axis=1)
    return df

# dataset = read_dataset("/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/FULL_OG_dataframe_with_impact_sites.csv")
# dataset, cluster_centroids = Kmeans_cluster_cartesian_coordinates(dataset, 10, 'impact_sites')
# plot_points_and_clusters(dataset, 'cluster_assignments', 'impact_sites', cluster_centroids)


