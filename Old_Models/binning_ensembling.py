import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os.path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import KFold
from CNN_function_library import *
import tensorflow.keras.backend as K
from PIL import Image
from quaternions import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
import csv
from datetime import date
import os
from CNN_2_in_binned import *
from keras.utils import to_categorical


""" loads all of the models and their stats from a single folder """
def load_models_and_stats(folder_path):
    all_models = {}
    all_stats = {}
    for filename in os.listdir(folder_path):
        # getting all of the tensorflow models and 
        if filename.startswith('trained_model_fold_'):
            fold_no = filename.split("_")[-1].split(".")[0]
            model = tf.keras.models.load_model(folder_path + f'/{filename}')
            all_models[folder_path + f" fold {fold_no}"] = model
 
            
        if filename.endswith('_model_stats.csv'):
            with open(folder_path + f'/{filename}', 'r') as csv_file:
                fold_no = filename.split("_")[0][-1]
                csv_reader = csv.reader(csv_file)
                next(csv_reader) #skipping the first row
                stats = {row[0]: row[1] for row in csv_reader}
                all_stats[folder_path + f" fold {fold_no}"] = stats
    return all_models, all_stats

def get_everything_ready(parent_folder_name, data_folder_1D, dataset_1D, included_features,  bin_type, num_phi_bins, num_theta_bins):
    df_1D = get_1D_inputs(data_folder_1D, dataset_1D, bin_type)      
    simple_df_1D = remove_features(df_1D, features_to_keep=included_features)

    print("getting image data... ")
    args = prepare_data(parent_folder_name, ["OG"])
    df_images = get_image_inputs(args[0], args[1], args[2], args[3])
    print("syncing data...")
    phiandtheta_df, df_images = sync_image_and_1D_inputs(simple_df_1D, df_images)
    
    if(bin_type == "solid center phi and theta"):
        phiandtheta_df = phiandtheta_df.drop("height", axis=1)
        df_1D, y_col_values, bins_and_values = Bin_phi_and_theta_center_target(phiandtheta_df, num_phi_bins, num_theta_bins)
        folder = f"/Users/jakehirst/Desktop/sfx/captain_america_plots_new_data/center_circle_phi_{num_phi_bins}_theta_{num_theta_bins}_bins/"
    elif(bin_type == "phi and theta"):
        phiandtheta_df = phiandtheta_df.drop("height", axis=1)
        df_1D, y_col_values, bins_and_values = Bin_phi_and_theta(phiandtheta_df, num_phi_bins, num_theta_bins)
        folder = f"/Users/jakehirst/Desktop/sfx/captain_america_plots_new_data/phi_{num_phi_bins}_theta_{num_theta_bins}_bins/"
    elif(bin_type == "theta"):
        phiandtheta_df = phiandtheta_df.drop("height", axis=1)
        df_1D, y_col_values, bins_and_values = Bin_just_theta(phiandtheta_df, num_theta_bins)
        folder = f"/Users/jakehirst/Desktop/sfx/captain_america_plots_new_data/JustTheta_{num_theta_bins}_bins/"
    elif(bin_type == "phi"):
        phiandtheta_df = phiandtheta_df.drop("height", axis=1)
        df_1D, y_col_values, bins_and_values = Bin_just_phi(phiandtheta_df, num_phi_bins)
        folder = f"/Users/jakehirst/Desktop/sfx/captain_america_plots_new_data/JustPhi_{num_phi_bins}_bins/"
    else:
        y_col_values = []
        bins_and_values = []

    if('index' in df_1D.columns):
        df_1D = df_1D.drop("index", axis=1)
    return df_1D, y_col_values, bins_and_values, df_images

''' getting the 1D and image dataset ready to test after ensembling '''
def prepare_test_dataset(folder_1D_data, dataset_name, folder_image_data, label_to_predict, included_features,  bin_type, num_phi_bins, num_theta_bins):
    df_1D, y_col_values, bins_and_values, df_images = get_everything_ready(folder_image_data, folder_1D_data, dataset_name, included_features,  bin_type, num_phi_bins, num_theta_bins)
    
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255, #all pixel values set between 0 and 1 instead of 0 and 255.
    )
    
    #flow the images through the generators
    flow_train_images = train_generator.flow_from_dataframe(
        dataframe=df_images,
        x_col = 'image_path',
        y_col = [label_to_predict],
        # target_size=args[4][:2],  #can reduce the images to a certain size to reduce training time. 120x120 for example here
        color_mode='rgb',
        class_mode = 'raw', #keeps the classes of our labels the same after flowing
        batch_size=5, #can increase this to up to like 10 or so for how much data we have
        shuffle=False,
        seed=42,
    )
    
    #pulling the image matricies from the filepaths of images
    inputs = []

    #puttin the rgb matricies into train_inputs, test_inputs, and val_inputs
    #to see the image, use plt.imshow(arr)
    for image in flow_train_images._filepaths:
        arr = imageio.v2.imread(image)[:,:,0:3] / 255.0
        inputs.append(arr)
    actual_train_images = np.array(inputs)

    
    return df_1D, actual_train_images

def ensembled_confusion_matrix(y_true, y_pred, figname=None, folder=None):
    cm = metrics.confusion_matrix(y_true, y_pred)
    axes = np.arange(cm.shape[0])
    
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(cm,
                        index = axes, 
                        columns = axes)
    #Plotting the confusion matrix
    plt.figure(figsize=(9,9))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    if(folder == None):
        plt.show()
    else:
        plt.savefig(folder + f"/{figname}")
    plt.close()
    
def get_binned_ensembling_weights(all_stats, metric):
    total = 0
    weights = {}
    for key in all_stats.keys():
        stats = all_stats[key]
        if(metric.endswith('cat_cross')):
                metric_stat = 1/float(stats[metric])
                total += metric_stat
                weights[key] = metric_stat
        else:
            metric_stat = float(stats[metric])
            total += metric_stat
            weights[key] = metric_stat
    return weights, total

def ensemble_binned_predictions(folder_1D_data, dataset_name, folder_image_data, model_folder, weighted=False, metric=None):
    all_models, all_stats = load_models_and_stats(model_folder)
    bin_type = all_stats[list(all_stats.keys())[0]]['bin_type']
    included_features = eval(all_stats[list(all_stats.keys())[0]]['included_features'])
    
    if(bin_type == 'clusters'):
        clusterCSV = pd.read_csv(model_folder + '/clusters_means.csv')
        cluster_means = eval(clusterCSV['clusters'][0])
        raw_dataset = pd.read_csv(folder_1D_data + dataset_name)
        df_1D, cluster_arrays = cluster_points(raw_dataset, cluster_means)
        print("getting image data... ")
        args = prepare_data(folder_image_data, ["OG"])
        df_images = get_image_inputs(args[0], args[1], args[2], args[3])
        print("syncing data...")
        df_1D, df_images = sync_image_and_1D_inputs(df_1D.drop('Unnamed: 0', axis=1), df_images)
        df_1D = df_1D.reset_index(drop=True)
        df_1D = df_1D.drop(['height', 'phi', 'theta', 'Distance_to_Cluster'], axis=1)

        cluster_assignments = to_categorical(np.asarray(df_1D['Cluster']))
        y_col_values = []
        bin_df = pd.DataFrame(cluster_assignments)
        for col in bin_df.columns: y_col_values.append(str(col))
        bin_df.columns = y_col_values
        df_1D = pd.concat([df_1D, bin_df], axis=1)
        df_1D = df_1D.drop(['index','Cluster'], axis=1)

        df_images = df_images.reset_index(drop=True)
        actual_images, df_1D, true_y = organize_test_data(df_1D, df_images, 'phi_and_theta')
        X_data_A = actual_images.astype(np.float32)
        X_data_B = df_1D[included_features]
    else:
        num_phi_bins = int(all_stats[list(all_stats.keys())[0]]['num_phi_bins'])
        num_theta_bins = int(all_stats[list(all_stats.keys())[0]]['num_theta_bins'])
        # df_1D, y_col_values, bins_and_values, df_images = get_everything_ready(parent_folder_name, data_folder_1D, dataset_1D, included_features,  bin_type, num_phi_bins, num_theta_bins)
        df_1D, actual_images = prepare_test_dataset(folder_1D_data, dataset_name, folder_image_data, bin_type, included_features,  bin_type, num_phi_bins, num_theta_bins)
        X_data_A = actual_images.astype(np.float32)
        X_data_B = df_1D[included_features]
        true_y = np.asarray(df_1D.drop(included_features, axis=1))


    all_predictions = {}

    ''' getting predictions from each model '''
    for key in all_models.keys():
        predictions = all_models[key].predict((X_data_A, X_data_B))
        all_predictions[key] = predictions
        
        '''checking out the results from each model'''
        # for key, value in all_stats[key].items(): print(key, ":", value)        
        # ensembled_confusion_matrix(np.argmax(true_y, axis=1), np.argmax(predictions, axis=1), figname=None, folder=None)


    if(weighted == True):
        """ getting weights based on each metric"""
        weights, total = get_binned_ensembling_weights(all_stats, metric)
        final_predictions = np.zeros(true_y.shape)
        for key in all_predictions.keys():
            for i in range(final_predictions.shape[0]):
                final_predictions[i] += all_predictions[key][i] * (weights[key] / total)
    else:
        """ not doing the weighted metric just leaves us with one option - just get the average/sum of the predictions """
        final_predictions = np.zeros(true_y.shape)
        for key in all_predictions.keys():
            for i in range(final_predictions.shape[0]):
                final_predictions[i] += all_predictions[key][i]
    return final_predictions, df_1D, true_y
        
folder_1D_data = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/"
dataset_name = "TRAIN_OG_dataframe.csv"
folder_image_data = "new_dataset/Visible_cracks_new_dataset_3"
# dataset = "TRAIN_OG_dataframe.csv"
model_folder = '/Users/jakehirst/Desktop/sfx/binning/2in_2out_BINNED_phi_phi2_theta0'
model_folder = '/Users/jakehirst/Desktop/sfx/binning/2in_2out_BINNED_theta_phi0_theta2_categorical_crossentropy'
# model_folder = '/Users/jakehirst/Desktop/sfx/binning/2in_2out_BINNED_clusters_phiNone_thetaNone_categorical_crossentropy'

#all_metrics = ['train_cat_cross', 'train_accuracy', 'val_cat_cross', 'val_accuracy', 'test_cat_cross', 'test_accuracy']
final_predictions, df_1D, true_y = ensemble_binned_predictions(folder_1D_data, dataset_name, folder_image_data, model_folder, weighted=False, metric=None) 
ensembled_confusion_matrix(np.argmax(true_y, axis=1), np.argmax(final_predictions, axis=1), figname=None, folder=None)

final_predictions, df_1D, true_y = ensemble_binned_predictions(folder_1D_data, dataset_name, folder_image_data, model_folder, weighted=True, metric='test_accuracy') 
ensembled_confusion_matrix(np.argmax(true_y, axis=1), np.argmax(final_predictions, axis=1), figname=None, folder=None)

final_predictions, df_1D, true_y = ensemble_binned_predictions(folder_1D_data, dataset_name, folder_image_data, model_folder, weighted=True, metric='test_cat_cross') 
ensembled_confusion_matrix(np.argmax(true_y, axis=1), np.argmax(final_predictions, axis=1), figname=None, folder=None)

final_predictions, df_1D, true_y = ensemble_binned_predictions(folder_1D_data, dataset_name, folder_image_data, model_folder, weighted=True, metric='val_accuracy') 
ensembled_confusion_matrix(np.argmax(true_y, axis=1), np.argmax(final_predictions, axis=1), figname=None, folder=None)

final_predictions, df_1D, true_y = ensemble_binned_predictions(folder_1D_data, dataset_name, folder_image_data, model_folder, weighted=True, metric='val_cat_cross') 
ensembled_confusion_matrix(np.argmax(true_y, axis=1), np.argmax(final_predictions, axis=1), figname=None, folder=None)

print("done")