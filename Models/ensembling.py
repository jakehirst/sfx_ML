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
from CNN_2_in import *

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

''' getting the 1D and image dataset ready to test after ensembling '''
def prepare_test_dataset(folder_1D_data, dataset_name, folder_image_data, label_to_predict):
    simple_df_1D = get_1D_inputs(folder_1D_data, dataset_name, label_to_predict)   

    print("getting image data... ")
    args = prepare_data(folder_image_data, ["OG"])
    df_images = get_image_inputs(args[0], args[1], args[2], args[3])
    print("syncing data...")
    df_1D, df_images = sync_image_and_1D_inputs(simple_df_1D, df_images)
    
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

def get_weights(ensemble_stats, predictions, metric, threshold):
    #getting all of the weights of the models based on the metric provided
    weights = {}
    if(metric.endswith("r^2")):
        new_ensemble_stats = ensemble_stats.copy()
        new_predictions = predictions.copy()
        if(threshold != None): #filtering out the really bad models based on the threshold
            for key in ensemble_stats.keys():
                if(float(ensemble_stats[key]) < threshold):
                    del new_ensemble_stats[key]
                    del new_predictions[key]
        total_r2 = 0.0
        for value in new_ensemble_stats.values(): total_r2 += float(value)
        for key in new_ensemble_stats.keys(): 
            weights[key] = float(new_ensemble_stats[key]) / total_r2
    
    elif(metric.endswith('mae') or metric.endswith('mse')):
        new_ensemble_stats = ensemble_stats.copy()
        new_predictions = predictions.copy()
        total_inv = 0.0
        for value in new_ensemble_stats.values(): total_inv += 1/float(value)
        for key in new_ensemble_stats.keys(): 
            weights[key] = (1/float(new_ensemble_stats[key])) / total_inv
    
    return weights, new_predictions, new_ensemble_stats

''' plots a parody plot of the ensemble predictions vs true values '''
def parody_plot(y_, ensemble_predictions, metric, label_to_predict):
    y = y_.reshape(y_.shape[0],)
    ensemble_pred = ensemble_predictions.reshape(ensemble_predictions.shape[0],)
    
    a = plt.axes(aspect='equal')
    plt.scatter(y, ensemble_pred)
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.title(f"ensembling on {metric} parody plot for predicting {label_to_predict}")
    lims = [0, max(y)]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.savefig('/Users/jakehirst/Desktop/sfx/regression_for_ensembling' + f'/{metric}')
    # plt.show()
    plt.close()

''' ensembles based on a single metric '''
def ensemble_with_single_metric(df_1D, df_images, path_to_model_folders, model_folders, metric, label_to_predict, threshold=None):
    quaternions = False
    labels = ['height', 'phi', 'theta']
    X__Images = df_images.astype(np.float32)
    X__1D = df_1D.drop(labels, axis=1).drop("index", axis=1)
    y_ = df_1D.get([label_to_predict]).to_numpy()
    
    predictions = {}
    ensemble_stats = {}
    
    for folder in model_folders:
        folder = path_to_model_folders + folder
        models, stats = load_models_and_stats(folder)
        for i in range(1, len(models)+1): #loading each model, and getting the predictions for each model
            model_name = folder + f' fold {i}'
            model = models[model_name]
            stat = stats[model_name]
            included_features = eval(stat['included_features'])
            for item in labels:
                if item in included_features: included_features.remove(item)
            X__1D = X__1D[included_features]
            
            model_predictions = model.predict((X__Images, X__1D))
            predictions[model_name] = model_predictions
          
    # translating all of the preditions made by quaternions back into phi/theta
    for key in predictions.keys():
        if(key.__contains__("QUATERNIONS")): 
            temp_predictions = quaternions_back_to_sphereical(predictions[key])
            if(label_to_predict == 'theta'):
                predictions[key] = temp_predictions[:,1].reshape(len(temp_predictions), 1)
            elif(label_to_predict == 'phi'):
                predictions[key] = temp_predictions[:,0].reshape(len(temp_predictions), 1)
            else:
                print('\nMAYBE CHECK THE LABEL TO PREDICT\n')
                break
    
        
    metrics = ['test_r^2', 'test_adj_r^2', 'test_mse', 'test_mae', 
               'val_r^2', 'val_adj_r^2', 'val_mse', 'val_mae',] 
               #'train_r^2', 'train_adj_r^2', 'train_mse', 'train_mae']  
    for metric in metrics:
        for i in range(1, len(models)+1): 
            model_name = folder + f' fold {i}'
            if(model_name.__contains__("QUATERNIONS")):#quaternions have different metrics for both phi and theta.
                specific_metric = label_to_predict + "_" + metric
                ensemble_stats[model_name] = stats[model_name][specific_metric]
            else:
                specific_metric = metric
                ensemble_stats[model_name] = stats[model_name][specific_metric]

        # the actual ensembling portion
        weights, new_predictions, new_ensemble_stats = get_weights(ensemble_stats, predictions, metric, threshold)
        ensemble_predictions = np.zeros((len(y_),1))
        for model in new_ensemble_stats.keys():
            ensemble_predictions += new_predictions[model] * weights[model]
        # the actual ensembling portion

        
        ensemble_metrics = get_model_metrics(y_, ensemble_predictions, len(included_features))
        print(f"\nmetrics for ensembling on {specific_metric}\n")
        print(ensemble_metrics)
        parody_plot(y_, ensemble_predictions, specific_metric, label_to_predict)
    

    
    
    print("here")
            
            
            
            
            
            
        
        
            

            
            
"""
metrics to choose to ensemble from as of now:
- 'r^2'
- 'adj_r^2'

"""
path_to_model_folders = '/Users/jakehirst/Desktop/sfx/regression_for_ensembling/'   


model_folders = [#'2in_regression_THETA_all_feats_PCA_False_Huber',
# '2in_regression_THETA_all_feats_PCA_False_Huber_LogCosh',
# '2in_regression_THETA_all_feats_PCA_False_mean_absolute_error',
# '2in_regression_THETA_all_feats_PCA_False_mean_absolute_error_mean_squared_error',
# '2in_regression_THETA_all_feats_PCA_False_mean_squared_logarithmic_error',
'2in_regression_QUATERNIONS_all_feats_PCA_False_CosineSimilarity',
# '2in_regression_QUATERNIONS_all_feats_PCA_False_mean_absolute_error',
# '2in_regression_QUATERNIONS_all_feats_PCA_False_mean_squared_error',
# '2in_regression_QUATERNIONS_all_feats_PCA_False_mean_squared_logarithmic_error'
]

label_to_predict = 'theta'

#loading the 1D dataset that we need to test the ensemble on
folder_1D_data = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/"
dataset_name = "OG_dataframe.csv"
print("getting 1D data... ")

#loading the image dataset that we need to test the ensemble on
folder_image_data = "new_dataset/Visible_cracks_new_dataset_2"



df_1D, df_images = prepare_test_dataset(folder_1D_data, dataset_name, folder_image_data, label_to_predict)

metric = 'test_r^2'
ensemble_with_single_metric(df_1D, df_images, path_to_model_folders, model_folders, metric, label_to_predict, threshold=0.1)
