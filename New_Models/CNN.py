import tensorflow as tf
from tensorflow import keras
import pandas as pd
from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import r2_score, precision_score, confusion_matrix, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import KFold
import keras_tuner as kt
from keras_tuner import RandomSearch
from keras_tuner import BayesianOptimization
import matplotlib.image as img
import imageio
# from CNN_function_library import *
import tensorflow.keras.backend as K
from PIL import Image
# from quaternions import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
import csv
from datetime import date
from prepare_data import *
from keras.utils import to_categorical
import seaborn as sns


''' 
CNN architecture before output layer for 1D features
'''
def CNN_1D(train_features_1D):
    #quote from tensorflow:
    """One reason this is important is because the features are multiplied by the model weights. So, the scale of the outputs and the scale of the gradients are affected by the scale of the inputs.
    Although a model might converge without feature normalization, normalization makes training much more stable"""
    """ normalizing features and labels """

    normalizer = tf.keras.layers.Normalization(axis=-1) #creating normalization layer
    normalizer.adapt(np.array(train_features_1D)) #fitting the state of the preprocessing layer
        
    numfeatures = len(train_features_1D.columns)

    """############## 1D model ##############"""
    csv_data_shape = train_features_1D.shape[1]
    csv_input = tf.keras.layers.Input(shape=csv_data_shape, name="csv")
    csv_model = normalizer(csv_input)
    csv_model = tf.keras.layers.Dense(256, activation='relu', name="csv_dense1")(csv_model)
    csv_model = tf.keras.layers.Dense(128, activation='relu', name="csv_dense2")(csv_model)
    csv_model = tf.keras.layers.Dense(64, activation='relu', name="csv_dense3")(csv_model)
    csv_model = tf.keras.layers.Dense(32, activation='relu', name="csv_dense4")(csv_model)
    csv_model = tf.keras.layers.Dense(16, activation='relu', name="csv_dense5")(csv_model)
    csv_model = tf.keras.layers.Dense(8, activation='relu', name="csv_dense6")(csv_model)
    csv_output = tf.keras.layers.Dropout(0.5, name="csv_output")(csv_model)
    """############## 1D model ##############"""

    return csv_output, csv_input

""" 
CNN architecture before output layer for image features
image shape should be (642, 802, 3) 
returns the input 
"""
def CNN_image(image_shape):
    input = tf.keras.layers.Input(shape=image_shape, name="img_input")
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape)(input)
    pool1 = tf.keras.layers.MaxPooling2D(2)(conv1)
    # conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape)(pool1)
    # pool2 = tf.keras.layers.MaxPooling2D(2)(conv2)

    flat = tf.keras.layers.Flatten()(pool1)
    
    # dense1 = tf.keras.layers.Dense(128, activation='relu')(flat)
    img_output = tf.keras.layers.Dense(8, activation='relu')(flat)
    return input, img_output


def run_kfold_CNN(full_dataset, raw_images, full_dataset_labels, patience, max_epochs, saving_folder='/Users/jakehirst/Desktop/model_results'):
    #turning labels into one-hot vectors
    full_dataset_labels = to_categorical(full_dataset_labels)

    #setting aside a test dataset
    np.random.seed(6) #this should reset the randomness to the same randomness so that the test_indicies are the same throughout the tests
    test_indicies = np.random.choice(np.arange(0, len(full_dataset)), size=30, replace=False) #30 for the test dataset
    test_df = full_dataset.iloc[test_indicies]
    test_images = raw_images[test_indicies]
    y_test = full_dataset_labels[test_indicies]
    full_dataset = full_dataset.drop(test_indicies, axis=0)
    raw_images = np.delete(raw_images, test_indicies, axis=0)
    full_dataset_labels = np.delete(full_dataset_labels, test_indicies, axis=0)

    models = []
    
    rnge = range(1, len(full_dataset)+1)
    kf5 = KFold(n_splits=5, shuffle=True)
    fold_no = 1
    for train_index, val_index in kf5.split(rnge):
        train_df = full_dataset.iloc[train_index]
        train_images = raw_images[train_index]
        y_train = full_dataset_labels[train_index]
        val_df = full_dataset.iloc[val_index]
        val_images = raw_images[val_index]
        y_val = full_dataset_labels[val_index]
        
        csv_output, csv_input = CNN_1D(train_df)
        image_input, image_output= CNN_image(val_images[0].shape)
        x = tf.keras.layers.concatenate([image_output, csv_output], name="concat_csv_img")
        x = tf.keras.layers.Dense(units= 32, activation='relu')(x)
        predictions = tf.keras.layers.Dense(units= full_dataset_labels.shape[1], activation='softmax')(x) 
        model = tf.keras.Model(inputs = [image_input, csv_input], outputs = [predictions])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),#can define learning rate here
        loss = 'categorical_crossentropy',
        metrics = ['acc'],
        )
    
        history = model.fit((train_images, train_df), 
                        y_train, 
                        epochs=max_epochs, 
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(
                                # monitor='loss',
                                monitor='val_loss',
                                patience=patience,
                                restore_best_weights=True
                            )
                        ],
                        validation_data=((val_images, val_df), y_val),
                        verbose=1,
                        )

        model.save(saving_folder + f"/trained_model_fold_{fold_no}.h5")        
        print("here")
        val_pred = model.predict([val_images, val_df])
        test_pred = model.predict([test_images, test_df])

        val_pred_for_metrics = np.argmax(val_pred, axis=1).reshape((len(val_pred), 1))
        y_val_for_metrics = np.argmax(y_val, axis=1).reshape((len(y_val), 1))
        test_pred_for_metrics = np.argmax(test_pred, axis=1).reshape((len(test_pred), 1))
        y_test_for_metrics = np.argmax(y_test, axis=1).reshape((len(y_test), 1))
        
        val_precision = precision_score(y_val_for_metrics, val_pred_for_metrics, average='weighted')
        val_f1 = f1_score(y_val_for_metrics, val_pred_for_metrics, average='weighted')
        val_recall = recall_score(y_val_for_metrics, val_pred_for_metrics, average='weighted')
        val_accuracy = np.count_nonzero(val_pred_for_metrics == y_val_for_metrics) / len(y_val_for_metrics)
        val_conf_matrx = confusion_matrix(y_val_for_metrics, val_pred_for_metrics)
        test_precision = precision_score(y_test_for_metrics, test_pred_for_metrics, average='weighted')
        test_recall = recall_score(y_test_for_metrics, test_pred_for_metrics, average='weighted')
        test_f1 = f1_score(y_test_for_metrics, test_pred_for_metrics, average='weighted')
        test_conf_matrx = confusion_matrix(y_test_for_metrics, test_pred_for_metrics)
        test_accuracy = np.count_nonzero(test_pred_for_metrics == y_test_for_metrics) / len(y_test_for_metrics)

        
        # open the file for writing
        with open(saving_folder + f"/model_metrics_fold_{fold_no}.csv", 'w', newline='') as file:
            writer = csv.writer(file)

            # write the header row
            writer.writerow(['dataset', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
            writer.writerow(['test', test_accuracy, test_precision, test_recall, test_f1])
            writer.writerow(['validation', val_accuracy, val_precision, val_recall, val_f1])

        # Create a heatmap using seaborn
        sns.heatmap(test_conf_matrx, annot=True, cmap='Blues', xticklabels=np.arange(0, test_pred.shape[1]), yticklabels=np.arange(0, test_pred.shape[1]))
        plt.title('Test dataset Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(saving_folder + f"/Test_confusion_matrix_fold_{fold_no}.png")
        plt.close()
        sns.heatmap(val_conf_matrx, annot=True, cmap='Blues', xticklabels=np.arange(0, val_pred.shape[1]), yticklabels=np.arange(0, val_pred.shape[1]))
        plt.title('Validation dataset Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(saving_folder + f"/Validation_confusion_matrix_fold_{fold_no}.png")
        plt.close()
            
        fold_no += 1
    
    return models

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
    corr_matrix, p_matrix, important_features = Pearson_correlation(dataset, label_to_predict, minimum_p_value=0.01)
    correlated_featureset = dataset[important_features]
    print(corr_matrix[important_features])
    print(p_matrix[important_features])
        
    return correlated_featureset, raw_images, full_dataset_labels

def prepare_dataset_discrete_Binning(full_dataset_pathname, image_folder, label_to_predict, num_bins=2, saving_folder=None):
    dataset = read_dataset(full_dataset_pathname)
    #adding images to the dataset
    raw_images = get_images_from_dataset(dataset, image_folder)
    new_label_to_predict = 'binned_' + label_to_predict
    dataset, bin_edges, counts = Discretely_bin_height(dataset, num_bins, new_label_to_predict, label_to_predict, saving_folder=None)
    corr_matrix, p_matrix, important_features = Pearson_correlation(dataset, label_to_predict, minimum_p_value=0.01)
    labels = ['height', 'phi', 'theta', 'impact_sites', new_label_to_predict]
    dataset = remove_unwanted_labels(dataset, new_label_to_predict, labels)
    full_dataset_labels = dataset[new_label_to_predict].to_numpy()
    for label in labels: 
        if(important_features.__contains__(label)): important_features.remove(label)

    correlated_featureset = dataset[important_features]
    print(corr_matrix[important_features])
    print(p_matrix[important_features])
    return correlated_featureset, raw_images, full_dataset_labels

    


full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/FULL_OG_dataframe_with_impact_sites.csv"
image_folder = '/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx/new_dataset/Visible_cracks'


''' preparing data for logistic regression on impact site using k-means clustering'''
# label_to_predict = 'impact_cluster_assignments'
# num_clusters = 5
# saving_folder=f'/Users/jakehirst/Desktop/model_results/impact_site_logi_reg_{num_clusters}_clusters/'
# if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
# correlated_featureset, raw_images, full_dataset_labels = prepare_dataset_Kmeans_cluster(full_dataset_pathname, image_folder, label_to_predict, cluster='impact_sites', num_clusters=num_clusters, saving_folder=saving_folder)
# run_kfold_CNN(correlated_featureset, raw_images, full_dataset_labels, patience=100, max_epochs=1000, saving_folder=saving_folder)

# ''' preparing data for logistic regression on height using discrete binning '''
# num_bins = 2
# saving_folder=f'/Users/jakehirst/Desktop/model_results/height_logi_reg_{num_bins}_bins/'
# if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
# correlated_featureset, raw_images, full_dataset_labels = prepare_dataset_discrete_Binning(full_dataset_pathname, image_folder, 'height', num_bins=num_bins, saving_folder=None)
# run_kfold_CNN(correlated_featureset, raw_images, full_dataset_labels, patience=100, max_epochs=1000, saving_folder=saving_folder)

