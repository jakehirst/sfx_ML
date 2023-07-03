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
from Metric_collection import *


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


'''
runs a Classification CNN on the given dataset, doing a 5 fold cross validation and testing it on a test dataset pulled from the full_dataset
'''
def run_kfold_Categorical_CNN(full_dataset, raw_images, full_dataset_labels, patience, max_epochs, saving_folder='/Users/jakehirst/Desktop/model_results'):
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

'''
runs a dual input (image and 1D features) Regression CNN on the given dataset, doing a 5 fold cross validation and testing it 
on a test dataset pulled from the full_dataset
'''
def run_kfold_Regression_CNN(full_dataset, raw_images, full_dataset_labels, patience, max_epochs, num_outputs=1, lossfunc='mae', saving_folder='/Users/jakehirst/Desktop/model_results', use_images=True):

    #setting aside a test dataset
    np.random.seed(6) #this should reset the randomness to the same randomness so that the test_indicies are the same throughout the tests
    test_indicies = np.random.choice(np.arange(0, len(full_dataset)), size=30, replace=False) #30 for the test dataset
    test_df = full_dataset.iloc[test_indicies]
    y_test = full_dataset_labels[test_indicies]
    full_dataset = full_dataset.drop(test_indicies, axis=0)
    full_dataset_labels = np.delete(full_dataset_labels, test_indicies, axis=0)
    
    if(use_images):
        test_images = raw_images[test_indicies]
        raw_images = np.delete(raw_images, test_indicies, axis=0)

    models = []
    
    rnge = range(1, len(full_dataset)+1)
    kf5 = KFold(n_splits=5, shuffle=True)
    fold_no = 1
    for train_index, val_index in kf5.split(rnge):
        train_df = full_dataset.iloc[train_index]
        y_train = full_dataset_labels[train_index]
        val_df = full_dataset.iloc[val_index]
        y_val = full_dataset_labels[val_index]
        
        csv_output, csv_input = CNN_1D(train_df)
        
        #use_images makes the CNN use the images in train_images/val_images as inputs 
        if(use_images):
            train_images = raw_images[train_index]
            val_images = raw_images[val_index]

            
            image_input, image_output= CNN_image(val_images[0].shape)
            x = tf.keras.layers.concatenate([image_output, csv_output], name="concat_csv_img")
            # x = tf.keras.layers.Dense(units= 32, activation='relu')(x)
            predictions = tf.keras.layers.Dense(units=num_outputs)(x) 
            model = tf.keras.Model(inputs = [image_input, csv_input], outputs = [predictions])
            
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(),#can define learning rate here
                loss = lossfunc,
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
            
        else:
            x = csv_output
            predictions = tf.keras.layers.Dense(units=num_outputs)(x) 
            model = tf.keras.Model(inputs = [csv_input], outputs = [predictions])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(),#can define learning rate here
                loss = lossfunc,
            )
        
            history = model.fit((train_df), 
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
                            validation_data=((val_df), y_val),
                            verbose=1,
                            )



        model.save(saving_folder + f"/trained_model_fold_{fold_no}.h5")        
        print("here")
        if(use_images):
            val_pred = model.predict([val_images, val_df])
            test_pred = model.predict([test_images, test_df])
        else:
            val_pred = model.predict([val_df])
            test_pred = model.predict([test_df])
        val_r2 = r2_score(y_val, val_pred)
        val_adj_r2 = adjusted_r2(y_val, val_pred, len(full_dataset), len(full_dataset.columns))
        val_mae = mean_absolute_error(y_val, val_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        val_rmse = np.sqrt(val_mse)
        test_r2 = r2_score(y_test, test_pred)
        test_adj_r2 = adjusted_r2(y_test, test_pred, len(full_dataset), len(full_dataset.columns))
        test_mae = mean_absolute_error(y_test, test_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        test_rmse = np.sqrt(test_mse)
        
        def parody_plot(true_values, predictions, file_to_save=None):
            true_values = true_values.reshape(predictions.shape)
            # Plot the parody plot
            plt.scatter(true_values, predictions, color='blue', label='True vs. Predicted')
            plt.plot(true_values, true_values, color='red', linestyle='--', label='Parity Line')
            # Add labels and title
            plt.ylabel('Predictions')
            plt.xlabel('True Values')
            plt.title('Parody Plot')

            # Add legend
            plt.legend()
            if(file_to_save == None):
                plt.show()
            else:
                plt.savefig(file_to_save)
            plt.close()
            
        parody_plot(y_test, test_pred, saving_folder + f'/parody_plot_test_fold{fold_no}.png')
        parody_plot(y_val, val_pred, saving_folder + f'/parody_plot_val_fold{fold_no}.png')
        
        # open the file for writing
        with open(saving_folder + f"/model_metrics_fold_{fold_no}.csv", 'w', newline='') as file:
            writer = csv.writer(file)

            #write the header row
            writer.writerow(['dataset', 'r^2', 'adj_r^2', 'MAE', 'MSE', 'RMSE'])
            writer.writerow(['test', test_r2, test_adj_r2, test_mae, test_mse, test_rmse])
            writer.writerow(['validation', val_r2, val_adj_r2, val_mae, val_mse, val_rmse])

            
        fold_no += 1
    
    return models

