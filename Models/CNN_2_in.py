import tensorflow as tf
from tensorflow import keras
import pandas as pd
from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import r2_score
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
from CNN_function_library import *
import tensorflow.keras.backend as K
from PIL import Image
from quaternions import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
import csv
from datetime import date





""" This code was made while referencing https://towardsdatascience.com/neural-networks-with-multiple-data-sources-ef91d7b4ad5a """


def get_1D_inputs(folder, dataset, label_to_predict):
    df = pd.read_csv(folder + dataset, index_col = [0])
    return df

def remove_unpredicted_labels(df, label_to_predict):
    if(label_to_predict == "phi_and_theta"):
        labels = ["height", "x", "y", "z"]
    elif(label_to_predict == 'quaternions'):
        labels = ["height","phi", "theta", "x", "y", "z"]
    else:
        labels = ["height", "phi", "theta", "x", "y", "z"]

    """ drops all of the labesl that are not the one we are trying to predict """
    for label in labels:
        if((not label == label_to_predict) and df.columns.__contains__(label)):
            df = df.drop(label, axis=1)
    return df

def remove_features(df, features_to_remove=[], features_to_keep=[]):
    labels = ["theta", "phi", "height", "x", "y", "z", 'quats']
    if(len(features_to_remove) == 0 and len(features_to_keep)>0):
        print("remove all features except listed")
        new_df = df.copy()
        for feature in df.columns:
            if(features_to_keep.__contains__(feature) or labels.__contains__(feature)):
                continue
            else:
                new_df = new_df.drop(feature, axis=1)
        
    elif(len(features_to_remove) > 0 and len(features_to_keep)==0):
        print("remove listed features")
        new_df = df.copy()
        for feature in features_to_remove:
            new_df = new_df.drop(feature, axis=1)
        
    else:
        print("keep all features")
        new_df = df
    return new_df
    
def get_image_inputs(image_paths, heights, phis, thetas):
    
    image_df = pd.DataFrame(list(zip(image_paths, heights, phis, thetas)))
    image_df.columns = ["image_path", "height", "phi", "theta"]
    
    return image_df

""" only uses the 1D rows that correspond to the image rows and orders the two datafames in the same order as well. """
def sync_image_and_1D_inputs(df_1D, df_images):
    
    trimmed_df_1D = pd.DataFrame(columns=df_1D.columns.tolist())
    df_image_labels = df_images.drop("image_path", axis=1)

    for img_row in df_image_labels.iterrows():
        height = img_row[1]["height"]
        phi = img_row[1]["phi"]
        theta = img_row[1]["theta"]
        row_1D = df_1D.query(f"height == {height} and phi == {phi} and theta == {theta}")
        if(row_1D.shape[0] == 0): 
            df_images = df_images.drop(img_row[0], axis=0)
            continue
        
        trimmed_df_1D = pd.concat([trimmed_df_1D, row_1D]) 
        
    trimmed_df_1D = trimmed_df_1D.reset_index()
    df_images = df_images.reset_index()

    
    return trimmed_df_1D, df_images 
    

def CNN_1D(train_1D, label_to_predict, batch_size=5, kernel_size=(3,3)):
    if(label_to_predict == "phi_and_theta"):
        labels = ["phi", "theta"]
        train_1D = remove_unpredicted_labels(train_1D, label_to_predict)
        feature_list = [ elem for elem in train_1D.columns.tolist() if not labels.__contains__(elem)]
        train_features = train_1D[feature_list].drop("index", axis=1)
        train_labels = train_1D[labels]
    elif(label_to_predict == 'quaternions'):
        labels = ['quats']
        train_1D = remove_unpredicted_labels(train_1D, label_to_predict)
        feature_list = [ elem for elem in train_1D.columns.tolist() if not labels.__contains__(elem)]
        train_features = train_1D[feature_list].drop("index", axis=1)
        train_labels = train_1D[labels]
    elif(label_to_predict == 'height' or label_to_predict == 'phi' or label_to_predict == 'theta'):
        labels = [label_to_predict]
        train_1D = remove_unpredicted_labels(train_1D, label_to_predict)
        feature_list = [ elem for elem in train_1D.columns.tolist() if not labels.__contains__(elem)]
        if(train_1D.columns.__contains__('index')):
            train_features = train_1D[feature_list].drop("index", axis=1)
        else:
            train_features = train_1D[feature_list]
        train_labels = train_1D[labels]

    
    #quote from tensorflow:
    """One reason this is important is because the features are multiplied by the model weights. So, the scale of the outputs and the scale of the gradients are affected by the scale of the inputs.
    Although a model might converge without feature normalization, normalization makes training much more stable"""
    """ normalizing features and labels """

    normalizer = tf.keras.layers.Normalization(axis=-1) #creating normalization layer
    normalizer.adapt(np.array(train_features)) #fitting the state of the preprocessing layer
        
    numfeatures = len(train_features.columns)

    """############## 1D model ##############"""
    csv_data_shape = train_features.shape[1]
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
    
def CNN_img(train_image_dataset, val_image_dataset, label_to_predict, batch_size=5, kernel_size=(3,3)):
    if(label_to_predict == "phi_and_theta"):
        labels = ["phi", "theta"]
        train_image_dataset = remove_unpredicted_labels(train_image_dataset, label_to_predict).drop("index", axis=1)
        val_image_dataset = remove_unpredicted_labels(val_image_dataset, label_to_predict).drop("index", axis=1)
        # feature_list = [ elem for elem in train_1D.columns.tolist() if not labels.__contains__(elem)]
        # train_features = train_1D[feature_list]
        # train_labels = train_1D[labels]
    """ ##################### TRYING WITH IMAGEDATAGENERATOR'S ##################### """
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255, #all pixel values set between 0 and 1 instead of 0 and 255.
    )
    
    val_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255, #all pixel values set between 0 and 1 instead of 0 and 255.
    )
    
    """ I don think this matters since we get the outputs somehwere else. """
    if(label_to_predict == "phi_and_theta" or label_to_predict == "quaternions"):
        y_col = ["phi", "theta"]
    elif(label_to_predict == "x_and_y"):
        y_col = ["x", "y"]
    elif(label_to_predict == 'height' or label_to_predict == 'phi' or label_to_predict == 'theta'):
        y_col = [label_to_predict]
        
    #flow the images through the generators
    flow_train_images = train_generator.flow_from_dataframe(
        dataframe=train_image_dataset,
        x_col = 'image_path',
        y_col = y_col,
        # target_size=args[4][:2],  #can reduce the images to a certain size to reduce training time. 120x120 for example here
        color_mode='rgb',
        class_mode = 'raw', #keeps the classes of our labels the same after flowing
        batch_size=batch_size, #can increase this to up to like 10 or so for how much data we have
        shuffle=False,
        seed=42,
    )

    flow_val_images = train_generator.flow_from_dataframe(
        dataframe=val_image_dataset,
        x_col = 'image_path',
        y_col = y_col,
        # target_size=args[4][:2],  #can reduce the images to a certain size to reduce training time. 120x120 for example here
        color_mode='rgb',
        class_mode = 'raw', #keeps the classes of our labels the same after flowing
        batch_size=batch_size, #can increase this to up to like 10 or so for how much data we have
        shuffle=False,
        seed=42,
    )
    """ ##################### TRYING WITH IMAGEDATAGENERATOR'S ##################### """
    
    
    
    """ ##################### TRYING WITH numpy array'S ##################### """
    #pulling the image matricies from the filepaths of images
    train_inputs = []
    val_inputs = []

    #puttin the rgb matricies into train_inputs, test_inputs, and val_inputs
    #to see the image, use plt.imshow(arr)

    for image in flow_train_images._filepaths:
        arr = imageio.v2.imread(image)[:,:,0:3] / 255.0
        train_inputs.append(arr)
    for image in flow_val_images._filepaths:
        arr = imageio.v2.imread(image)[:,:,0:3] / 255.0
        val_inputs.append(arr)

    actual_train_images = np.array(train_inputs)
    actual_val_images = np.array(val_inputs)
    
    """ ##################### TRYING WITH numpy array'S ##################### """



    """############## image resnet model ##############"""
    """ include_top = true for classification tasks, and youd also have to define classes=3 for the number of classes. """
    # resnet_model = tf.keras.applications.resnet50.ResNet50(weights= None, include_top=False, input_shape=(642, 802,3)) #TODO might just be (642, 802)
    # resnet_output = resnet_model.output
    # img_output = tf.keras.layers.GlobalAveragePooling2D()(resnet_output)
    # img_output = tf.keras.layers.Dense(8, activation= 'relu', name="img_out")(img_output)
    
    resnet_model = []
    input = tf.keras.layers.Input(shape=(642, 802, 3), name="img_input")
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(642, 802, 3))(input)
    pool1 = tf.keras.layers.MaxPooling2D(2)(conv1)
    # conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(642, 802, 3))(pool1)
    # pool2 = tf.keras.layers.MaxPooling2D(2)(conv2)

    flat = tf.keras.layers.Flatten()(pool1)
    
    # dense1 = tf.keras.layers.Dense(128, activation='relu')(flat)
    img_output = tf.keras.layers.Dense(8, activation='relu')(flat)
    
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(642, 802, 3)))
    # model.add(tf.keras.layers.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.layers.Flatten())
    # model.add(tf.keras.layers.layers.Dense(64, activation='relu'))
    """############## image resnet model ##############"""

    return img_output, actual_train_images, actual_val_images, resnet_model, input

def organize_test_data(test_1D, test_images, label_to_predict):
    if(label_to_predict == "phi_and_theta"):
        X_test_B = test_1D.drop(["height", "phi", "theta"], axis=1).drop("index", axis=1)
        y_test = test_1D.get(["phi", "theta"]).to_numpy()
    elif(label_to_predict == 'quaternions'):
        X_test_B = test_1D.drop(["height", "phi", "theta", 'quats'], axis=1).drop("index", axis=1)
        y_test = np.asarray(test_1D['quats'].tolist())
    elif(label_to_predict == 'height' or label_to_predict == 'phi' or label_to_predict == 'theta'):
        X_test_B = test_1D.drop([label_to_predict], axis=1).drop("index", axis=1)
        y_test = test_1D.get([label_to_predict]).to_numpy()
        
    if(label_to_predict == "phi_and_theta"):
        labels = ["phi", "theta"]
        test_image_dataset = remove_unpredicted_labels(test_images, label_to_predict).drop("index", axis=1)
    elif(label_to_predict == 'quaternions'):
        labels = ['quats']
        test_image_dataset = remove_unpredicted_labels(test_images, label_to_predict).drop("index", axis=1)
        test_image_dataset['quats'] = np.zeros(len(test_image_dataset))
    elif(label_to_predict == 'height' or label_to_predict == 'phi' or label_to_predict == 'theta'):
        labels = [label_to_predict]
        test_image_dataset = remove_unpredicted_labels(test_images, label_to_predict).drop("index", axis=1)

    
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255, #all pixel values set between 0 and 1 instead of 0 and 255.
    )
    
    if(label_to_predict == "phi_and_theta"):
        y_col = ["phi", "theta"]
    elif(label_to_predict == "x_and_y"):
        y_col = ["x", "y"]
    elif(label_to_predict == "quaternions"):
        y_col = ['quats']
    elif(label_to_predict == 'height' or label_to_predict == 'phi' or label_to_predict == 'theta'):
        y_col = [label_to_predict]


        
        
    #flow the images through the generators
    flow_test_images = test_generator.flow_from_dataframe(
        dataframe=test_image_dataset,
        x_col = 'image_path',
        y_col = y_col,
        # target_size=args[4][:2],  #can reduce the images to a certain size to reduce training time. 120x120 for example here
        color_mode='rgb',
        class_mode = 'raw', #keeps the classes of our labels the same after flowing
        # batch_size=batch_size, #can increase this to up to like 10 or so for how much data we have
        shuffle=False,
        seed=42,
    )
    
    test_inputs = []

    #puttin the rgb matricies into train_inputs, test_inputs, and val_inputs
    #to see the image, use plt.imshow(arr)

    for image in flow_test_images._filepaths:
        arr = imageio.v2.imread(image)[:,:,0:3] / 255.0
        test_inputs.append(arr)

    X_test_A = np.array(test_inputs)
    
    return X_test_A, X_test_B, y_test

def create_combined_model(train_1D, train_images, val_images, label_to_predict, lossfunc):
    csv_output, csv_input = CNN_1D(train_1D, label_to_predict, batch_size=5, kernel_size=(3,3))
    image_output, actual_train_images, actual_val_images, resnet_model, input = CNN_img(train_images, val_images, label_to_predict, batch_size=5, kernel_size=(3,3))
    
    print(csv_output.shape)
    print(image_output.shape)
    x = tf.keras.layers.concatenate([image_output, csv_output], name="concat_csv_img")

    if(label_to_predict == "phi_and_theta"):
        predictions = tf.keras.layers.Dense(units=2)(x)
    elif(label_to_predict == 'quaternions'):
        predictions = tf.keras.layers.Dense(units=4)(x)
    elif(label_to_predict == 'height' or label_to_predict == 'phi' or label_to_predict == 'theta'):
        predictions = tf.keras.layers.Dense(units=1)(x)

    # model = tf.keras.Model(inputs = [resnet_model.input,csv_input], outputs = [predictions])
    model = tf.keras.Model(inputs = [input, csv_input], outputs = [predictions])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),#can define learning rate here
    loss = lossfunc,
    )
    return model, actual_train_images, actual_val_images

def plot_things(history, y_train, training_predictions, y_test, test_predictions, folder_path, fold_no, label_to_predict):
    """ gets r^2 value of the test dataset with the predictions made from above ^ """
    metric = tfa.metrics.r_square.RSquare()
    metric.update_state(y_train, training_predictions)
    training_result = metric.result()
    print("Training R^2 = " + str(training_result.numpy()))
    

    """ gets r^2 value of the training dataset """
    metric = tfa.metrics.r_square.RSquare()
    metric.update_state(y_test, test_predictions)
    test_result = metric.result()
    print("Test R^2 = " + str(test_result.numpy()))
    
    plt.plot(history.history['loss'], label='loss (mean absolute error)')
    plt.plot(history.history['val_loss'], label='val_loss')
    #plt.ylim([0, 4])
    plt.xlabel(f'Train R^2 = {str(training_result.numpy())}, Test R^2 = {str(test_result.numpy())}')
    plt.ylabel('loss')
    plt.title(f"parody plot for predicting {label_to_predict}")
    plt.legend()
    plt.grid(True)
    # plt.text(.5, .0001, f"Train R^2 = {str(training_result.numpy())}, Test R^2 = {str(test_result.numpy())}")
    plt.savefig(folder_path + f"/loss_vs_epochs_fold{fold_no}")
    # plt.show()
    plt.close()
    
    if(label_to_predict == 'quaternions' or label_to_predict == 'phi_and_theta'):
        a = plt.axes(aspect='equal')
        plt.scatter(y_test[:,0], test_predictions[:,0])
        plt.xlabel('True labels')
        plt.ylabel('Predicted labels')
        plt.title("phi")
        lims = [0, max(y_test[:,0])]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)
        plt.savefig(folder_path + f"/phi_predictions_fold{fold_no}")
        # plt.show()
        plt.close()

        a = plt.axes(aspect='equal')
        plt.scatter(y_test[:,1], test_predictions[:,1])
        plt.xlabel('True labels')
        plt.ylabel('Predicted labels')
        plt.title("theta")
        lims = [0, max(y_test[:,1])]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)
        plt.savefig(folder_path + f"/theta_predictions_fold{fold_no}")
        # plt.show()
        plt.close()
        
    elif(label_to_predict == 'height' or label_to_predict == 'phi' or label_to_predict == 'theta'):
        a = plt.axes(aspect='equal')
        plt.scatter(y_test, test_predictions)
        plt.xlabel('True labels')
        plt.ylabel('Predicted labels')
        plt.title(label_to_predict + f'fold{fold_no}')
        lims = [0, max(y_test[:,0])]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)
        plt.savefig(folder_path + f"/{label_to_predict}_predictions_fold{fold_no}")
        # plt.show()
        plt.close()

def principal_component_analysis(df_1D, label_to_predict , pca):
    features = df_1D.columns.tolist()
    features.remove('height')
    features.remove('phi')
    features.remove('theta')
    x = StandardScaler().fit_transform(df_1D.loc[:,features].values)
    PCA_s = PCA(n_components=pca)
    principalComponents = PCA_s.fit_transform(x)
    
    col = []
    for i in range(pca):
        col.append(f"PrincComp_{i}")
    
    principalDf = pd.DataFrame(data = principalComponents
             , columns = col)
    
    labels = df_1D[['height', 'phi', 'theta']]
    finalDf = pd.concat([principalDf, labels], axis = 1)
    return finalDf

def get_model_metrics(true_y, predictions, num_features):
    y_test = true_y
    test_predictions = predictions
    mae = mean_absolute_error(y_test, test_predictions)
    mse = mean_squared_error(y_test, test_predictions)
    if(np.any(test_predictions < 0)): #if there are any negative numbers, you cannot calculate msle
        msle = None
    else:
        msle = mean_squared_log_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)
    n = len(y_test)
    p = num_features # assuming test_predictions is a 2D array
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    return {'mae': mae, 'mse':mse, 'msloge':msle, 'r^2':r2, 'adj_r^2':adj_r2}

def save_model_metrics(y_train, y_val, y_test, training_predictions, val_predictions, test_predictions, label_to_predict, folder_path, fold_no, num_features):
    if(label_to_predict == "quaternions"):
        phi_train_metrics = get_model_metrics(y_train[:,0], training_predictions[:,0], num_features)
        phi_train_metrics = {'phi_train_' + key: value for key, value in phi_train_metrics.items()}
        phi_val_metrics = get_model_metrics(y_val[:,0], val_predictions[:,0], num_features)
        phi_val_metrics = {'phi_val_' + key: value for key, value in phi_val_metrics.items()}
        phi_test_metrics = get_model_metrics(y_test[:,0], test_predictions[:,0], num_features)
        phi_test_metrics = {'phi_test_' + key: value for key, value in phi_test_metrics.items()}
        
        theta_train_metrics = get_model_metrics(y_train[:,1], training_predictions[:,1], num_features)
        theta_train_metrics = {'theta_train_' + key: value for key, value in theta_train_metrics.items()}
        theta_val_metrics = get_model_metrics(y_val[:,1], val_predictions[:,1], num_features)
        theta_val_metrics = {'theta_val_' + key: value for key, value in theta_val_metrics.items()}
        theta_test_metrics = get_model_metrics(y_test[:,1], test_predictions[:,1], num_features)
        theta_test_metrics = {'theta_test_' + key: value for key, value in theta_test_metrics.items()}
        # Append the dictionaries together
        model_stats = {'included_features': df_1D.columns.to_list(), 
                        'date_created': date.today(),
                        'pca':pca,
                        'loss_func': lossfunc,
                        'label_to_predict':label_to_predict,
                        **phi_train_metrics, 
                        **phi_val_metrics, 
                        **phi_test_metrics, 
                        **theta_train_metrics,
                        **theta_val_metrics,
                        **theta_test_metrics}
        # Write the merged dictionary to a CSV file
        with open(folder_path + f'/fold{fold_no}_model_stats.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['metric', 'value'])
            for key, value in model_stats.items():
                writer.writerow([key, value])
        
    else:
        train_metrics = get_model_metrics(y_train, training_predictions, num_features)
        train_metrics = {'train_' + key: value for key, value in train_metrics.items()}
        val_metrics = get_model_metrics(y_val, val_predictions, num_features)
        val_metrics = {'val_' + key: value for key, value in val_metrics.items()}
        test_metrics = get_model_metrics(y_test, test_predictions, num_features)
        test_metrics = {'test_' + key: value for key, value in test_metrics.items()}
        # Append the dictionaries together
        model_stats = {'included_features': df_1D.columns.to_list(), 
                        'date_created': date.today(),
                        'pca':pca,
                        'loss_func': lossfunc,
                        'label_to_predict':label_to_predict,
                        **train_metrics, 
                        **val_metrics, 
                        **test_metrics}
        # Write the merged dictionary to a CSV file
        with open(folder_path + f'/fold{fold_no}_model_stats.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['metric', 'value'])
            for key, value in model_stats.items():
                writer.writerow([key, value])
        


def run_kfold(simple_df_1D, saving_folder, lossfunc, label_to_predict, parent_folder_name, pca=False):

    print("getting image data... ")
    args = prepare_data(parent_folder_name, ["OG"])
    df_images = get_image_inputs(args[0], args[1], args[2], args[3])
    print("syncing data...")
    df_1D, df_images = sync_image_and_1D_inputs(simple_df_1D, df_images)
    
    if(pca != False):
        print(f"doing pca with {pca} components")
        df_1D = principal_component_analysis(df_1D, label_to_predict, pca)
        df_1D['index'] = 0 #adding this cause I was lazy before and didnt fix the index row till later...
    else:
        print("skipping pca")
        
    if(label_to_predict == "phi_and_theta"):
        print("df is good")
    elif(label_to_predict == "quaternions"):
        df_1D = height_phi_theta_df_to_quaternions(df_1D)
    elif(label_to_predict == 'height' or label_to_predict == 'phi' or label_to_predict == 'theta'):
        labels_to_remove = ['height', 'phi', 'theta']
        labels_to_remove.remove(label_to_predict)
        df_1D = df_1D.drop(labels_to_remove, axis=1)
        

    rnge = range(1, len(df_1D)+1)
    kf5 = KFold(n_splits=5, shuffle=True)
    
    fold_no = 1
    """ running CNN for each kfold """
    for train_index, test_index in kf5.split(rnge):
        train_1D = df_1D.loc[train_index]
        train_images = df_images.loc[train_index]
        test_1D = df_1D.loc[test_index]
        test_images = df_images.loc[test_index]
        
        
        
        (train_1D, val_1D, train_images, val_images) = train_test_split(train_1D, train_images, test_size=0.2, random_state=42)

        
        model, actual_train_images, actual_val_images = create_combined_model(train_1D, train_images, val_images, label_to_predict, lossfunc)

        # X_train_A = np.asarray(actual_train_images.get("image")).astype(np.float32)
        # X_val_A = np.asarray(actual_val_images.get("image")).astype(np.float32)
        
        #TODO: try to make the image data into numpy arrays with dtype float32 ( arr.astype(np.float32) )

        
        if(label_to_predict == "phi_and_theta"):
            y_train = train_1D.get(["phi", "theta"]).to_numpy()
            y_val = val_1D.get(["phi", "theta"]).to_numpy()
            X_train_A = actual_train_images.astype(np.float32)
            X_val_A = actual_val_images.astype(np.float32)
            X_train_B = train_1D.drop(["height", "phi", "theta"], axis=1).drop("index", axis=1)
            X_val_B = val_1D.drop(["height", "phi", "theta"], axis=1).drop("index", axis=1)
            
        elif(label_to_predict == 'quaternions'):
            X_train_A = actual_train_images.astype(np.float32)
            X_val_A = actual_val_images.astype(np.float32)
            X_train_B = train_1D.drop(["height", "phi", "theta"], axis=1).drop("index", axis=1)
            X_val_B = val_1D.drop(["height", "phi", "theta"], axis=1).drop("index", axis=1)
            y_train = np.asarray(train_1D['quats'].tolist())
            y_val = np.asarray(val_1D['quats'].tolist())
            X_val_B = X_val_B.drop('quats', axis=1)
            X_train_B = X_train_B.drop('quats', axis=1)
            
        elif(label_to_predict == 'height' or label_to_predict == 'phi' or label_to_predict == 'theta'):
            X_train_A = actual_train_images.astype(np.float32)
            X_val_A = actual_val_images.astype(np.float32)
            X_train_B = train_1D.drop([label_to_predict], axis=1).drop("index", axis=1)
            X_val_B = val_1D.drop([label_to_predict], axis=1).drop("index", axis=1)
            y_train = train_1D.get([label_to_predict]).to_numpy()
            y_val = val_1D.get([label_to_predict]).to_numpy()



        history = model.fit((X_train_A, X_train_B), 
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
                            validation_data=((X_val_A, X_val_B), y_val),
                            verbose=1,
                            )
        
        X_test_A, X_test_B, y_test = organize_test_data(test_1D, test_images, label_to_predict)
        
        #saving the model for later use. to load the model again, use syntax loaded_model = tf.keras.models.load_model(MODEL_PATH) 
        model.save(saving_folder + f"/trained_model_fold_{fold_no}.h5")
        

        
        
        print("minimum MAE: ")
        print(min(history.history['loss']))
        print("minimum validation MAE: ")
        print(min(history.history['val_loss']))
        
        if(not os.path.exists(saving_folder)):
            os.mkdir(saving_folder)
        folder_path = saving_folder
        if(not os.path.exists(folder_path)):
            os.mkdir(folder_path)
            

        test_predictions = model.predict((X_test_A, X_test_B))
        training_predictions = model.predict((X_train_A, X_train_B))
        val_predictions =  model.predict((X_val_A, X_val_B))
        
        
        if(label_to_predict == 'quaternions'):
            test_predictions = quaternions_back_to_sphereical(test_predictions)
            training_predictions = quaternions_back_to_sphereical(training_predictions)
            val_predictions = quaternions_back_to_sphereical(val_predictions)
            y_test = quaternions_back_to_sphereical(y_test)
            y_val = quaternions_back_to_sphereical(y_val)
            y_train = quaternions_back_to_sphereical(y_train)
            plot_things(history, y_train, training_predictions, y_test, test_predictions, folder_path, fold_no, label_to_predict)
        else:
            plot_things(history, y_train, training_predictions, y_test, test_predictions, folder_path, fold_no, label_to_predict)

        save_model_metrics(y_train, y_val, y_test, training_predictions, val_predictions, test_predictions, label_to_predict, folder_path, fold_no, num_features=len(X_test_B.columns))
        
        fold_no +=1
        print("done")



    
# """   ********* phi and theta **********   """
# folder = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/"
# dataset = "OG_dataframe.csv"
# patience = 250
# max_epochs = 3000


# label_to_predict = "quaternions"
# # label_to_predict = "height"
# # label_to_predict = "phi"
# # label_to_predict = "theta"
# # label_to_predict = "phi_and_theta"


# """ getting 1D features """
# print("getting 1D data... ")
# df_1D = get_1D_inputs(folder, dataset, label_to_predict)          
# height_p_less_point5 = ["init y", "crack len", "linearity", "max thickness", "max_kink", "abs_val_mean_kink", "abs_val_sum_kink"]    
# phi_p_less_point5 = ["front 0 x", "front 0 z", "front 1 y", "front 1 z", "init y", "linearity", "angle_btw"]
# theta_p_less_point5 = ["front 0 y", "front 0 z", "init y", "init z", "angle_btw"]
# phi_and_theta_p_less_point5 = set(phi_p_less_point5) | set(theta_p_less_point5)
# # simple_df_1D = remove_features(df_1D, features_to_keep=phi_and_theta_p_less_point5)
# simple_df_1D = remove_features(df_1D, features_to_remove=[])
# # simple_df_1D = remove_features(df_1D, features_to_remove=[])

# parent_folder_name = "new_dataset/Original"
# parent_folder_name = "new_dataset/Visible_cracks"
# parent_folder_name = "new_dataset/Visible_cracks_new_dataset_2"

# """ loss function options """
# # lossfunc = 'mean_distance_error_phi_theta',
# # lossfunc = 'mean_absolute_error', 
# # lossfunc = 'mean_squared_error',
# # lossfunc = 'mean_squared_logarithmic_error', - BAD FOR [phi]
# # lossfunc = tf.keras.losses.CosineSimilarity(axis=1) - BAD FOR [phi, theta]
# # lossfunc = tf.keras.losses.Huber()
# # lossfunc = tf.keras.losses.LogCosh()

# # pca = 5
# pca = False








# general_saving_folder = f"/Users/jakehirst/Desktop/sfx/regression_for_ensembling/2in_regression_{label_to_predict.upper()}_all_feats_PCA_{str(pca)}"
# # saving_folder = f"/Users/jakehirst/Desktop/sfx/regression/TEST_SAVING_MODELS"


# # lossfunc = 'mean_absolute_error'
# # saving_folder = general_saving_folder + "_" + lossfunc
# # run_kfold(simple_df_1D, saving_folder, lossfunc, label_to_predict, parent_folder_name, pca)

# # lossfunc = 'mean_squared_error'
# # saving_folder = general_saving_folder + "_" + lossfunc
# # run_kfold(simple_df_1D, saving_folder, lossfunc, label_to_predict, parent_folder_name, pca)

# # lossfunc = 'mean_squared_logarithmic_error'
# # saving_folder = general_saving_folder + "_" + lossfunc
# # run_kfold(simple_df_1D, saving_folder, lossfunc, label_to_predict, parent_folder_name, pca)


# # lossfunc = tf.keras.losses.CosineSimilarity(axis=1)
# # saving_folder = general_saving_folder + "_CosineSimilarity"
# # run_kfold(simple_df_1D, saving_folder, lossfunc, label_to_predict, parent_folder_name, pca)


# lossfunc = tf.keras.losses.Huber()
# saving_folder = general_saving_folder + "_Huber"
# run_kfold(simple_df_1D, saving_folder, lossfunc, label_to_predict, parent_folder_name, pca)


# lossfunc = tf.keras.losses.LogCosh()
# saving_folder = general_saving_folder + "_LogCosh"
# run_kfold(simple_df_1D, saving_folder, lossfunc, label_to_predict, parent_folder_name, pca)



