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
from Binning_phi_and_theta import *
import csv
from keras.losses import categorical_crossentropy
import sys
from k_means_clustering import *


""" This code was made while referencing https://towardsdatascience.com/neural-networks-with-multiple-data-sources-ef91d7b4ad5a """


def get_1D_inputs(folder, dataset, label_to_predict):
    df = pd.read_csv(folder + dataset, index_col = [0])
    return df

def remove_unpredicted_labels(df, label_to_predict):
    if(label_to_predict == "phi_and_theta"):
        labels = ["height", "x", "y", "z"]
    else:
        labels = ["height", "phi", "theta", "x", "y", "z"]

    """ drops all of the labesl that are not the one we are trying to predict """
    for label in labels:
        if((not label == label_to_predict) and df.columns.__contains__(label)):
            df = df.drop(label, axis=1)
    return df

def remove_features(df, features_to_remove=[], features_to_keep=[]):
    labels = ["theta", "phi", "height", "x", "y", "z"]
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
        labels = train_1D.columns[train_1D.columns.to_list().index('0'):].to_list()
        feature_list = train_1D.columns[:train_1D.columns.to_list().index('0')].to_list()
        # train_1D = remove_unpredicted_labels(train_1D, label_to_predict)
        # feature_list = [ elem for elem in train_1D.columns.tolist() if not labels.__contains__(elem)]
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
    
    if(label_to_predict == "phi_and_theta"):
        y_col = ["phi", "theta"]
    elif(label_to_predict == "x_and_y"):
        y_col = ["x", "y"]
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
    labels = test_1D.columns[test_1D.columns.to_list().index('0'):].to_list()
    feature_list = test_1D.columns[:test_1D.columns.to_list().index('0')].to_list()
    X_test_B = test_1D.drop(labels, axis=1)

    y_test = test_1D.get(labels).to_numpy()
        
    if(label_to_predict == "phi_and_theta"):
        labels = labels
        test_image_dataset = remove_unpredicted_labels(test_images, label_to_predict).drop("index", axis=1)
    
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255, #all pixel values set between 0 and 1 instead of 0 and 255.
    )
    
    if(label_to_predict == "phi_and_theta"):
        y_col = ["phi", "theta"]
    elif(label_to_predict == "x_and_y"):
        y_col = ["x", "y"]
    else:
        y_col = ['phi']
        
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

def write_csv_data(bin_type, included_features, num_phi_bins, num_theta_bins, val_predictions, test_predictions, train_predictions, y_train, y_val, y_test, saving_folder, fold_no):
    def get_accuracy(true_labels, sigmoid_preds):
        binary_preds = (sigmoid_preds >= 0.5).astype(int)
        accuracy = np.mean(binary_preds == np.asarray(true_labels))
        return accuracy
    
    def get_categorical_cross_entropy(true_labels, sigmoid_preds):
        crossentropy = categorical_crossentropy(np.asarray(true_labels), sigmoid_preds)
        return np.average(crossentropy)
    
    data = {'bin_type': bin_type,
            'included_features': included_features,
            'num_phi_bins': num_phi_bins,
            'num_theta_bins': num_theta_bins,
            'train_cat_cross': get_categorical_cross_entropy(y_train, train_predictions),
            'train_accuracy': get_accuracy(y_train, train_predictions),
            'val_cat_cross': get_categorical_cross_entropy(y_val, val_predictions),
            'val_accuracy': get_accuracy(y_val, val_predictions),
            'test_cat_cross': get_categorical_cross_entropy(y_test, test_predictions),
            'test_accuracy': get_accuracy(y_test, test_predictions),
            }
    csv_file_path = saving_folder + f'/fold{fold_no}_model_stats.csv'
    fieldnames = list(data.keys())
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['metric', 'value'])
        for key, value in data.items():
            writer.writerow([key, value])
    
def conf_mtx(y_true, y_pred, figname=None, folder=None):
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

def run_kfold(bin_type, parent_folder_name, simple_df_1D, lossfunc, saving_folder, num_phi_bins=None, num_theta_bins=None, num_clusters=None):
    if(not os.path.exists(saving_folder)):
        os.mkdir(saving_folder)    
    print("getting image data... ")
    args = prepare_data(parent_folder_name, ["OG"])
    df_images = get_image_inputs(args[0], args[1], args[2], args[3])
    print("syncing data...")
    phiandtheta_df, df_images = sync_image_and_1D_inputs(simple_df_1D, df_images)
    df_images['index'] = phiandtheta_df['index'] #making the index column the same for clustering purposes

    if(bin_type == "solid center phi and theta"):
        if(num_phi_bins == None or num_theta_bins == None): 
            print('\nmust specify number of phi and theta bins')
            sys.exit()
        phiandtheta_df = phiandtheta_df.drop("height", axis=1)
        df_1D, y_col_values, bins_and_values = Bin_phi_and_theta_center_target(phiandtheta_df, num_phi_bins, num_theta_bins)
        folder = f"/Users/jakehirst/Desktop/sfx/captain_america_plots_new_data/center_circle_phi_{num_phi_bins}_theta_{num_theta_bins}_bins/"
    elif(bin_type == "phi and theta"):
        if(num_phi_bins == None or num_theta_bins == None): 
            print('\nmust specify number of phi and theta bins')
            sys.exit()
        phiandtheta_df = phiandtheta_df.drop("height", axis=1)
        df_1D, y_col_values, bins_and_values = Bin_phi_and_theta(phiandtheta_df, num_phi_bins, num_theta_bins)
        folder = f"/Users/jakehirst/Desktop/sfx/captain_america_plots_new_data/phi_{num_phi_bins}_theta_{num_theta_bins}_bins/"
    elif(bin_type == "theta"):
        if(num_theta_bins == None): 
            print('\nmust specify number of theta bins')
            sys.exit()
        phiandtheta_df = phiandtheta_df.drop("height", axis=1)
        df_1D, y_col_values, bins_and_values = Bin_just_theta(phiandtheta_df, num_theta_bins)
        folder = f"/Users/jakehirst/Desktop/sfx/captain_america_plots_new_data/JustTheta_{num_theta_bins}_bins/"
    elif(bin_type == "phi"):
        if(num_phi_bins == None): 
            print('\nmust specify number of phi bins')
            sys.exit()
        phiandtheta_df = phiandtheta_df.drop("height", axis=1)
        df_1D, y_col_values, bins_and_values = Bin_just_phi(phiandtheta_df, num_phi_bins)
        folder = f"/Users/jakehirst/Desktop/sfx/captain_america_plots_new_data/JustPhi_{num_phi_bins}_bins/"
    elif(bin_type == 'clusters'):
        if(num_clusters == None): 
            print('\nmust specify number of clusters bins')
            sys.exit()
        phiandtheta_df = phiandtheta_df.drop("height", axis=1)
        df_1D, clusters, y_col_values = main_clustering_call(phiandtheta_df, num_clusters, 1000, saving_folder)
        df_images = df_images.sort_values('index').reset_index(drop=True)
        df_1D = df_1D.sort_values('index').reset_index(drop=True)
        print("done clustering")

    df_1D = df_1D.drop("index", axis=1)
    rnge = range(1, len(df_1D)+1)
    kf5 = KFold(n_splits=5, shuffle=True)
    
    fold_no = 1
    all_test_predictions = [] #test predictions from each of the folds
    all_test_labels = []
    """ running CNN for each kfold """
    for train_index, test_index in kf5.split(rnge):
        train_1D = df_1D.loc[train_index]
        train_images = df_images.loc[train_index]
        test_1D = df_1D.loc[test_index]
        test_images = df_images.loc[test_index]
        
        
        
        (train_1D, val_1D, train_images, val_images) = train_test_split(train_1D, train_images, test_size=0.2, random_state=42)

        
        csv_output, csv_input = CNN_1D(train_1D, label_to_predict, batch_size=5, kernel_size=(3,3))
        image_output, actual_train_images, actual_val_images, resnet_model, input = CNN_img(train_images, val_images, label_to_predict, batch_size=5, kernel_size=(3,3))
        
        print(csv_output.shape)
        print(image_output.shape)
        x = tf.keras.layers.concatenate([image_output, csv_output], name="concat_csv_img")

        predictions = tf.keras.layers.Dense(units= len(y_col_values), activation='softmax')(x)
        # model = tf.keras.Model(inputs = [resnet_model.input,csv_input], outputs = [predictions])
        model = tf.keras.Model(inputs = [input, csv_input], outputs = [predictions])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),#can define learning rate here
        loss = lossfunc,
        metrics = ['acc'],
        )

        # X_train_A = np.asarray(actual_train_images.get("image")).astype(np.float32)
        # X_val_A = np.asarray(actual_val_images.get("image")).astype(np.float32)
        
        #TODO: try to make the image data into numpy arrays with dtype float32 ( arr.astype(np.float32) )
        labels = train_1D.columns[train_1D.columns.to_list().index('0'):].to_list()
        feature_list = train_1D.columns[:train_1D.columns.to_list().index('0')].to_list()
        
        X_train_A = actual_train_images.astype(np.float32)
        X_val_A = actual_val_images.astype(np.float32)
        X_train_B = train_1D[feature_list]
        X_val_B = val_1D[feature_list]
        y_train = train_1D[labels]
        y_val = val_1D[labels]

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
        model.save(saving_folder + f"/trained_model_fold_{fold_no}.h5")
        val_predictions = model.predict((X_val_A, X_val_B))
        test_predictions = model.predict((X_test_A, X_test_B))
        train_predicitons = model.predict((X_train_A, X_train_B))
        included_features = X_train_B.columns.to_list()
        write_csv_data(bin_type, included_features, num_phi_bins, num_theta_bins, val_predictions, test_predictions, train_predicitons, y_train, y_val, y_test, saving_folder, fold_no)

        
        print("minimum MAE: ")
        print(min(history.history['loss']))
        print("minimum validation MAE: ")
        print(min(history.history['val_loss']))
        
        if(not os.path.exists(saving_folder)):
            os.mkdir(saving_folder)
        folder_path = saving_folder + f"/fold{fold_no}"
        if(not os.path.exists(folder_path)):
            os.mkdir(folder_path)
        
        test_predictions = np.squeeze(model.predict((X_test_A, X_test_B)))
        training_predictions = np.squeeze(model.predict((X_train_A, X_train_B)))
        all_test_predictions.append((test_predictions, test_images["image_path"].to_list()))

        
        """ gets r^2 value of the test dataset with the predictions made from above ^ """
        metric = tfa.metrics.r_square.RSquare()
        metric.update_state(y_train, training_predictions)
        training_result = metric.result()
        print("Training R^2 = " + str(training_result.numpy()))
        
        fold_folder = folder_path
        if(not os.path.isdir(fold_folder.split("/fold")[0])):
            os.mkdir(fold_folder.split("/fold")[0])
        if not os.path.isdir(fold_folder.removesuffix("/")):
            os.mkdir(fold_folder.removesuffix("/"))

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
        plt.title("theta")
        plt.legend()
        plt.grid(True)
        # plt.text(.5, .0001, f"Train R^2 = {str(training_result.numpy())}, Test R^2 = {str(test_result.numpy())}")
        plt.savefig(folder_path + "/loss_vs_epochs")
        plt.close()

        if(bin_type == 'clusters'):
            print("do somethig here")
        else:
            for i in range(len(test_predictions)):
                make_circle(bins_and_values, test_predictions[i], test_images['image_path'].tolist()[i], fold_folder+"/")
        
        fold_no +=1
        print("done")
    with open(saving_folder + "/clusters_means.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['clusters'])
        writer.writerow([str(clusters[2])])


        




    
# """   ********* phi and theta **********   """
# folder = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/"
# dataset = "TRAIN_OG_dataframe.csv"
# label_to_predict = "phi_and_theta"
# patience = 100
# max_epochs = 2000

# """
# HEIGHT:
# init y
# crack Len
# linearity
# max thickness
# max_kink
# abs val mean kink

# PHI:
# front 0 x
# front 0 z
# front 1 y
# front 1 z
# init y
# linearity
# angle_btw

# THETA:
# front 0 y
# front 0 z
# init y
# init z
# angle btw
# """

# """ getting 1D features """
# print("getting 1D data... ")
# df_1D = get_1D_inputs(folder, dataset, label_to_predict)      
# height_p_less_point5 = ["init y", "crack len", "linearity", "max thickness", "max_kink", "abs_val_mean_kink", "abs_val_sum_kink"]    
# phi_p_less_point5 = ["front 0 x", "front 0 z", "front 1 y", "front 1 z", "init y", "linearity"]
# theta_p_less_point5 = ["front 0 y", "front 0 z", "front 1 y", "front 1 z", "init y", "init z", "max_kink", "angle_btw"]
# phi_and_theta_p_less_point5 = ["front 0 x", "front 0 y", "front 0 z", "front 1 y", "front 1 z", "init y", "init z", "linearity", "max_kink", "angle_btw"]

# simple_df_1D = remove_features(df_1D, features_to_keep=phi_and_theta_p_less_point5)
# # simple_df_1D = remove_features(df_1D, features_to_remove=[])

# parent_folder_name = "new_dataset/Original"
# parent_folder_name = "new_dataset/Visible_cracks_new_dataset_3"

# # lossfunc = 'mean_distance_error_phi_theta',
# # lossfunc = 'mean_absolute_error',
# # lossfunc = 'mean_squared_error',
# # lossfunc = 'mean_squared_logarithmic_error',
# # lossfunc = 'categorical_crossentropy'

# bin_type = "phi and theta"
# bin_type = "theta"
# bin_type = "phi"
# bin_type = "solid center phi and theta"
# bin_type = "clusters"




# bin_type = "clusters"
# bin_type_no_space = bin_type.replace(" ", "_")
# num_theta_bins = None
# num_phi_bins = None
# num_clusters = 2
# lossfunc = 'categorical_crossentropy'
# saving_folder = f"/Users/jakehirst/Desktop/sfx/binning/2in_2out_BINNED_{bin_type_no_space}_phi{num_phi_bins}_theta{num_theta_bins}_{lossfunc}"
# run_kfold(bin_type, parent_folder_name, simple_df_1D, lossfunc, saving_folder, num_phi_bins=num_phi_bins, num_theta_bins=num_theta_bins, num_clusters=num_clusters)


# bin_type = "theta"
# bin_type_no_space = bin_type.replace(" ", "_")
# num_theta_bins = 3
# num_phi_bins = 0
# num_clusters = None
# lossfunc = 'categorical_crossentropy'
# saving_folder = f"/Users/jakehirst/Desktop/sfx/binning/test"
# run_kfold(bin_type, parent_folder_name, simple_df_1D, lossfunc, saving_folder, num_phi_bins=num_phi_bins, num_theta_bins=num_theta_bins)
