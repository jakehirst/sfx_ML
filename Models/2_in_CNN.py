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
        if(row_1D.shape == (0, 11)): 
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
    csv_model = tf.keras.layers.Dense(64, activation='relu', name="csv_dense")(csv_model)
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
        shuffle=True,
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
        shuffle=True,
        seed=42,
    )
    """ ##################### TRYING WITH IMAGEDATAGENERATOR'S ##################### """
    
    
    
    """ ##################### TRYING WITH numpy array'S ##################### """
    # cols = train_image_dataset.columns.to_list()
    # cols.remove("image_path")
    # cols.insert(0,"image")
    # actual_train_images = pd.DataFrame(columns=cols)
    # actual_val_images = pd.DataFrame(columns=cols)

    # for row in train_image_dataset.iterrows():
    #     path = row[1]["image_path"]
    #     image = Image.open(path)
    #     img_data = np.asarray(image, dtype="float32") /255.0
    #     img_data = img_data.tolist()
    #     new_row = {"image": img_data}
    #     for label in cols[1:]:
    #         val = row[1][label]
    #         new_row[label] = val
    #     actual_train_images = actual_train_images.append(new_row, ignore_index=True)
        
    # for row in val_image_dataset.iterrows():
    #     path = row[1]["image_path"]
    #     image = Image.open(path)
    #     img_data = np.asarray(image, dtype="float32") /255.0
    #     img_data = img_data.tolist()
    #     new_row = {"image": img_data}
    #     for label in cols[1:]:
    #         val = row[1][label]
    #         new_row[label] = val
    #     actual_val_images = actual_val_images.append(new_row, ignore_index=True)
        
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
    resnet_model = tf.keras.applications.resnet50.ResNet50(weights= None, include_top=False, input_shape=(642, 802,3)) #TODO might just be (642, 802)
    resnet_output = resnet_model.output
    img_output = tf.keras.layers.GlobalAveragePooling2D()(resnet_output)
    img_output = tf.keras.layers.Dense(64, activation= 'softmax', name="img_out")(img_output)
    
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(642, 802, 3)))
    # model.add(tf.keras.layers.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.layers.Flatten())
    # model.add(tf.keras.layers.layers.Dense(64, activation='relu'))
    """############## image resnet model ##############"""

    return img_output, actual_train_images, actual_val_images, resnet_model


    
    
    
"""   ********* phi and theta **********   """
folder = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/"
dataset = "OG_dataframe.csv"
label_to_predict = "phi_and_theta"
saving_folder = "/Users/jakehirst/Desktop/sfx/regression/multi_regression_2_out_3-29_no_features_out_dropout"
patience = 20
max_epochs = 100

""" getting 1D features """
print("getting 1D data... ")
df_1D = get_1D_inputs(folder, dataset, label_to_predict)          
theta_p_less_point5 = ["front 0 y", "front 0 z", "init y", "init z", "dist btw frts"]
phi_p_less_point5 = ["front 0 x", "front 0 z", "front 1 z"]
simple_df_1D = remove_features(df_1D, features_to_keep=["front 0 x", "front 0 y", "front 0 z", "front 1 z", "init y", "init z", "dist btw frts", "angle_btw"])
# simple_df_1D = remove_features(df_1D, features_to_remove=[])

parent_folder_name = "new_dataset/Original"
parent_folder_name = "new_dataset/Visible_cracks"

print("getting image data... ")
args = prepare_data(parent_folder_name, ["OG"])
df_images = get_image_inputs(args[0], args[1], args[2], args[3])
print("syncing data...")
df_1D, df_images = sync_image_and_1D_inputs(simple_df_1D, df_images)

rnge = range(1, len(df_1D)+1)
kf5 = KFold(n_splits=5, shuffle=True)

""" running CNN for each kfold """
for train_index, test_index in kf5.split(rnge):
    train_1D = df_1D.loc[train_index]
    train_images = df_images.loc[train_index]
    test_1D = df_1D.loc[test_index]
    test_images = df_images.loc[test_index]
    
    
    
    (train_1D, val_1D, train_images, val_images) = train_test_split(train_1D, train_images, test_size=0.2, random_state=42)

    
    csv_output, csv_input = CNN_1D(train_1D, label_to_predict, batch_size=5, kernel_size=(3,3))
    image_output, actual_train_images, actual_val_images, resnet_model = CNN_img(train_images, val_images, label_to_predict, batch_size=5, kernel_size=(3,3))
    
    print(csv_output.shape)
    print(image_output.shape)
    x = tf.keras.layers.concatenate([image_output, csv_output], name="concat_csv_img")

    predictions = tf.keras.layers.Dense(units=2)(x)
    model = tf.keras.Model(inputs = [resnet_model.input,csv_input], outputs = [predictions])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),#can define learning rate here
        loss="mae",
    )

    # X_train_A = np.asarray(actual_train_images.get("image")).astype(np.float32)
    # X_val_A = np.asarray(actual_val_images.get("image")).astype(np.float32)
    
    #TODO: try to make the image data into numpy arrays with dtype float32 ( arr.astype(np.float32) )
    X_train_A = actual_train_images
    X_val_A = actual_val_images
    X_train_B = train_1D.drop(["height", "phi", "theta"], axis=1).drop("index", axis=1)
    X_val_B = val_1D.drop(["height", "phi", "theta"], axis=1).drop("index", axis=1)
    
    if(label_to_predict == "phi_and_theta"):
        y_train = train_1D.get(["phi", "theta"]).to_numpy()
        y_val = val_1D.get(["phi", "theta"]).to_numpy()

    history = model.fit((X_train_A, X_train_B), 
                        y_train, 
                        epochs=20, 
                        validation_data=((X_val_A, X_val_B), y_val))

    # history = model.fit(
    #     actual_train_images,
    #     validation_data=val_images,
    #     epochs=max_epochs, #max number of epochs to go over data
    #     #this callback makes the learning stop if the validation loss stops improving for 'patience' epochs in a row. very useful tool should use in other models
    #     callbacks=[
    #         tf.keras.callbacks.EarlyStopping(
    #             # monitor='loss',
    #             monitor='val_loss',
    #             patience=patience,
    #             restore_best_weights=True
    #         )
    #     ],
    #     verbose=1
    # )

    

(train_1D, test_1D, train_images, test_images) = train_test_split(df_1D, df_images, test_size=0.2, random_state=42)

intermediate_out_1D = CNN_1D(train_1D, label_to_predict, batch_size=5, patience=50, max_epochs=1000, kernel_size=(3,3)) 
intermediate_out_image = CNN_img(train_images, label_to_predict, batch_size=5, patience=50, max_epochs=1000, kernel_size=(3,3))
print("done")


