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


""" Builds model for hyperparameter tuning """
#https://www.analyticsvidhya.com/blog/2021/06/create-convolutional-neural-network-model-and-optimize-using-keras-tuner-deep-learning/
def build_model(hp):
    # create model object
    model = keras.Sequential([
    #adding first convolutional layer    
    keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_1_filter', min_value=16, max_value=64, step=3),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        #activation function
        activation='relu',
        input_shape=(642, 802, 3)),
    keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_2_filter', min_value=16, max_value=64, step=3),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        #activation function
        activation='relu',
        input_shape=(642, 802, 3)),
    # adding second convolutional layer 
    keras.layers.MaxPooling2D(2),
    # keras.layers.Conv2D(
    #     #adding filter 
    #     filters=hp.Int('conv_3_filter', min_value=32, max_value=64, step=16),
    #     #adding filter size or kernel size
    #     kernel_size=hp.Choice('conv_3_kernel', values = [3,5]),
    #     #activation function
    #     activation='relu'
    # ),
    # adding flatten layer
    keras.layers.Flatten(),
    # adding dense layer    
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=16, max_value=64, step=3),
        activation='relu'
    ),
    keras.layers.Dense(
        units=hp.Int('dense_2_units', min_value=16, max_value=64, step=3),
        activation='relu'
    ),
    keras.layers.Dense(
        units=hp.Int('dense_3_units', min_value=16, max_value=64, step=3),
        activation='relu'
    ),
    # output layer    
    keras.layers.Dense(units=1)
    ])
    #compilation of model
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-3])),#[1e-2, 1e-3, 5e-4])),
              loss='mae',
              metrics=['mae'])
    return model







def make_CNN(args, label_to_predict, batch_size=5, patience=25, max_epochs=1000, optimizer="adam", activation="relu", kernel_size=(5,5), plot = True, augmentation_list = [], num_folds=5, storing_Dir="Desktop", Name="_"):
    results = []
    #TODO: do this for phi and theta later
    #shuffles the dataset and puts it into a dataframe
    if(label_to_predict == "height"):
        df = pd.concat([args[0], args[1]], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
        df.columns = ["Filepath", "height"]
        kfold = KFold(n_splits=num_folds, shuffle=True)
        inputs = np.array(df["Filepath"])
        outputs = np.array(df["height"])
    elif(label_to_predict == "phi"):
        df = pd.concat([args[0], args[2]], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
        df.columns = ["Filepath", "phi"]
        kfold = KFold(n_splits=num_folds, shuffle=True)
        inputs = np.array(df["Filepath"])
        outputs = np.array(df["phi"])
    elif(label_to_predict == "theta"):
        df = pd.concat([args[0], args[3]], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
        df.columns = ["Filepath", "theta"]
        kfold = KFold(n_splits=num_folds, shuffle=True)
        inputs = np.array(df["Filepath"])
        outputs = np.array(df["theta"])
    elif(label_to_predict == "x"):
        df = pd.concat([args[0], args[2], args[3]], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
        df.columns = ["Filepath", "phi", "theta"]
        df = convert_to_xy(df, label_to_predict)
        kfold = KFold(n_splits=num_folds, shuffle=True)
        inputs = np.array(df["Filepath"])
        outputs = np.array(df["x"])
    elif(label_to_predict == "y"):
        df = pd.concat([args[0], args[2], args[3]], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
        df.columns = ["Filepath", "phi", "theta"]
        df = convert_to_xy(df, label_to_predict)
        kfold = KFold(n_splits=num_folds, shuffle=True)
        inputs = np.array(df["Filepath"])
        outputs = np.array(df["y"])


    fold_no = 1
    for train, test in kfold.split(inputs, outputs):
        train_df = df.iloc[train]
        test_df = df.iloc[test]



        #train_df, test_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=1)

        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255, #all pixel values set between 0 and 1 instead of 0 and 255.
            #validation_split=0.2 #creating a validation split in our training generator
        )

        val_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255,
        )

        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255, #all pixel values set between 0 and 1 instead of 0 and 255.
        )

        #creating a validation set
        val_df = train_df.sample(frac=0.2)
        #removing the validation set from the training set
        train_df = train_df.drop(val_df.index)
        #adding data augmentation to training dataset
        train_df = add_augmentations(train_df, augmentation_list)


        #flow the images through the generators
        train_images = train_generator.flow_from_dataframe(
            dataframe=train_df,
            x_col = 'Filepath',
            y_col = label_to_predict,
            target_size=args[4][:2],  #can reduce the images to a certain size to reduce training time. 120x120 for example here
            color_mode='rgb',
            class_mode = 'raw', #keeps the classes of our labels the same after flowing
            batch_size=batch_size, #can increase this to up to like 10 or so for how much data we have
            shuffle=True,
            seed=42,
        )

        val_images = val_generator.flow_from_dataframe(
            dataframe=val_df,
            x_col = 'Filepath',
            y_col = label_to_predict,
            target_size=args[4][:2], #can reduce the images to a certain size to reduce training time. 120x120 for example here
            color_mode='rgb',
            class_mode = 'raw', #keeps the classes of our labels the same after flowing
            batch_size=batch_size, #can increase this to up to like 10 or so for how much data we have
            shuffle=True,
            seed=42,
        )

        test_images = test_generator.flow_from_dataframe(
            dataframe=test_df,
            x_col = 'Filepath',
            y_col = label_to_predict,
            target_size=args[4][:2],  #can reduce the images to a certain size to reduce training time. 120x120 for example here
            color_mode='rgb',
            class_mode = 'raw', #keeps the classes of our labels the same after flowing
            batch_size=batch_size, #can increase this to up to like 10 or so for how much data we have
            shuffle=False
        )


        #pulling the image matricies from the filepaths of images
        test_inputs = []
        train_inputs = []
        val_inputs = []

        #puttin the rgb matricies into train_inputs, test_inputs, and val_inputs
        #to see the image, use plt.imshow(arr)
        for image in test_images._filepaths:
            arr = imageio.v2.imread(image)[:,:,0:3] / 255.0
            test_inputs.append(arr)
        for image in train_images._filepaths:
            arr = imageio.v2.imread(image)[:,:,0:3] / 255.0
            train_inputs.append(arr)
        for image in val_images._filepaths:
            arr = imageio.v2.imread(image)[:,:,0:3] / 255.0
            val_inputs.append(arr)

        test_inputs = np.array(test_inputs)
        train_inputs = np.array(train_inputs)
        val_inputs = np.array(val_inputs)
        #https://datascience.stackexchange.com/questions/106600/how-to-perform-regression-on-image-data-using-tensorflow
       
        """ Random search hyperparameter tuner """
        # tuner = RandomSearch(build_model,
        #             objective='val_mae',
        #             max_trials = 20)

        """ Baysian Optimization hyperparameter tuner """
        tuner = BayesianOptimization(
            build_model,
            objective='val_loss',
            max_trials=25,
            overwrite = True, #need this otherwise it will compare to previous trials done in the past
            directory = storing_Dir + "/" + Name + "_tb"
        )

        #https://datascience.stackexchange.com/questions/106600/how-to-perform-regression-on-image-data-using-tensorflow

        tuner.search(train_inputs, 
                     train_images._targets, 
                     epochs=6, 
                     validation_data=(val_inputs, val_images._targets),
                     callbacks=[keras.callbacks.TensorBoard(storing_Dir + "/" + Name + "_tb_logs")]
                     )
        # model=tuner.get_best_models(num_models=1)[0]
        # #summary of best model
        # model.summary()

        # model.compile( 
        #     optimizer=tf.keras.optimizers.Adam(),
        #     loss='mae' #could change back to mae
        # )


        # history = model.fit(
        #     train_images,
        #     validation_data=val_images,
        #     epochs=max_epochs, #max number of epochs to go over data
        #     #this callback makes the learning stop if the validation loss stops improving for 'patience' epochs in a row. very useful tool should use in other models
        #     callbacks=[
        #         tf.keras.callbacks.EarlyStopping(
        #             monitor='val_loss',
        #             patience=patience,
        #             restore_best_weights=True
        #         )
        #     ],
        #     verbose=1
        # )


        test_labels = test_images.labels
        train_labels = train_images.labels

        # """ makes predictions with the test dataset and plots them. Good predictions should lie on the line. """
        test_predictions = np.squeeze(model.predict(test_images))

        """ gets r^2 value of the test dataset with the predictions made from above ^ """
        metric = tfa.metrics.r_square.RSquare()
        metric.update_state(test_labels, test_predictions)
        training_result = metric.result()
        print("Train R^2 = " + str(training_result.numpy()))

        """ gets r^2 value of the training dataset """
        training_predictions = model.predict(train_images).flatten()
        metric = tfa.metrics.r_square.RSquare()
        metric.update_state(train_labels, training_predictions)
        test_result = metric.result()
        print("Test R^2 = " + str(test_result.numpy()))
        #return {"Training R^2": training_result.numpy(), "Test R^2": test_result.numpy(), "history": history}
        print("min val loss = " + str(min(history.history['val_loss'])))
        print("min loss = " + str(min(history.history['loss'])))

        #if(plot == True):
            #plot_stuff(test_images, train_images, history, training_result.numpy() , test_result.numpy(), test_predictions, augmentation_list, label_to_predict)
        results.append([training_result.numpy(), test_result.numpy(), min(history.history['val_loss']), min(history.history['loss'])])
    results = np.array(results)
    avg_training_r = np.average(results[:,0])
    avg_test_r = np.average(results[:,1])
    avg_val_loss = np.average(results[:,2])
    avg_train_loss = np.average(results[:,3])
    print("AVERAGE TRAINING R^2: " + str(np.average(results[:,0])))
    print("AVERAGE TEST R^2: " + str(np.average(results[:,1])))
    print("AVERAGE MIN VALIDATION LOSS: " + str(np.average(results[:,2])))
    print("AVERAGE MIN TRAINING LOSS: " + str(np.average(results[:,3])))

    return [history, model, avg_training_r, avg_test_r, avg_val_loss, avg_train_loss]




def plot_stuff(test_images, train_images, history, trainr2, testr2, test_predictions, augmentation_list, label_to_predict):
    test_labels = test_images.labels
    train_labels = train_images.labels
    date_and_time = (str(datetime.datetime.now())[:10]).replace("-", "_")

    fig = plt.figure()
    plt.plot(history.history['loss'], label='loss (mean absolute error)')
    plt.plot(history.history['val_loss'], label='val_loss')
    #plt.ylim([0, 4])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title("loss over epochs predicting " + label_to_predict)
    plt.legend()
    plt.grid(True)
    fig.text(.5, .007, "Best loss = " + str(round(min(history.history['loss'])*10000)/10000) + " Best val_loss = " + str(round(min(history.history['val_loss'])*10000)/10000) + " train r^2 = " + str(round(trainr2*10000)/10000) + " test r^2 = " + str(round(testr2*10000)/10000),fontsize = 7, ha='center')
    fig_name = "C:\\Users\\u1056\\sfx\\Result_plots_after_fixing_augmentation\\" + label_to_predict + "_loss_vs_epochs_" + date_and_time + ".png"
    plt.savefig(fig_name)
    #plt.show()

    fig = plt.figure()
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.title("predictions vs true labels predicting " + label_to_predict)
    lims = [0, max(test_labels)]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims) 
    fig.text(.5, .007, "Best loss = " + str(round(min(history.history['loss'])*10000)/10000) + " Best val_loss = " + str(round(min(history.history['val_loss'])*10000)/10000) + " train r^2 = " + str(round(trainr2*10000)/10000) + " test r^2 = " + str(round(testr2*10000)/10000), fontsize = 8, ha='center')
    fig_name = "C:\\Users\\u1056\\sfx\\Result_plots_after_fixing_augmentation\\" + label_to_predict + "_predictions_vs_true_" + date_and_time + ".png"
    plt.savefig(fig_name)
    #plt.show()




# parent_folder_name = "Highlighted_only_Parietal"
""" limited augmentations below"""
# augmentation_list = ["OG", "Posterize", "Color", "Flipping", "Rotation", "Solarize"]
""" all augmentations below """
# augmentation_list = ["OG", "autoContrast", "Brightness Manipulation", "Color", "Contrast", "Equalize", "Flipping", "Gaussian Noise", "Identity", "Posterize", "Rotation", "Sharpness", "Shearing", "Shifting", "Solarize", "Zooming"]


parent_folder_name = "Original"
#parent_folder_name = "Highlighted_only_Parietal"
label_to_predict = "x"
augmentation_list = ["OG", "Posterize", "Color", "Flipping", "Rotation", "Solarize"]
args = prepare_data(parent_folder_name, augmentation_list)
height_result = make_CNN(args, label_to_predict, batch_size=5, patience=20, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, storing_Dir="/Users/jakehirst/Desktop/sfx/regression_tuner", Name=label_to_predict) 
#print(result)

label_to_predict = "y"
augmentation_list = ["OG", "Posterize", "Color", "Flipping", "Rotation", "Solarize"]
args = prepare_data(parent_folder_name, augmentation_list)
phi_result = make_CNN(args, label_to_predict, batch_size=5, patience=20, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, storing_Dir="/Users/jakehirst/Desktop/sfx/regression_tuner", Name=label_to_predict) 
#print(result)

label_to_predict = "theta"
augmentation_list = ["OG", "Posterize", "Color", "Flipping", "Rotation", "Solarize"]
args = prepare_data(parent_folder_name, augmentation_list)
theta_result = make_CNN(args, label_to_predict, batch_size=5, patience=20, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, storing_Dir="/Users/jakehirst/Desktop/sfx/regression_tuner", Name=label_to_predict) 

#print(result)
print(height_result)
print(phi_result)
print(theta_result)
print("done")
# label_to_predict = "phi"
# parent_folder_name = "With_Width_only_parietal"
# augmentations = ["OG", "Posterize", "Color", "Flipping", "Rotation", "Solarize"]

# results = {}
# for i in range(1, len(augmentations)):
#     augmentation_list = [augmentations[0], augmentations[i]]
#     args = prepare_data(parent_folder_name, augmentation_list)
#     result = make_CNN(args, label_to_predict, batch_size=5, patience=20, max_epochs=100, optimizer="Nadam", activation="relu", kernel_size=(10,10))
#     results[augmentations[i]] = [min(result[0].history['val_loss']), min(result[0].history['loss']), result[2], result[3]]

# print(results)
# print("done")





