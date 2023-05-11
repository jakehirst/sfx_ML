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
from Binning_phi_and_theta import *
from PIL import Image
from keras_tuner import BayesianOptimization
import imageio
from time import time

"""
to show tensorboard do the following:
- open anaconda prompt
- cd to directory with the log files (example C:\\Users\\u1056\\sfx\\bin_tuner_models)
- activate tensorflow env 
    - conda activate tensorflow_env
- type the following with "tb_logs" being the log directory you want to show
    - tensorboard --logdir=tb_logs
- it will say something like this:    TensorBoard 2.9.0 at http://localhost:6006/ (Press CTRL+C to quit)
    - copy http://localhost:6006/ and paste into your browser
"""




#Gathers image list
def get_images(pathname, image_name_list):
    print(pathname)
    for root, dirs, files in os.walk(pathname):
        # select file name
            for file in files:
                # check the extension of files
                if file.endswith('.png'):
                    if file.find("mesh")==-1:
                        image_name_list.append (os.path.join(root, file))
    return image_name_list

def prepare_data(parent_folder_name, augmentation_list):
    image_name_list = []
    #in lab
    #image_name_list = get_images('C:\\Users\\u1056\\sfx\\images_sfx\\' + parent_folder_name + "\\" + "OG", image_name_list)
    
    #at home
    image_name_list = get_images('/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx/' + parent_folder_name + "/" + "OG", image_name_list)    # for folder in augmentation_list:
    #     image_name_list = get_images('C:\\Users\\u1056\\sfx\\images_sfx\\' + parent_folder_name + "\\" + folder, image_name_list)

    #finds the max uci and step for each fall parameter image folder
    max_steps_and_UCIs = dict()
    for image_path in image_name_list:
        #if(image_path.endswith("Dynamic.png") or image_path.split("_")[-2] == "Dynamic"):
        if(image_path.endswith("Dynamic.png")):
            # if(image_path.__contains__("Para_2ft_PHI_30_THETA_135")):
            #     print(image_path)
            
            #in lab
            # image_name = image_path.split("\\")[-1]
            # folder_name = image_path.split("\\")[-2]
            
            #at home
            image_name = image_path.split("/")[-1]
            folder_name = image_path.split("/")[-2]    
            
            UCI = int(image_name.split("_")[2])
            step = int(image_name.split("_")[0].split("p")[1])
            if(not (folder_name in max_steps_and_UCIs.keys())):
                max_steps_and_UCIs[folder_name] = [step, UCI]
            else:
                if(step > max_steps_and_UCIs[folder_name][0]):
                    max_steps_and_UCIs[folder_name] = [step, UCI]
                elif(step == max_steps_and_UCIs[folder_name][0] and UCI > max_steps_and_UCIs[folder_name][1]):
                    max_steps_and_UCIs[folder_name] = [step, UCI]
        else:
            continue

    img_arr_list = []
    image_path_list = []
    height_list = []
    phi_list = []
    theta_list = []
    #Extracts parameters from image names
    for image_path in image_name_list:
        if(image_path.endswith("Dynamic.png") or image_path.split("_")[-2] == "Dynamic"):
            #in lab
            # image_name = image_path.split("\\")[-1]
            # folder_name = image_path.split("\\")[-2]
            
            #at home
            image_name = image_path.split("/")[-1]
            folder_name = image_path.split("/")[-2]
            
            UCI = int(image_name.split("_")[2])
            step = int(image_name.split("_")[0].split("p")[1])
            #Only selects the images from the max step/uci combinations
            if(step == max_steps_and_UCIs[folder_name][0] and UCI == max_steps_and_UCIs[folder_name][1]):
                image_path_list.append(str(image_path))
                img = load_img(image_path)#,color_mode = "grayscale")
                img_arr = img_to_array(img)
                input_shape = img_arr.shape
                img_arr_list.append(img_arr)

                #in lab
                #image_name = image_path.split('\\')[-1]
                
                #at home
                image_name = image_path.split('/')[-1]

                height = folder_name.split('ft_')[0]
                height = height.split('Para_')[1]
                height = height.replace('-', '.')
                height_list = np.append(height_list,float(height)) 
                phi = folder_name.split('_THETA')[0]
                phi = phi.split('PHI_')[1]
                phi_list = np.append(phi_list,float(phi)) 
                if image_name.find("LOCAL")==-1:
                    theta = folder_name.split('_Stp')[0]
                else:
                    theta = folder_name.split('_LOCAL')[0]
                theta = theta.split('THETA_')[1].replace(".png", "")
                theta = theta.split("_")[0]
                theta_list = np.append(theta_list,float(theta))
        else:
            continue

    image_path_list = pd.Series(image_path_list).astype(str)
    height_list = pd.Series(height_list)
    phi_list = pd.Series(phi_list)
    theta_list = pd.Series(theta_list)

    """ img_arr_list contains all of the images for training (final steps and ucis) in matrix form"""
    print("number of examples: " + str(len(img_arr_list)))
    print("number of heights = " + str(len(height_list)))
    print("number of phis = " + str(len(phi_list)))
    print("number of thetas = " + str(len(theta_list)))
    return [image_path_list, height_list, phi_list, theta_list, input_shape, img_arr_list]



def add_augmentations(df, augmentation_list):
    image_name_list = []
    new_df = df
    if(augmentation_list.__contains__("OG")):
        augmentation_list.remove("OG")
    for folder in augmentation_list:
        for row in df.iterrows():
            #in lab
            # OG_picture_step_UCI = row[1]["Filepath"].split("\\")[-1].split(".")[0]
            # parameters = row[1]["Filepath"].split("\\")[-2]
            
            #at home
            OG_picture_step_UCI = row[1]["Filepath"].split("/")[-1].split(".")[0]
            parameters = row[1]["Filepath"].split("/")[-2]
            
            #print("\nrow filepath = " + str(row[1]["Filepath"]))
            #print(row[1].to_frame().transpose())

            #in lab
            #pathname = 'C:\\Users\\u1056\\sfx\\images_sfx\\' + parent_folder_name + "\\" + folder
            
            #at home
            pathname = '/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx' + parent_folder_name + '/' + folder

            for root, dirs, files in os.walk(pathname):
            # select file name
                for file in files:
                    # check the extension of files
                    if file.endswith('.png'):
                        if file.find("mesh")==-1:
                            if((OG_picture_step_UCI in file) and (parameters in root)):
                                path = os.path.join(root, file)
                                new_row = row[1].to_frame().transpose()
                                new_row["Filepath"][0] = path
                                #new_row = pd.DataFrame([{'Filepath':str(path), df.columns[1]:label}])
                                new_df = pd.concat([new_df, new_row], ignore_index=True) 
                                #new_df.append({'Filepath':str(path), df.columns[1]:label}, ignore_index=True) #DONT use pd.concat here... it changes the type of things in the dataframe
    for i in range(len(new_df.columns) - 1):
        new_df[[str(i)]] = new_df[[str(i)]].apply(pd.to_numeric)
    return new_df


def remove_augmentations(images):
    indexes_to_delete = []
    for i in range(images.n):
        if(not images.filenames[i].endswith("Dynamic.png")):
            indexes_to_delete.append(i)
    
    images.filenames = np.delete(images.filenames, indexes_to_delete)
    images.filepaths = (np.delete(np.array(images.filepaths), indexes_to_delete)).tolist()
    images.labels = np.delete(images.labels, indexes_to_delete)
    images._filepaths = (np.delete(np.array(images._filepaths), indexes_to_delete)).tolist()
    images._targets = np.delete(images._targets, indexes_to_delete)


    return images



""" Builds model for hyperparameter tuning """
#https://www.analyticsvidhya.com/blog/2021/06/create-convolutional-neural-network-model-and-optimize-using-keras-tuner-deep-learning/
def build_model_1_Con_layer(hp):
    # create model object
    model = keras.Sequential([
    #adding first convolutional layer    
    keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_1_filter', min_value=16, max_value=128, step=16),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_1_kernel', values = [3]),
        #activation function
        activation='relu'),
        #input_shape=(642, 802, 3)),
    keras.layers.MaxPooling2D(2),
    # adding flatten layer
    keras.layers.Flatten(),
    # adding dense layer    
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=16, max_value=128, step=16),
        activation='relu'
    ),
    # output layer    
    keras.layers.Dense(units=NUM_BINS, activation = 'softmax') #TODO change units to be equal to the number of total bins
    ])
    #compilation of model
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-3])),
              loss='categorical_crossentropy',
              metrics=['acc'])
    return model


def build_model_2_Con_layer(hp):
    # create model object
    model = keras.Sequential([
    #adding first convolutional layer    
    keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_1_filter', min_value=8, max_value=128, step=8),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_1_kernel', values = [3]),
        #activation function
        activation='relu',
        input_shape=(642, 802, 3)),
    keras.layers.MaxPooling2D(2),
    
    keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_2_filter', min_value=8, max_value=128, step=8),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_2_kernel', values = [3]),
        #activation function
        activation='relu',
        input_shape=(642, 802, 3)),
    keras.layers.MaxPooling2D(2),
    # adding flatten layer
    keras.layers.Flatten(),
    # adding dense layer    
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=16, max_value=256, step=16),
        activation='relu'
    ),
    # output layer    
    keras.layers.Dense(units=NUM_BINS, activation = 'softmax')
    ])
    #compilation of model
    # model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5])),
    #           loss='categorical_crossentropy',
    #           metrics=['acc'])
    
    model.compile(optimizer=keras.optimizers.Adam(),
                loss = 'categorical_crossentropy',
                metrics=['acc'])
    return model



def build_model_2_Con_2_Dense_layer(hp):
    # create model object
    model = keras.Sequential([
    #adding first convolutional layer    
    keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_1_filter', min_value=8, max_value=64, step=8),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_1_kernel', values = [3]),
        #activation function
        activation='relu',
        input_shape=(642, 802, 3)),
    keras.layers.MaxPooling2D(2),
    
    keras.layers.Conv2D(
        #adding filter 
        filters=hp.Int('conv_2_filter', min_value=8, max_value=64, step=8),
        # adding filter size or kernel size
        kernel_size=hp.Choice('conv_2_kernel', values = [3]),
        #activation function
        activation='relu',
        input_shape=(642, 802, 3)),
    keras.layers.MaxPooling2D(2),
    # adding flatten layer
    keras.layers.Flatten(),
    # adding dense layer    
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=8, max_value=256, step=8),
        activation='relu'
    ),

    keras.layers.Dense(
        units=hp.Int('dense_2_units', min_value=8, max_value=256, step=8),
        activation='relu'
    ),
    # output layer    
    keras.layers.Dense(units=NUM_BINS, activation = 'softmax')
    ])
    #compilation of model
    # model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5])),
    #           loss='categorical_crossentropy',
    #           metrics=['acc'])
    
    model.compile(optimizer=keras.optimizers.Adam(),
                loss = 'categorical_crossentropy',
                metrics=['acc'])
    return model






def make_CNN(args, batch_size=5, patience=25, max_epochs=10, plot = True, augmentation_list = [], num_folds=5, num_phi_bins=5, num_theta_bins=5, bin_type="solid center phi and theta", Name="_", storing_Dir="Desktop"):
    results = []
    #shuffles the dataset and puts it into a dataframe
    phiandtheta_df = pd.concat([args[0], args[2], args[3]], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
    phiandtheta_df.columns = ["Filepath", "phi", "theta"]
    if(bin_type == "solid center phi and theta"):
        df, y_col_values, bins_and_values = Bin_phi_and_theta_center_target(phiandtheta_df, num_phi_bins, num_theta_bins)
    elif(bin_type == "phi and theta"):
        df, y_col_values, bins_and_values = Bin_phi_and_theta(phiandtheta_df, num_phi_bins, num_theta_bins)
    elif(bin_type == "theta"):
        df, y_col_values, bins_and_values = Bin_just_theta(phiandtheta_df, num_theta_bins)
    elif(bin_type == "phi"):
        df, y_col_values, bins_and_values = Bin_just_phi(phiandtheta_df, num_phi_bins)   
    
    print(f"\n########## CHANGE NUM_BINS TO {len(y_col_values)} ##########")
    kfold = KFold(n_splits=num_folds, shuffle=True)
    inputs = np.array(df["Filepath"])
    outputs = np.array(df[y_col_values])



    fold_no = 1
    for train, test in kfold.split(inputs, outputs):
        train_df = df.iloc[train]
        test_df = df.iloc[test]



        #train_df, test_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=1)

        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255, #all pixel values set between 0 and 1 instead of 0 and 255.
            validation_split=0.2 #creating a validation split in our training generator
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
            y_col = y_col_values,
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
            y_col = y_col_values,
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
            y_col = y_col_values,
            target_size=args[4][:2],  #can reduce the images to a certain size to reduce training time. 120x120 for example here
            color_mode='rgb',
            class_mode = 'raw', #keeps the classes of our labels the same after flowing
            batch_size=batch_size, #can increase this to up to like 10 or so for how much data we have
            shuffle=False,
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

        """ Random search hyperparameter tuner """
        # tuner = RandomSearch(build_model,
        #             objective='val_mae',
        #             max_trials = 20)

        """ Baysian Optimization hyperparameter tuner """
        tuner = BayesianOptimization(
            build_model_1_Con_layer,
            #build_model_2_Con_layer,
            #build_model_2_Con_2_Dense_layer,
            objective='val_loss',
            max_trials=50,
            overwrite = True, #need this otherwise it will compare to previous trials done in the past
            directory = storing_Dir+ Name + "_tb"
            # seed=42,
            # executions_per_trial=2
        )

        tuner.search(
            train_inputs,
            train_images._targets, 
            epochs=max_epochs, 
            validation_data=(val_inputs, val_images._targets), 
            callbacks=[keras.callbacks.TensorBoard(storing_Dir+ Name + "_tb_logs")]
         )

        best_models = tuner.get_best_models(num_models=5)
        model= best_models[0]
        #summary of best model
        model.summary()

        #https://datascience.stackexchange.com/questions/106600/how-to-perform-regression-on-image-data-using-tensorflow

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), 
            loss= 'categorical_crossentropy', #TODO idk about this loss #https://stackoverflow.com/questions/47034888/how-to-choose-cross-entropy-loss-in-tensorflow
            #loss = tf.keras.losses.SparseCategoricalCrossentropy,
            #https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8
            metrics = ['acc']
        )


        history = model.fit(
            train_images,
            validation_data=val_images,
            epochs=max_epochs, #max number of epochs to go over data
            #this callback makes the learning stop if the validation loss stops improving for 'patience' epochs in a row. very useful tool should use in other models
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True
                )
                #,tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 30))
            ],
            verbose=1 #prints training process
        )


        test_labels = test_images.labels
        train_labels = train_images.labels

        # """ makes predictions with the test dataset and plots them. Good predictions should lie on the line. """
        test_predictions = np.squeeze(model.predict(test_images))

        """prediction sheet should tell us alot about the location of the prediction vs the actual location to see if it is relatively close"""
        prediction_sheet = [] 



        fold_no += 1
    print(prediction_sheet)
    print("done")



    return model
    # return [history, model, avg_training_r, avg_test_r, avg_train_loss, avg_val_loss]





# parent_folder_name = "Highlighted_only_Parietal"
""" limited augmentations below"""
# augmentation_list = ["OG", "Posterize", "Color", "Flipping", "Rotation", "Solarize"]
""" all augmentations below """
# augmentation_list = ["OG", "autoContrast", "Brightness Manipulation", "Color", "Contrast", "Equalize", "Flipping", "Gaussian Noise", "Identity", "Posterize", "Rotation", "Sharpness", "Shearing", "Shifting", "Solarize", "Zooming"]

NUM_BINS = 2 #COMMENT must change this to the total number of bins you are optimizing for every time you want to run this code

parent_folder_name = "Original"
label_to_predict = "binned_orientation"
augmentation_list = ["OG", "Posterize", "Color", "Flipping", "Rotation", "Solarize"]
Name = "1_Conv_1_Dense_2_total_bins_just_theta_{}".format(int(time()))
storing_Dir = "F:\\Jake\\model_tuning\\bin_tuner_models\\"#for lab
storing_Dir = "/Users/jakehirst/Desktop/sfx/hyperparameter_tuning/"#for at home

args = prepare_data(parent_folder_name, augmentation_list)
make_CNN(args, batch_size=5, patience=10, max_epochs=15, augmentation_list=augmentation_list, plot=True, num_phi_bins=2, num_theta_bins=2, bin_type="theta", Name=Name, storing_Dir=storing_Dir)
#make_CNN(args, label_to_predict, batch_size=5, patience=3, max_epochs=20, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, num_bins=6)
