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
import os




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
    image_name_list = get_images('/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx/' + parent_folder_name + "/" + "OG", image_name_list)
    
    # for folder in augmentation_list:
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
                                new_row.iloc[0]["Filepath"] = path
                                new_df = pd.concat([new_df, new_row], ignore_index=True) 
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




def make_CNN(args, label_to_predict, batch_size=5, patience=25, max_epochs=1000, optimizer="adam", activation="relu", kernel_size=(5,5), plot = True, augmentation_list = [], num_folds=5, num_phi_bins=5, num_theta_bins=5, bin_type="solid center phi and theta"):
    results = []
    #TODO: do this for phi and theta later
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


    print("\nTOTAL NUMBER OF BINS = " + str(len(y_col_values)))

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

        #https://datascience.stackexchange.com/questions/106600/how-to-perform-regression-on-image-data-using-tensorflow

        model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2),
        # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(units=1024, activation='relu'),
        # tf.keras.layers.Dense(units=512, activation='relu'),
        # tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units= len(y_col_values), activation='softmax') #one node per class label for a softmax activation function
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(), 
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

        print("min val loss = " + str(min(history.history['val_loss'])))

        #making nice plots to look at :)
        label_to_predict = "Orientation via bins"
        
        #in lab
        #folder = f"C:\\Users\\u1056\\sfx\\captain_america_plots\\center_circle_phi_{num_phi_bins}_theta_{num_theta_bins}_bins\\fold{fold_no}\\"
        
        #at home
        folder = f"/Users/jakehirst/Desktop/sfx/captain_america_plots/center_circle_phi_{num_phi_bins}_theta_{num_theta_bins}_bins/fold{fold_no}/"
        
        folder = f"/Users/jakehirst/Desktop/sfx/captain_america_plots/JustTheta_{num_theta_bins}_bins/fold{fold_no}/"

        for i in range(len(test_predictions)):
            make_sphere(bins_and_values, test_predictions[i], test_images._filepaths[i], folder)
        plot_stuff(history, label_to_predict, folder) #this has to be after make_sphere because make_sphere makes the folder duh

        # """prediction sheet should tell us alot about the location of the prediction vs the actual location to see if it is relatively close"""
        # prediction_sheet = [] 


        # hit_arr = []
        # bins_checked_arr = []
        # misses_arr = np.zeros(len(y_col_values))
        # for bins_checked in range(1,4):
        #     hits = 0
        #     for test in range(len(test_predictions)):
        #         tuples = []
        #         for i in range(len(test_predictions[test])):
        #             tuples.append((test_predictions[test][i], i))
        #         tuples.sort(key=lambda t: -t[0]) #sorted array of tuples by prediction value ==> [(prediction val of bin, bin number)]
        #         real_location = np.where(test_labels[test] == 1)[0][0]
                
        #         #in lab
        #         #prediction_sheet.append([fold_no, tuples[0], tuples[1], tuples[2], real_location, test_images.filenames[0].split("\\")[-2]])
                
        #         #at home
        #         prediction_sheet.append([fold_no, tuples[0], tuples[1], tuples[2], real_location, test_images.filenames[0].split("/")[-2]])

                
        #         for i in range(bins_checked):
        #             if(bins_checked == 5 and i == bins_checked - 1 and tuples[i][1] != real_location):
        #                 misses_arr[real_location] = misses_arr[real_location] + 1
        #             elif(tuples[i][1] == real_location):
        #                 hits += 1
        #                 continue
        #     hit_arr.append(hits)
        #     bins_checked_arr.append(bins_checked)
            
        #     print("There were " + str(hits) + " hits in the test set with the top " + str(bins_checked) + " bins checked.")
        #     print("There were " + str(len(test_predictions) - hits) + " misses in the test set with the top " + str(bins_checked) + " bins checked.")


        #plot_3_Bin_misses(misses_arr, fold_no, num_bins)
        #check_bins_plot(hit_arr, len(test_predictions), bins_checked_arr, fold_no, num_bins)
            

        fold_no += 1
    # print(prediction_sheet)
    print("done")



    #     """ gets r^2 value of the test dataset with the predictions made from above ^ """
    #     metric = tfa.metrics.r_square.RSquare()
    #     metric.update_state(test_labels, test_predictions)
    #     training_result = metric.result()
    #     print("Train R^2 = " + str(training_result.numpy()))

    #     """ gets r^2 value of the training dataset """
    #     training_predictions = model.predict(train_images).flatten()
    #     metric = tfa.metrics.r_square.RSquare()
    #     metric.update_state(train_labels, training_predictions)
    #     test_result = metric.result()
    #     print("Test R^2 = " + str(test_result.numpy()))
    #     #return {"Training R^2": training_result.numpy(), "Test R^2": test_result.numpy(), "history": history}
    #     print("min val loss = " + str(min(history.history['val_loss'])))
    #     print("min loss = " + str(min(history.history['loss'])))

    #     if(plot == True):
    #         plot_stuff(test_images, train_images, history, training_result.numpy() , test_result.numpy(), test_predictions, augmentation_list, label_to_predict, fold_no)
    #     results.append([training_result.numpy(), test_result.numpy(), min(history.history['val_loss']), min(history.history['loss'])])
    #     fold_no += 1

    # results = np.array(results)
    # avg_training_r = np.average(results[:,0])
    # avg_test_r = np.average(results[:,1])
    # avg_val_loss = np.average(results[:,2])
    # avg_train_loss = np.average(results[:,3])
    # print("AVERAGE TRAINING R^2: " + str(np.average(results[:,0])))
    # print("AVERAGE TEST R^2: " + str(np.average(results[:,1])))
    # print("AVERAGE MIN VALIDATION LOSS: " + str(np.average(results[:,2])))
    # print("AVERAGE MIN TRAINING LOSS: " + str(np.average(results[:,3])))

    return model
    # return [history, model, avg_training_r, avg_test_r, avg_train_loss, avg_val_loss]

def plot_stuff(history, label_to_predict, folder):

    fig = plt.figure()
    plt.plot(history.history['loss'], label='loss (categorical cross entropy)')
    plt.plot(history.history['val_loss'], label='val_loss')
    #plt.ylim([0, 4])
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title("loss over epochs predicting " + label_to_predict)
    plt.legend()
    plt.grid(True)
    fig.text(.5, .007, "Best val_loss = " + str(round(min(history.history['val_loss'])*10000)/10000),fontsize = 7, ha='center')
    fig_name = folder + "loss_vs_epochs.png"
    plt.savefig(fig_name)
    plt.close()

    fig = plt.figure()
    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label='validation accuracy')
    #plt.ylim([0, 4])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title("Accuracy over epochs predicting " + label_to_predict)
    plt.legend()
    plt.grid(True)
    fig.text(.5, .007, "Best validation accuracy = " + str(round(max(history.history['val_acc'])*10000)/10000),fontsize = 7, ha='center')
    fig_name = folder + "accuracy_vs_epochs.png"
    plt.savefig(fig_name)
    plt.close()


def plot_3_Bin_misses(misses_arr, fold, num_bins):
    bins = []
    for i in range(len(misses_arr)):
        bins.append(i)
    
    plt.title("True location of missed test cases (checking 5 bins)")
    plt.xlabel("bin number")
    plt.ylabel("number of misses per bin")
    plt.bar(bins, misses_arr)
    plt.savefig("C:\\Users\\u1056\\sfx\\bin_plots_" + str(num_bins) + "_bins_3\\misses_plot_FOLD" + str(fold) +".png")
    plt.close()
    return



def check_bins_plot(hit_arr, num_tests, bins_checked_arr, fold, num_bins):
    misses_arr = []
    for i in range(len(hit_arr)):
        misses_arr.append(num_tests - hit_arr[i])
    
    x = np.arange(len(bins_checked_arr))
    width = 0.35

    fig, ax = plt.subplots()
    hit_bar = ax.bar(x - width/2, hit_arr, width, label="hits")
    miss_bar = ax.bar(x + width/2, misses_arr, width, label="misses")

    ax.set_xticks(x)
    ax.set_xticklabels(bins_checked_arr)
    ax.legend()
    ax.set_title("#hits vs #bins checked FOLD " + str(fold))
    ax.set_xlabel("#bins checked")
    ax.set_ylabel("#hits/misses")

    plt.savefig("C:\\Users\\u1056\\sfx\\bin_plots_" + str(num_bins) + "_bins_3\\Bins_plot_FOLD" + str(fold) +".png")  
    plt.close()

    return





# parent_folder_name = "Highlighted_only_Parietal"
""" limited augmentations below"""
# augmentation_list = ["OG", "Posterize", "Color", "Flipping", "Rotation", "Solarize"]
""" all augmentations below """
# augmentation_list = ["OG", "autoContrast", "Brightness Manipulation", "Color", "Contrast", "Equalize", "Flipping", "Gaussian Noise", "Identity", "Posterize", "Rotation", "Sharpness", "Shearing", "Shifting", "Solarize", "Zooming"]


parent_folder_name = "Original"
# parent_folder_name = "Original_from_test_matrix"
#parent_folder_name = "Highlighted_only_Parietal"

label_to_predict = "binned_orientation"
augmentation_list = ["OG", "Posterize", "Color", "Flipping", "Rotation", "Solarize"]
args = prepare_data(parent_folder_name, augmentation_list)
make_CNN(args, label_to_predict, batch_size=5, patience=10, max_epochs=100, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, num_phi_bins=2, num_theta_bins=2, bin_type="theta")
make_CNN(args, label_to_predict, batch_size=5, patience=10, max_epochs=100, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, num_phi_bins=2, num_theta_bins=3, bin_type="theta")
make_CNN(args, label_to_predict, batch_size=5, patience=10, max_epochs=100, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, num_phi_bins=2, num_theta_bins=4, bin_type="theta")
make_CNN(args, label_to_predict, batch_size=5, patience=10, max_epochs=100, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, num_phi_bins=3, num_theta_bins=5, bin_type="theta")
make_CNN(args, label_to_predict, batch_size=5, patience=10, max_epochs=100, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, num_phi_bins=3, num_theta_bins=6, bin_type="theta")
make_CNN(args, label_to_predict, batch_size=5, patience=10, max_epochs=100, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, num_phi_bins=3, num_theta_bins=7, bin_type="theta")
make_CNN(args, label_to_predict, batch_size=5, patience=10, max_epochs=100, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, num_phi_bins=4, num_theta_bins=8, bin_type="theta")
make_CNN(args, label_to_predict, batch_size=5, patience=10, max_epochs=100, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, num_phi_bins=4, num_theta_bins=9, bin_type="theta")
make_CNN(args, label_to_predict, batch_size=5, patience=10, max_epochs=100, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, num_phi_bins=4, num_theta_bins=10, bin_type="theta")
# make_CNN(args, label_to_predict, batch_size=5, patience=10, max_epochs=100, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, num_phi_bins=5, num_theta_bins=3, bin_type="theta")
# make_CNN(args, label_to_predict, batch_size=5, patience=10, max_epochs=100, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, num_phi_bins=5, num_theta_bins=4, bin_type="theta")
# make_CNN(args, label_to_predict, batch_size=5, patience=10, max_epochs=100, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, num_phi_bins=5, num_theta_bins=5, bin_type="theta")

print("done")
#make_CNN(args, label_to_predict, batch_size=5, patience=3, max_epochs=20, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, num_bins=6)
