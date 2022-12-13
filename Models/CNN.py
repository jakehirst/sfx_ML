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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array



# num_folds = 3
# num_epochs = 10
# kernel_size = (9, 9)
# pool_size = (3, 3)
# strides = 2
# filter1 = 16
# filter2 = filter1*2
# filter3 = filter2*2
# dropout_rate = 0.5
# image_name_list = []
# height_list = []
# phi_list = []
# theta_list = []
# img_arr_list = []
# x_train = []
# y_train = []
# x_test = []
# y_test = []
# x_val = []
# y_val = []
# loss_per_fold = []
# R2_per_fold = []
# MSE_per_fold = []
# image_path_list = []

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
    for folder in augmentation_list:
        image_name_list = get_images('C:\\Users\\u1056\\sfx\\images_sfx\\' + parent_folder_name + "\\" + folder, image_name_list)

    #finds the max uci and step for each fall parameter image folder
    max_steps_and_UCIs = dict()
    for image_path in image_name_list:
        #if(image_path.endswith("Dynamic.png") or image_path.split("_")[-2] == "Dynamic"):
        if(image_path.endswith("Dynamic.png")):
            image_name = image_path.split("\\")[-1]
            folder_name = image_path.split("\\")[-2]
            UCI = int(image_name.split("_")[2])
            step = int(image_name.split("_")[0].split("p")[1])
            if(not (folder_name in max_steps_and_UCIs.keys())):
                max_steps_and_UCIs[folder_name] = [0, 0]
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
            image_name = image_path.split("\\")[-1]
            folder_name = image_path.split("\\")[-2]
            UCI = int(image_name.split("_")[2])
            step = int(image_name.split("_")[0].split("p")[1])
            #Only selects the images from the max step/uci combinations
            if(step == max_steps_and_UCIs[folder_name][0] and UCI == max_steps_and_UCIs[folder_name][1]):
                image_path_list.append(str(image_path))
                img = load_img(image_path)#,color_mode = "grayscale")
                img_arr = img_to_array(img)
                input_shape = img_arr.shape
                img_arr_list.append(img_arr)
                image_name = image_path.split('\\')[-1]
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


def make_CNN(args, label_to_predict, batch_size=5, patience=25, max_epochs=1000, optimizer="adam", activation="relu", kernel_size=(5,5), plot = True):
    #TODO: do this for phi and theta later
    #shuffles the dataset and puts it into a dataframe
    if(label_to_predict == "height"):
        df = pd.concat([args[0], args[1]], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
        df.columns = ["Filepath", "height"]
    elif(label_to_predict == "phi"):
        df = pd.concat([args[0], args[2]], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
        df.columns = ["Filepath", "phi"]
    elif(label_to_predict == "theta"):
        df = pd.concat([args[0], args[3]], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
        df.columns = ["Filepath", "theta"]




    train_df, test_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=1)

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255, #all pixel values set between 0 and 1 instead of 0 and 255.
        validation_split=0.2 #creating a validation split in our training generator
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255, #all pixel values set between 0 and 1 instead of 0 and 255.
    )

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
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col = 'Filepath',
        y_col = label_to_predict,
        target_size=args[4][:2], #can reduce the images to a certain size to reduce training time. 120x120 for example here
        color_mode='rgb',
        class_mode = 'raw', #keeps the classes of our labels the same after flowing
        batch_size=batch_size, #can increase this to up to like 10 or so for how much data we have
        shuffle=True,
        seed=42,
        subset='validation'
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

    #TODO check out examples in this stack overflow article:
    #https://stackoverflow.com/questions/45528285/cnn-image-recognition-with-regression-output-on-tensorflow

    # """ start training """
    # inputs = tf.keras.Input(shape=args[4]) #not sure if the shape is right here. i got (256, 256, 1) by looking at train_images.image_shape
    #                                             #trying (642, 802, 3) by looking at img_arr_list[0].shape
    #                                             #tryig  (256, 256, 3) by looking at train_images[0][0].shape
    # #TODO: layers study
    # x = tf.keras.layers.Conv2D(filters=16, kernel_size=kernel_size, activation='relu')(inputs)
    # x = tf.keras.layers.Conv2D(filters=16, kernel_size=kernel_size, activation='relu')(inputs)
    # x = tf.keras.layers.MaxPool2D()(x) #takes the max of each window to reduce the size of the image... dont know if i need this...
    # x = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, activation='relu')(inputs)
    # x = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size, activation='relu')(inputs)
    # x = tf.keras.layers.MaxPool2D()(x)
    # x = tf.keras.layers.Conv2D(filters=64, kernel_size=kernel_size, activation='relu')(inputs)
    # x = tf.keras.layers.MaxPool2D()(x)
    # x = tf.keras.layers.Conv2D(filters=128, kernel_size=kernel_size, activation='relu')(inputs)
    # x = tf.keras.layers.MaxPool2D()(x)
    # x = tf.keras.layers.Conv2D(filters=256, kernel_size=kernel_size, activation='relu')(inputs)
    # x = tf.keras.layers.MaxPool2D()(x)
    # x = tf.keras.layers.Conv2D(filters=512, kernel_size=kernel_size, activation='relu')(inputs)
    # x = tf.keras.layers.MaxPool2D()(x)

    # x = tf.keras.layers.GlobalAveragePooling2D()(x) #could try GlobalMaxPooling2D instead
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(64, activation='relu')(x) #TODO: figure out why these are here
    # #x = tf.keras.layers.Dense(64, activation='relu')(x) #TODO: figure out why these are here
    # outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    # model = tf.keras.Model(inputs=inputs, outputs=outputs)

    #https://datascience.stackexchange.com/questions/106600/how-to-perform-regression-on-image-data-using-tensorflow
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(3, 3, activation='relu'),
    tf.keras.layers.Conv2D(3, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(3, 3, activation='relu'),
    tf.keras.layers.Conv2D(3, 3, activation='relu'),
    tf.keras.layers.Dropout(0.1), #TODO: study dropout rates and frequency of dropouts. Dropouts are supposed to reduce overfitting. generally increase dropout rate in the later layers, and reduce for initial layers
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(3, 3, activation='relu'),
    tf.keras.layers.Conv2D(3, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1024, activation='relu'),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
    ])
    #https://datascience.stackexchange.com/questions/106600/how-to-perform-regression-on-image-data-using-tensorflow

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='mae' #could change back to mae
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
        ],
        verbose=1
    )


    test_labels = test_images.labels
    train_labels = train_images.labels

    # """ makes predictions with the test dataset and plots them. Good predictions should lie on the line. """
    test_predictions = np.squeeze(model.predict(test_images))
    if(plot == True):
        plot_stuff(test_images, train_images, history, test_predictions)

    """ gets r^2 value of the test dataset with the predictions made from above ^ """
    metric = tfa.metrics.r_square.RSquare()
    metric.update_state(test_labels, test_predictions)
    training_result = metric.result()
    print("Test R^2 = " + str(training_result.numpy()))

    """ gets r^2 value of the training dataset """
    training_predictions = model.predict(train_images).flatten()
    metric = tfa.metrics.r_square.RSquare()
    metric.update_state(train_labels, training_predictions)
    test_result = metric.result()
    print("Training R^2 = " + str(test_result.numpy()))
    #return {"Training R^2": training_result.numpy(), "Test R^2": test_result.numpy(), "history": history}
    print("min val loss = " + str(min(history.history['val_loss'])))
    print("min loss = " + str(min(history.history['loss'])))
    return [history, model, training_result.numpy(), test_result.numpy()]




def plot_stuff(test_images, train_images, history, test_predictions):
    test_labels = test_images.labels
    train_labels = train_images.labels

    plt.plot(history.history['loss'], label='loss (mean absolute error)')
    plt.plot(history.history['val_loss'], label='val_loss')
    #plt.ylim([0, 4])
    plt.xlabel('Epoch')
    plt.ylabel('Error [deg]')
    plt.legend()
    plt.grid(True)
    plt.show()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    lims = [0, max(test_labels)]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()



label_to_predict = "theta"
parent_folder_name = "With_Width"
# parent_folder_name = "Highlighted_only_Parietal"
augmentation_list = ["OG", "Posterize", "Color", "Flipping", "Rotation", "Solarize"]
args = prepare_data(parent_folder_name, augmentation_list)
result = make_CNN(args, label_to_predict, batch_size=5, patience=50, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3)) #smaller kernel size leads to better results usually.
print(result)

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





