import tensorflow as tf
from tensorflow import keras
import pandas as pd
from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array



num_folds = 3
num_epochs = 10
kernel_size = (9, 9)
pool_size = (3, 3)
strides = 2
filter1 = 16
filter2 = filter1*2
filter3 = filter2*2
dropout_rate = 0.5
image_name_list = []
height_list = []
phi_list = []
theta_list = []
img_arr_list = []
x_train = []
y_train = []
x_test = []
y_test = []
x_val = []
y_val = []
loss_per_fold = []
R2_per_fold = []
MSE_per_fold = []
image_path_list = []

#Gathers image list
def get_images(pathname):
    for root, dirs, files in os.walk(pathname):
        # select file name
            for file in files:
                # check the extension of files
                if file.endswith('.png'):
                    if file.find("mesh")==-1:
                        image_name_list.append (os.path.join(root, file))

#Select which folders from which to obtain images
get_images('C:\\Users\\u1056\\sfx\\images_sfx\\\Original\\OG')
get_images('C:\\Users\\u1056\\sfx\\images_sfx\\\Original\\Flipping')
get_images('C:\\Users\\u1056\\sfx\\images_sfx\\\Original\\Rotation')
get_images('C:\\Users\\u1056\\sfx\\images_sfx\\\Original\\Color')
get_images('C:\\Users\\u1056\\sfx\\images_sfx\\\Original\\Solarize')
get_images('C:\\Users\\u1056\\sfx\\images_sfx\\\Original\\Posterize')

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
            #image_name = image_path.split('\\')[-1]
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



#TODO: do this for phi and theta later
#shuffles the dataset and puts it into a dataframe
# df = pd.concat([image_path_list, height_list], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
# df.columns = ["Filepath", "height"]

df = pd.concat([image_path_list, phi_list], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
df.columns = ["Filepath", "phi"]

# df = pd.concat([image_path_list, theta_list], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
# df.columns = ["Filepath", "theta"]



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
    y_col = 'phi',
    #target_size=(120,120),  #can reduce the images to a certain size to reduce training time. 120x120 for example here
    color_mode='grayscale',
    class_mode = 'raw', #keeps the classes of our labels the same after flowing
    batch_size=1, #can increase this to up to like 10 or so for how much data we have
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col = 'Filepath',
    y_col = 'phi',
    #target_size=(120,120),  #can reduce the images to a certain size to reduce training time. 120x120 for example here
    color_mode='grayscale',
    class_mode = 'raw', #keeps the classes of our labels the same after flowing
    batch_size=1, #can increase this to up to like 10 or so for how much data we have
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col = 'Filepath',
    y_col = 'phi',
    #target_size=(120,120),  #can reduce the images to a certain size to reduce training time. 120x120 for example here
    color_mode='grayscale',
    class_mode = 'raw', #keeps the classes of our labels the same after flowing
    batch_size=1, #can increase this to up to like 10 or so for how much data we have
    shuffle=False
)



""" start training """
inputs = tf.keras.Input(shape=(802, 642, 1)) #not sure if the shape is right here. i got (256, 256, 1) by looking at train_images.image_shape
                                             #trying (642, 802, 3) by looking at img_arr_list[0].shape
                                             #tryig  (256, 256, 3) by looking at train_images[0][0].shape
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D()(x) #takes the max of each window to reduce the size of the image... dont know if i need this...
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')(inputs) #just increasing the filters here to get more features
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(inputs) #just increasing the filters here to get more features
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(inputs) #just increasing the filters here to get more features
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x) #could try GlobalMaxPooling2D instead
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='linear')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='mae'
)

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=1000, #max number of epochs to go over data
    #this callback makes the learning stop if the validation loss stops improving for 'patience' epochs in a row. very useful tool should use in other models
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True
        )
    ]
)

test_labels = test_images.labels

plt.plot(history.history['loss'], label='loss (mean absolute error)')
plt.plot(history.history['val_loss'], label='val_loss')
#plt.ylim([0, 4])
plt.xlabel('Epoch')
plt.ylabel('Error [deg]')
plt.legend()
plt.grid(True)
plt.show()


""" makes predictions with the test dataset and plots them. Good predictions should lie on the line. """
test_predictions = np.squeeze(model.predict(test_images))

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True labels')
plt.ylabel('Predicted labels')
lims = [0, max(test_labels)]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()


print("done")




