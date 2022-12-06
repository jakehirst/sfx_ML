import tensorflow as tf
import numpy as np
from keras import layers, models, Sequential, losses, applications, Model, Input, optimizers, callbacks, backend, utils
from matplotlib import pyplot as plt
import os
import random
from sklearn.model_selection import KFold
import pandas as pd
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.keras.applications.vgg16
import openpyxl
from tf_keras_vis.saliency import Saliency
import winsound
#import autokeras as ak

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
get_images('C:\\Users\\u1056\\sfx\\images_sfx\\\With_Width\\OG')
# get_images('C:\\Users\\u1056\\sfx\\images_sfx\\\Original\\Flipping')
# get_images('C:\\Users\\u1056\\sfx\\images_sfx\\\Original\\Rotation')
# get_images('C:\\Users\\u1056\\sfx\\images_sfx\\\Original\\Color')
# get_images('C:\\Users\\u1056\\sfx\\images_sfx\\\Original\\Solarize')
# get_images('C:\\Users\\u1056\\sfx\\images_sfx\\\Original\\Posterize')

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
        if(step == max_steps_and_UCIs[folder_name][0] and UCI == max_steps_and_UCIs[folder_name][1]):
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

#Randomly divides into training, validation, and test sets
img_arr_list = np.stack(img_arr_list)
params_list = np.stack((height_list,phi_list,theta_list))
params_list = np.transpose(params_list)
values  = list(range(len(height_list)))
random.shuffle(values)
newarr = np.array_split(values, 10, axis=0)
train_idx = np.concatenate([newarr[0],newarr[1],newarr[2],newarr[3],newarr[4],newarr[5]])
test_idx = np.concatenate([newarr[6],newarr[7]])
val_idx = np.concatenate([newarr[8],newarr[9]])

for x in train_idx:
    x_train.append(img_arr_list[x])
    y_train.append(params_list[x])
for y in test_idx:
    x_test.append(img_arr_list[y])
    y_test.append(params_list[y])
for z in val_idx:
    x_val.append(img_arr_list[z])
    y_val.append(params_list[z])
x_train = np.stack(x_train)
x_test = np.stack(x_test)
x_val = np.stack(x_val)
y_train = np.stack(y_train)
y_test = np.stack(y_test)
y_val = np.stack(y_val)
val_data = [x_val,y_val]

train_images = x_train
train_labels = y_train
test_images = x_test
test_labels = y_test
val_images = x_val
val_labels = y_val

# Normalize pixel values to be between 0 and 1
train_images, test_images, val_images = train_images / 255.0, test_images / 255.0, val_images / 255.0

# Merge inputs and targets
inputs = np.concatenate((train_images, test_images, val_images), axis=0)
targets = np.concatenate((train_labels, test_labels, val_labels), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    # Define the model architecture
    cnn4 = Sequential([
    layers.Conv2D(filters=filter1, kernel_size=kernel_size, activation='relu',padding='same',input_shape=input_shape),
    layers.Conv2D(filters=filter1, kernel_size=kernel_size, actabaqusivation='relu',padding='same'),
    layers.MaxPooling2D(pool_size=pool_size,strides=strides),
    layers.Conv2D(filters=filter2, kernel_size=kernel_size, activation='relu'),
    layers.Conv2D(filters=filter2, kernel_size=kernel_size, activation='relu'),
    layers.MaxPooling2D(pool_size=pool_size,strides=strides),
    layers.Conv2D(filters=filter3, kernel_size=kernel_size, activation='relu'),
    layers.Conv2D(filters=filter3, kernel_size=kernel_size, activation='relu'),
    layers.MaxPooling2D(pool_size=pool_size,strides=strides),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(rate=dropout_rate),
    layers.Dense(3, activation="linear")])
    model=cnn4

    # Compile the model
    callback = callbacks.EarlyStopping(monitor='loss', patience=5)
    model.compile(optimizer= 'adam',
                loss="mean_squared_error",
                #metrics=[tfa.metrics.RSquare(multioutput = 'uniform_average', dtype=tf.float32, y_shape=(3,)), 'mean_squared_error'])
                metrics=[tfa.metrics.RSquare(multioutput = 'uniform_average', dtype=tf.float32), 'mean_squared_error'])

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    
    # Fit data to model
    history = model.fit(inputs[train], targets[train], epochs=num_epochs, validation_split = 0.2, callbacks=[callback])#, batch_size=len(inputs[train])

    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=2)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}; {model.metrics_names[2]} of {scores[2]}')
    loss_per_fold.append(scores[0])
    R2_per_fold.append(scores[1])
    MSE_per_fold.append(scores[2])
    
    # Increase fold number
    fold_no = fold_no + 1

    backend.clear_session()

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(loss_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} -  R-squared: {R2_per_fold[i]} - MSE: {MSE_per_fold[i]}')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Loss: {np.mean(loss_per_fold)}')
print(f'>  R-Squared: {np.mean(R2_per_fold)}')
print(f'>  MSE: {np.mean(MSE_per_fold)} (+- {np.std(MSE_per_fold)})')
print('------------------------------------------------------------------------')

#Saves to Excel File
d = {'R-Squared':np.array(R2_per_fold), 'MSE':np.array(MSE_per_fold), 'Fold':np.array(range(1,num_folds+1)), 'Mean R-Squared':[np.mean(R2_per_fold)], 'Mean MSE':[np.mean(MSE_per_fold)], '# Images':[len(image_name_list)], '# Folds':[num_folds], '# Epochs':[num_epochs]}
output = pd.DataFrame({k:pd.Series(v) for k,v in d.items()})
dfmaster = pd.read_excel('Documents\\PostProcessing_CNN.xlsx')
dfmaster = pd.concat([dfmaster,output])

with pd.ExcelWriter('Documents\\PostProcessing_CNN.xlsx',
                    engine="openpyxl", 
                    mode='a', if_sheet_exists='overlay') as writer:  
    dfmaster.to_excel(writer, sheet_name='Sheet1')

#Beeps when code ends
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
