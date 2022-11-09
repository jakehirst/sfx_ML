import tensorflow as tf
import numpy as np
from keras import datasets, layers, models, Sequential, losses, applications, Model, Input, optimizers
from matplotlib import pyplot as plt
import os
import random
from sklearn.model_selection import KFold
import pandas as pd
import tensorflow_addons as tfa
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import keras.applications.vgg16
import openpyxl
import autokeras as ak

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
get_images('Fixed\\Original Images')

for image_name in image_name_list:
    img = load_img(image_name)
    img_arr = img_to_array(img)
    input_shape = img_arr.shape
    img_arr_list.append(img_arr)

    height = image_name.split('ft')[0]
    height = height.split('Para_')[1]
    height_list = np.append(height_list,int(height)) 
    phi = image_name.split('_THETA')[0]
    phi = phi.split('PHI_')[1]
    phi_list = np.append(phi_list,int(phi)) 
    if image_name.find("LOCAL")==-1:
        theta = image_name.split('_Stp')[0]
    else:
        theta = image_name.split('_LOCAL')[0]
    theta = theta.split('THETA_')[1]
    theta_list = np.append(theta_list,int(theta))

img_arr_list = np.stack(img_arr_list)
params_list = np.stack((height_list,phi_list,theta_list))
params_list = np.transpose(params_list)
values  = list(range(len(image_name_list)))
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
x_train, x_val, x_test = x_train/255.0, x_val/255.0, x_test/255.0
val_data = [x_val,y_val]

# Initialize the image regressor.
reg = ak.ImageRegressor(overwrite=True, metrics=[tfa.metrics.RSquare(multioutput = 'uniform_average', dtype=tf.float32, y_shape=(3,)), 'mean_squared_error'])

# Feed the image regressor with training data.
reg.fit(
   x=x_train,
   y=y_train,
   validation_data=val_data,
   batch_size = 1)

# Predict with the best model.
predicted_y = reg.predict(x_test)
print(predicted_y)

# Evaluate the best model with testing data.
print(reg.evaluate(x=x_test, y=y_test))
reg.export_model()



