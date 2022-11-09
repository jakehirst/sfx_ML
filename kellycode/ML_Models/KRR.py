from sklearn.feature_selection import chi2
from sklearn.kernel_ridge import KernelRidge
import numpy as np
from matplotlib import pyplot as plt
import os
import random
from sklearn.model_selection import KFold
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
import openpyxl
import winsound
import tensorflow as tf

num_folds = 3
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
R2_train_per_fold = []
R2_test_per_fold = []

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
get_images('Fixed\\Wireframe Images')
get_images('Data Augmentation\\Wireframe Images\\autoContrast')
get_images('Data Augmentation\\Wireframe Images\\Color')
get_images('Data Augmentation\\Wireframe Images\\Equalize')
get_images('Data Augmentation\\Wireframe Images\\Gaussian Noise')
get_images('Data Augmentation\\Wireframe Images\\Identity')
get_images('Data Augmentation\\Wireframe Images\\Posterize')
get_images('Data Augmentation\\Wireframe Images\\Sharpness')
get_images('Data Augmentation\\Wireframe Images\\Solarize')

#Extracts parameters from image names
for image_name in image_name_list:
    img = load_img(image_name)
    img_arr = img_to_array(img)
    input_shape = img_arr.shape

    img_arr_list.append(img_arr)
    height = image_name.split('ft_')[0]
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

#Randomly divides into training, validation, and test sets
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

#sklearn expects i/p to be 2d array-model.fit(x_train,y_train)=>reshape to 2d array
nsamples, nx, ny, nrgb = inputs.shape
inputs = inputs.reshape((nsamples,nx*ny*nrgb))

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1

# #Histogram intersection kernel function
# def histogram_intersection(A,B):
#     hist_1, _ = np.histogram(A, bins=256)
#     hist_2, _ = np.histogram(B, bins=256)
#     minima = np.minimum(hist_1, hist_2)
#     intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
#     return intersection

for train, test in kfold.split(inputs, targets):
    regr = KernelRidge(kernel = 'linear')
    regr.fit(X=inputs[train], y=targets[train])
    R2_train = regr.score(X=inputs[train], y=targets[train])
    R2_test = regr.score(X=inputs[test], y=targets[test])
    R2_train_per_fold.append(R2_train)
    R2_test_per_fold.append(R2_test)
    print("Fold "+str(fold_no))
    print("R^2 Train:"+str(R2_train))
    print("R^2 Test:"+str(R2_test))
    # Increase fold number
    fold_no = fold_no + 1

#Saves to Excel File
d = {'R-Squared Train':np.array(R2_train_per_fold), 'R-Squared Test':np.array(R2_test_per_fold), 'Fold':np.array(range(1,num_folds+1)), 'Mean R-Squared Train':[np.mean(R2_train_per_fold)], 'Mean R-Squared Test':[np.mean(R2_test_per_fold)], '# Images':[len(image_name_list)], '# Folds':[num_folds]}
output = pd.DataFrame({k:pd.Series(v) for k,v in d.items()})
dfmaster = pd.read_excel('Documents\\PostProcessing_KRR.xlsx')
dfmaster = pd.concat([dfmaster,output])

with pd.ExcelWriter('Documents\\PostProcessing_KRR.xlsx',
                    engine="openpyxl", 
                    mode='a', if_sheet_exists='overlay') as writer:  
    dfmaster.to_excel(writer, sheet_name='Sheet1')

#Beeps when code ends
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
