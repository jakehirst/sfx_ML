""" This script turns the original ODB images from using "callodbimage.py" into square images """
import os
#from tf.keras.preprocessing.image import load_img, img_to_array, save_img
#from keras.preprocessing.image import *
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from PIL import Image

image_name_list = []
ODB_Image_filepath = "C:\\Users\\u1056\\OneDrive\\Desktop\\Original_Crack_Images"
for root, dirs, files in os.walk(ODB_Image_filepath):
    # select file name
        for file in files:
            # check the extension of files
            if file.endswith('.png'):
                if file.find("mesh")==-1:
                    image_name_list.append (os.path.join(root, file))
# for root, dirs, files in os.walk('Data Augmentation'):
#     # select file name
#         for file in files:
#             # check the extension of files
#             if file.endswith('.png'):
#                 if file.find("mesh")==-1:
#                     image_name_list.append (os.path.join(root, file))

for image_name in image_name_list:
    #img = tf.keras.preprocessing.image.load_img(image_name)
    img = load_img(image_name)#,color_mode = "grayscale") #For grayscale and binary
    img_arr = img_to_array(img)
    
    """ This binary block turns the images to black and white """
    #Binary
    thresh = 100
    maxval = 255
    img_arr = (img_arr > thresh) * maxval
    """ This binary block turns the images to black and white """


    # #Crop to square
    input_shape = img_arr.shape
    width = input_shape[1]
    height = input_shape[0]
    diff = (width-height)/3
    x1 = diff/width
    x2 = 1-(2*diff/width)
    boxes = [[0, x1, 1, x2]]
    box_indices = [0]
    crop_size = [height,height]
    img_arr = tf.expand_dims(img_arr, axis=0)
    img_arr = tf.image.crop_and_resize(img_arr, boxes, box_indices, crop_size)
    img_arr = tf.squeeze(img_arr)
    
    new_img = image_name.split('.png')[0]
    new_img = new_img.split('Desktop\\Original_Crack_Images')[1]

    #square_image_folder_filepath = "C:\\Users\\u1056\\OneDrive\\Desktop\\Square_Images"
    square_image_folder_filepath = "C:\\Users\\u1056\\OneDrive\\Desktop\\Square_Binary_Images"

    new_img = square_image_folder_filepath+new_img+"_square.png"
    save_img(new_img, img_arr)