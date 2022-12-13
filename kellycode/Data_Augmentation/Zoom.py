import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import tensorflow as tf
from PIL import Image
from keras import layers


def Zoom(pathname1, pathname2):
    image_name_list = []
    for root, dirs, files in os.walk(pathname1):
        # select file name
            for file in files:
                # check the extension of files
                if file.endswith('.png'):
                    if file.find("mesh")==-1:
                        image_name_list.append (os.path.join(root, file))

    for image_name in image_name_list:
        img = load_img(image_name)
        img_arr = img_to_array(img)
        
        img_arr = layers.RandomZoom(height_factor=(-0.3, -0.2),interpolation='bilinear', fill_mode='nearest')(img_arr)

        new_img = image_name.split('.png')[0]
        new_img = new_img.split(pathname1)[1]
        new_img = pathname2+"Zooming"+new_img+"_zoom.png"
        folder_path = new_img.removesuffix(new_img.split("\\")[-1]).removesuffix("\\")
        #if the folder doesnt exist, make it
        if(not os.path.isdir(folder_path)):
            os.mkdir(folder_path)
        #if the image already exists, delete it
        if(os.path.isfile(new_img)):
            os.remove(new_img)
        save_img(new_img, img_arr)