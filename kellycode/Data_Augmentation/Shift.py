import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image

def Shift(pathname1, pathname2):
    dx = 100
    dy = 0
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
        
        img_arr = tfa.image.translate(img_arr,translations = [dx,dy],interpolation='bilinear', fill_mode='nearest')

        new_img = image_name.split('.png')[0]
        new_img = new_img.split(pathname1)[1]
        if dx==0:
            new_img = pathname2+"Shifting\\Translate-y"+new_img+"_trans_y.png"
        elif dy==0:
            new_img = pathname2+"Shifting\\Translate-x"+new_img+"_trans_x.png"
        folder_path = new_img.removesuffix(new_img.split("\\")[-1]).removesuffix("\\")
        #if the folder doesnt exist, make it
        if(not os.path.isdir(folder_path)):
            os.mkdir(folder_path)
        #if the image already exists, delete it
        if(os.path.isfile(new_img)):
            os.remove(new_img)
        save_img(new_img, img_arr)