# importing PIL Module
from PIL import Image, ImageEnhance, ImageOps
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import os
import random


def Contrast(pathname1, pathname2):

    image_name_list = []
    for root, dirs, files in os.walk(pathname1):
        # select file name
            for file in files:
                # check the extension of files
                if file.endswith('.png'):
                    if file.find("mesh")==-1:
                        image_name_list.append (os.path.join(root, file))

    for image_name in image_name_list:
        # open the original image
        original_img = load_img(image_name)

        #image brightness enhancer
        enhancer = ImageEnhance.Contrast(original_img)
        factor = 1.5 
        im_output = enhancer.enhance(factor)

        new_img = image_name.split('.png')[0]
        new_img = new_img.split(pathname1)[1]
        if factor<1:
            new_img = pathname2+"Contrast\\Decrease Contrast"+new_img+"_deccont.png"
        if factor>1:
            new_img = pathname2+"Contrast\\Increase Contrast"+new_img+"_inccont.png"
        folder_path = new_img.removesuffix(new_img.split("\\")[-1]).removesuffix("\\")
        #if the folder doesnt exist, make it
        if(not os.path.isdir(folder_path)):
            os.mkdir(folder_path)
        #if the image already exists, delete it
        if(os.path.isfile(new_img)):
            os.remove(new_img)
        im_output.save(new_img)          
        # close all our files object
        original_img.close()
        im_output.close()
