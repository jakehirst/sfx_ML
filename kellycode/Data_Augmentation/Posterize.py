# importing PIL Module
from PIL import Image, ImageEnhance, ImageOps
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import os
import random
import tensorflow as tf
#import tensorflow_addons as tfa
def posterize(pathname1, pathname2):

# pathname1 = 'C:\\Users\\u1056\\sfx\\images_sfx\\With_Width\\OG'
# pathname2 = 'C:\\Users\\u1056\\sfx\\images_sfx\\With_Width\\' 
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
        img = load_img(image_name)
        bits = 4
        img = ImageOps.posterize(img, bits)
        new_img = image_name.split('.png')[0]
        new_img = new_img.split(pathname1)[1]
        new_img = pathname2+"Posterize\\"+new_img+"_post.png"
        folder_path = new_img.removesuffix(new_img.split("\\")[-1]).removesuffix("\\")
        #if the folder doesnt exist, make it
        if(not os.path.isdir(folder_path)):
            os.mkdir(folder_path)
        #if the image already exists, delete it
        if(os.path.isfile(new_img)):
            os.remove(new_img)
        img.save(new_img)

    
