# importing PIL Module
from PIL import Image, ImageEnhance, ImageOps
from keras.preprocessing.image import load_img, img_to_array, save_img
import os
import tensorflow_addons as tfa
import random
import tensorflow as tf

pathname1 = 'Fixed\\Wireframe Images'
pathname2 = 'Data Augmentation\\Wireframe Images\\' 
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
    img_arr = img_to_array(original_img)
    
    img_arr = tf.identity(img_arr)

    new_img = image_name.split('.png')[0]
    new_img = new_img.split(pathname1)[1]
    new_img = pathname2+"Identity"+new_img+"_ID.png"
    save_img(new_img, img_arr)
   
