import os
from keras.preprocessing.image import load_img, img_to_array, save_img
import tensorflow as tf
from PIL import Image
from keras import layers
import tensorflow_addons as tfa

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
    img = load_img(image_name)
    img_arr = img_to_array(img)
    
    img_arr = tfa.image.shear_x(image = img_arr,level = 0.3,replace = 255)

    new_img = image_name.split('.png')[0]
    new_img = new_img.split(pathname1)[1]
    new_img = pathname2+"Shearing\\Shear-x"+new_img+"_shearx.png"
    save_img(new_img, img_arr)