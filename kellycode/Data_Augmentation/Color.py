# importing PIL Module
from PIL import Image, ImageEnhance, ImageOps
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import os
import random
import tensorflow as tf
from multiprocess import Pool



def color_image_and_save(image_name, pathname1, pathname2):
    # open the original image
    img = load_img(image_name)
    d = img.getdata()
 
    new_image = []
    for item in d:
    
        # change all white (also shades of whites)
        # pixels to yellow
        if item[0] in list(range(1, 254)):
            new_image.append((255, 224, 100))
        else:
            new_image.append(item)
            
    # update image data
    img.putdata(new_image)
    new_img = image_name.split('.png')[0]
    new_img = new_img.split(pathname1)[1]
    new_img = pathname2+"Color\\"+new_img+"_color.png"
    folder_path = new_img.removesuffix(new_img.split("\\")[-1]).removesuffix("\\")
    #if the folder doesnt exist, make it
    if(not os.path.isdir(folder_path)):
        os.mkdir(folder_path)
    #if the image already exists, delete it
    if(os.path.isfile(new_img)):
        os.remove(new_img)
    # img.save(new_img)
    return [img, new_img]

if __name__ == "__main__":
    pathname1 = 'C:\\Users\\u1056\\sfx\\images_sfx\\Original\\OG'
    pathname2 = 'C:\\Users\\u1056\\sfx\\images_sfx\\Original\\' 
    image_name_list = []
    for root, dirs, files in os.walk(pathname1):
        # select file name
            for file in files:
                # check the extension of files
                if file.endswith('.png'):
                    if file.find("mesh")==-1:
                        image_name_list.append (os.path.join(root, file))

    p = Pool(10)
    colored = []
    for image_name in image_name_list:
        #p.apply_async(color_image_and_save , [image_name, pathname1, pathname2])
        color = p.apply_async(color_image_and_save , [image_name, pathname1, pathname2])
        colored.append(color)

    for r in colored:
        r.wait()
    for r in colored:
        r._value[0].save(r._value[1])
    p.close()
    p.join()

    
