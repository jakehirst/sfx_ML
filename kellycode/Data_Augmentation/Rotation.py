# importing PIL Module
from PIL import Image
import os
 
pathname1 = 'C:\\Users\\u1056\\sfx\\images_sfx\\Original\\OG'
pathname2 = 'C:\\Users\\u1056\\sfx\\images_sfx\\Original\\' 
image_name_list = []
#angle = 330
angles = [30,60,90,120,150,180,210,240,270,300,330]
for root, dirs, files in os.walk(pathname1):
    # select file name
        for file in files:
            # check the extension of files
            if file.endswith('.png'):
                if file.find("mesh")==-1:
                    image_name_list.append (os.path.join(root, file))
white = (255,255,255)
for angle in angles:
    for image_name in image_name_list:
        # open the original image
        original_img = Image.open(image_name)
        
        # Rotate the original image
        changed_img = original_img.rotate(angle, fillcolor=white)
        new_img = image_name.split('.png')[0]
        new_img = new_img.split(pathname1)[1]
        new_img = pathname2+"Rotation\\Rotation "+str(angle)+"deg\\"+new_img+"_r"+str(angle)+".png"
        folder_path = new_img.removesuffix(new_img.split("\\")[-1]).removesuffix("\\")
        #if the folder doesnt exist, make it
        if(not os.path.isdir(folder_path)):
            os.mkdir(folder_path)
        #if the image already exists, delete it
        if(os.path.isfile(new_img)):
            os.remove(new_img)
        # close all our files object
        changed_img.save(new_img)
        original_img.close()
        changed_img.close()
