# importing PIL Module
from PIL import Image
import os

def vertical_flip(pathname1, pathname2):

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
        original_img = Image.open(image_name)
        
        # Flip the original image vertically
        vertical_img = original_img.transpose(method=Image.FLIP_TOP_BOTTOM)
        new_img = image_name.split('.png')[0]
        new_img = new_img.split(pathname1)[1]
        new_img = pathname2+"Flipping\\Vertical Flip\\"+new_img+"_vert.png"
        folder_path = new_img.removesuffix(new_img.split("\\")[-1]).removesuffix("\\")
        #if the folder doesnt exist, make it
        if(not os.path.isdir(folder_path)):
            os.mkdir(folder_path)
        #if the image already exists, delete it
        if(os.path.isfile(new_img)):
            os.remove(new_img)
        vertical_img.save(new_img)
        
        # close all our files object
        original_img.close()
        vertical_img.close()
