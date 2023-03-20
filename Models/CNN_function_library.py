import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd

#Gathers image list
def get_images(pathname, image_name_list):
    print(pathname)
    for root, dirs, files in os.walk(pathname):
        # select file name
            for file in files:
                # check the extension of files
                if file.endswith('.png'):
                    if file.find("mesh")==-1:
                        image_name_list.append (os.path.join(root, file))
    return image_name_list

def prepare_data(parent_folder_name, augmentation_list):
    image_name_list = []
    
    #in lab
    #image_name_list = get_images('C:\\Users\\u1056\\sfx\\images_sfx\\' + parent_folder_name + "\\" + "OG", image_name_list)
    
    #at home
    image_name_list = get_images('/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx/' + parent_folder_name + "/" + "OG", image_name_list)
    
    # for folder in augmentation_list:
    #     image_name_list = get_images('C:\\Users\\u1056\\sfx\\images_sfx\\' + parent_folder_name + "\\" + folder, image_name_list)

    #finds the max uci and step for each fall parameter image folder
    max_steps_and_UCIs = dict()
    for image_path in image_name_list:
        #if(image_path.endswith("Dynamic.png") or image_path.split("_")[-2] == "Dynamic"):
        if(image_path.endswith("Dynamic.png")):
            # if(image_path.__contains__("Para_2ft_PHI_30_THETA_135")):
            #     print(image_path)
            
            #in lab
            # image_name = image_path.split("\\")[-1]
            # folder_name = image_path.split("\\")[-2]
            
            #at home
            image_name = image_path.split("/")[-1]
            folder_name = image_path.split("/")[-2]            
            
            UCI = int(image_name.split("_")[2])
            step = int(image_name.split("_")[0].split("p")[1])
            if(not (folder_name in max_steps_and_UCIs.keys())):
                max_steps_and_UCIs[folder_name] = [step, UCI]
            else:
                if(step > max_steps_and_UCIs[folder_name][0]):
                    max_steps_and_UCIs[folder_name] = [step, UCI]
                elif(step == max_steps_and_UCIs[folder_name][0] and UCI > max_steps_and_UCIs[folder_name][1]):
                    max_steps_and_UCIs[folder_name] = [step, UCI]
        else:
            continue

    img_arr_list = []
    image_path_list = []
    height_list = []
    phi_list = []
    theta_list = []
    #Extracts parameters from image names
    for image_path in image_name_list:
        if(image_path.endswith("Dynamic.png") or image_path.split("_")[-2] == "Dynamic"):
            #in lab
            # image_name = image_path.split("\\")[-1]
            # folder_name = image_path.split("\\")[-2]
            
            #at home
            image_name = image_path.split("/")[-1]
            folder_name = image_path.split("/")[-2]
               
            UCI = int(image_name.split("_")[2])
            step = int(image_name.split("_")[0].split("p")[1])
            #Only selects the images from the max step/uci combinations
            if(step == max_steps_and_UCIs[folder_name][0] and UCI == max_steps_and_UCIs[folder_name][1]):
                image_path_list.append(str(image_path))
                img = load_img(image_path)#,color_mode = "grayscale")
                img_arr = img_to_array(img)
                input_shape = img_arr.shape
                img_arr_list.append(img_arr)
                
                #in lab
                #image_name = image_path.split('\\')[-1]
                
                #at home
                image_name = image_path.split('/')[-1]
                
                height = folder_name.split('ft_')[0]
                height = height.split('Para_')[1]
                height = height.replace('-', '.')
                height_list = np.append(height_list,float(height)) 
                phi = folder_name.split('_THETA')[0]
                phi = phi.split('PHI_')[1]
                phi_list = np.append(phi_list,float(phi)) 
                if image_name.find("LOCAL")==-1:
                    theta = folder_name.split('_Stp')[0]
                else:
                    theta = folder_name.split('_LOCAL')[0]
                theta = theta.split('THETA_')[1].replace(".png", "")
                theta = theta.split("_")[0]
                theta_list = np.append(theta_list,float(theta))
                
        else:
            continue

    image_path_list = pd.Series(image_path_list).astype(str)
    height_list = pd.Series(height_list)
    phi_list = pd.Series(phi_list)
    theta_list = pd.Series(theta_list)

    """ img_arr_list contains all of the images for training (final steps and ucis) in matrix form"""
    print("number of examples: " + str(len(img_arr_list)))
    print("number of heights = " + str(len(height_list)))
    print("number of phis = " + str(len(phi_list)))
    print("number of thetas = " + str(len(theta_list)))
    return [image_path_list, height_list, phi_list, theta_list, input_shape, img_arr_list]



def add_augmentations_bins(df, augmentation_list):
    image_name_list = []
    new_df = df
    if(augmentation_list.__contains__("OG")):
        augmentation_list.remove("OG")
    parent_folder_name = "/Original"
    for folder in augmentation_list:
        for row in df.iterrows():
            
            #in lab
            # OG_picture_step_UCI = row[1]["Filepath"].split("\\")[-1].split(".")[0]
            # parameters = row[1]["Filepath"].split("\\")[-2]
            
            #at home
            OG_picture_step_UCI = row[1]["Filepath"].split("/")[-1].split(".")[0]
            parameters = row[1]["Filepath"].split("/")[-2]
            
            #print("\nrow filepath = " + str(row[1]["Filepath"]))
            #print(row[1].to_frame().transpose())

            #in lab
            #pathname = 'C:\\Users\\u1056\\sfx\\images_sfx\\' + parent_folder_name + "\\" + folder
            
            #at home
            pathname = '/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx' + parent_folder_name + '/' + folder
            
            for root, dirs, files in os.walk(pathname):
            # select file name
                for file in files:
                    # check the extension of files
                    if file.endswith('.png'):
                        if file.find("mesh")==-1:
                            if((OG_picture_step_UCI in file) and (parameters in root)):
                                path = os.path.join(root, file)
                                new_row = row[1].to_frame().transpose()
                                new_row.iloc[0]["Filepath"] = path
                                new_df = pd.concat([new_df, new_row], ignore_index=True) 
    for i in range(len(new_df.columns) - 1):
        new_df[[str(i)]] = new_df[[str(i)]].apply(pd.to_numeric)
    return new_df

def add_augmentations(df, augmentation_list):
    image_name_list = []
    new_df = df
    parent_folder_name = "/Original"

    if(augmentation_list.__contains__("OG")):
        augmentation_list.remove("OG")
    for folder in augmentation_list:
        for row in df.iterrows():
            #in lab
            # OG_picture_step_UCI = row[1]["Filepath"].split("\\")[-1].split(".")[0]
            # parameters = row[1]["Filepath"].split("\\")[-2]
            
            #at home
            OG_picture_step_UCI = row[1]["Filepath"].split("/")[-1].split(".")[0]
            parameters = row[1]["Filepath"].split("/")[-2]
            label = row[1][df.columns[1]]

            #in lab
            #pathname = 'C:\\Users\\u1056\\sfx\\images_sfx\\' + parent_folder_name + "\\" + folder
            
            #at home
            pathname = '/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx' + parent_folder_name + '/' + folder
            
            for root, dirs, files in os.walk(pathname):
            # select file name
                for file in files:
                    # check the extension of files
                    if file.endswith('.png'):
                        if file.find("mesh")==-1:
                            if((OG_picture_step_UCI in file) and (parameters in root)):
                                path = os.path.join(root, file)
                                new_row = pd.DataFrame([{'Filepath':str(path), df.columns[1]:label}])
                                new_df = pd.concat([new_df, new_row], ignore_index=True) 
                                #new_df.append({'Filepath':str(path), df.columns[1]:label}, ignore_index=True) #DONT use pd.concat here... it changes the type of things in the dataframe

    return new_df

def add_augmentations_clustering(df, augmentation_list, parent_folder_name="/Original"):
    image_name_list = []
    new_df = df
    if(augmentation_list.__contains__("OG")):
        augmentation_list.remove("OG")
    for folder in augmentation_list:
        for row in df.iterrows():
            
            #in lab
            # OG_picture_step_UCI = row[1]["Filepath"].split("\\")[-1].split(".")[0]
            # parameters = row[1]["Filepath"].split("\\")[-2]
            
            #at home
            OG_picture_step_UCI = row[1]["Filepath"].split("/")[-1].split(".")[0]
            parameters = row[1]["Filepath"].split("/")[-2]
            
            #print("\nrow filepath = " + str(row[1]["Filepath"]))
            #print(row[1].to_frame().transpose())

            #in lab
            #pathname = 'C:\\Users\\u1056\\sfx\\images_sfx\\' + parent_folder_name + "\\" + folder
            
            #at home
            pathname = '/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx' + parent_folder_name + '/' + folder
            
            for root, dirs, files in os.walk(pathname):
            # select file name
                for file in files:
                    # check the extension of files
                    if file.endswith('.png'):
                        if file.find("mesh")==-1:
                            if((OG_picture_step_UCI in file) and (parameters in root)):
                                path = os.path.join(root, file)
                                new_row = row[1].to_frame().transpose()
                                new_row.iloc[0]["Filepath"] = path
                                new_df = pd.concat([new_df, new_row], ignore_index=True) 
    for i in range(len(new_df.columns) - 1):
        new_df[[str(i)]] = new_df[[str(i)]].apply(pd.to_numeric)
    return new_df

def remove_augmentations(images):
    indexes_to_delete = []
    for i in range(images.n):
        if(not images.filenames[i].endswith("Dynamic.png")):
            indexes_to_delete.append(i)
    
    images.filenames = np.delete(images.filenames, indexes_to_delete)
    images.filepaths = (np.delete(np.array(images.filepaths), indexes_to_delete)).tolist()
    images.labels = np.delete(images.labels, indexes_to_delete)
    images._filepaths = (np.delete(np.array(images._filepaths), indexes_to_delete)).tolist()
    images._targets = np.delete(images._targets, indexes_to_delete)


    return images


def remove_augmentations(images):
    indexes_to_delete = []
    for i in range(images.n):
        if(not images.filenames[i].endswith("Dynamic.png")):
            indexes_to_delete.append(i)
    
    images.filenames = np.delete(images.filenames, indexes_to_delete)
    images.filepaths = (np.delete(np.array(images.filepaths), indexes_to_delete)).tolist()
    images.labels = np.delete(images.labels, indexes_to_delete)
    images._filepaths = (np.delete(np.array(images._filepaths), indexes_to_delete)).tolist()
    images._targets = np.delete(images._targets, indexes_to_delete)


    return images


def convert_to_xy(df, x_or_y):
    phi = np.asarray(df['phi'])
    theta = np.asarray(df['theta'])
    
    # x = r × cos( θ )
    # y = r × sin( θ )
    x = phi * np.cos(theta)
    y = phi * np.sin(theta)
    
    new_df = df.copy()
    
    if(x_or_y == "y"):
        new_df.insert(3, "y", y, True)
    elif(x_or_y == "x"):
        new_df.insert(3, "x", x, True)
    elif(x_or_y == "x_and_y"):
        new_df.insert(3, "x", x, True)
        new_df.insert(4, "y", y, True)

    new_df = new_df.drop("phi", axis=1)
    new_df = new_df.drop("theta", axis=1)
    
    
    
    return new_df