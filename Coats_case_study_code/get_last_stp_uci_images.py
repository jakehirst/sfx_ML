from Models.CNN_function_library import *
import shutil

dataset = "new_dataset/Visible_cracks_new_dataset_2"
augmentation_list = ['OG']
args = prepare_data(dataset, augmentation_list)

original_images = args[0].tolist()
for og_img in original_images:
    parameters = og_img.split("/")[-2] + ".png"
    destination_file = "/Users/jakehirst/Desktop/sfx_last_images/" + parameters
    shutil.copy(og_img, destination_file)


print("done")