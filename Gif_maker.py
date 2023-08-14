from PIL import Image
import os

def extract_step_number(filename):
    start_index = filename.index('Step') + 4
    end_index = filename.index('_UCI')
    return int(filename[start_index:end_index])

def extract_uci_number(filename):
    start_index = filename.index('UCI_') + 4
    end_index = filename.index('_Dynamic')
    return int(filename[start_index:end_index])

def create_gif(folder_path, gif_path, duration):
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')],
                         key=lambda x: (extract_step_number(x), extract_uci_number(x)))
    frames = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        frames.append(image)

    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)


# Usage example
folder_path = '/Users/jakehirst/Desktop/sfx/Presentations_and_Papers/USNCCM/gif_stuff'  # Replace with the path to your image folder
gif_path = '/Users/jakehirst/Desktop/sfx/Presentations_and_Papers/USNCCM/GIF1.gif'  # Replace with the desired path for the output GIF
duration = 200  # Time (in milliseconds) each frame should be displayed

create_gif(folder_path, gif_path, duration)