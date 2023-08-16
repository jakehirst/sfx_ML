import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys
sys.path.insert(0,'C:\\Users\\u1056\\sfx\\sfx_ML\\sfx_ML\\Feature_gathering') #making sure i can import things from here
from Feature_gathering.initiation_sites import *
from Feature_gathering.thickness import *
from Feature_gathering.front_locations import *
from Feature_gathering.linearity import *
from scipy.stats import pearsonr



#gets the maximum steps and UCIs from the simulation_results folder
def get_max_step_and_max_UCIs(folder_path):
    # folder_path = FOLDER_PATH

    dic = {}
    for root, dirs, files in os.walk(folder_path):
        # select file name
        for file in files:
            # check the extension of files
            if (file.startswith('Para') and file.__contains__("original") and file.endswith(".frt")):
                simulation = file.split("_Stp")[0]
                step = file.split("_")[-4].split("p")[1]
                uci = file.split("_")[-2]

                if(dic.keys().__contains__(simulation)):
                    if((int(step) > int(dic[simulation][0])) or (int(uci) > int(dic[simulation][1]))):
                        dic[simulation] = [step, uci]
                else:
                    dic[simulation] = [step, uci]

    return dic



OG_SIMULATION_FOLDER_PATH = "F:\\Jake\\good_simies\\"
NEW_R_CURVE_SIMULATION_FOLDER_PATH = "F:\\Jake\\new_R_curve_5000_kIc\\"

og_maxes = get_max_step_and_max_UCIs(OG_SIMULATION_FOLDER_PATH)
new_R_curve_maxes = get_max_step_and_max_UCIs(NEW_R_CURVE_SIMULATION_FOLDER_PATH)

# Create an empty DataFrame with column names
columns = ['height', 'new length', 'old length']
heights_and_lens = pd.DataFrame(columns=columns)

for key in new_R_curve_maxes.keys():
    final_front_locations = get_final_front_locations(NEW_R_CURVE_SIMULATION_FOLDER_PATH, key)
    inner_surface_nodes, outer_surface_nodes, main_side_nodes, node_locations = get_main_side_outer_and_inner_surface_nodes(NEW_R_CURVE_SIMULATION_FOLDER_PATH, key, get_max_dynamic_step(NEW_R_CURVE_SIMULATION_FOLDER_PATH, key))
    new_len = get_crack_len(outer_surface_nodes, main_side_nodes, node_locations, final_front_locations)

    final_front_locations = get_final_front_locations(OG_SIMULATION_FOLDER_PATH, key)
    inner_surface_nodes, outer_surface_nodes, main_side_nodes, node_locations = get_main_side_outer_and_inner_surface_nodes(OG_SIMULATION_FOLDER_PATH, key, get_max_dynamic_step(OG_SIMULATION_FOLDER_PATH, key))
    og_len = get_crack_len(outer_surface_nodes, main_side_nodes, node_locations, final_front_locations)
    
    print('\n')
    print(key)
    print('new' + str(new_len))
    print('old' + str(og_len))
    print('\n')
    new_row = {
        'height': float(key.split('_')[1].split('ft')[0].replace('-', '.')),
        'new length': new_len,
        'old length': og_len
    }
    heights_and_lens = heights_and_lens.append(new_row, ignore_index=True)


    
# plt.scatter(heights_and_lens['height'], heights_and_lens['old length'], c='red', label='KIc initial = 2187.23 kPa*sqrt(mm)')
# plt.scatter(heights_and_lens['height'], heights_and_lens['new length'], c='blue', label='KIc initial = 5000 kPa*sqrt(mm)')
# plt.xlabel('fall height (ft)')
# plt.ylabel('crack len (mm)')
# plt.legend(loc = 'lower right')
# plt.title('Fall heights vs crack lengths for different initial KIc')
# plt.show()

# Scatter plots
plt.scatter(heights_and_lens['height'], heights_and_lens['old length'], c='red', label='KIc initial = 2187.23 kPa*sqrt(mm)')
plt.scatter(heights_and_lens['height'], heights_and_lens['new length'], c='blue', label='KIc initial = 5000 kPa*sqrt(mm)')

# Connect dots with same x-value with black dotted line
for x, old_len, new_len in zip(heights_and_lens['height'], heights_and_lens['old length'], heights_and_lens['new length']):
    plt.plot([x, x], [old_len, new_len], color='black', linestyle='dotted')

plt.xlabel('fall height (ft)')
plt.ylabel('crack len (mm)')
plt.legend(loc='lower right')
plt.title('Fall heights vs crack lengths for different initial KIc')
plt.show()

plt.close()

new_correlation, new_p_value = pearsonr(heights_and_lens['height'], heights_and_lens['new length'])
old_correlation, old_p_value = pearsonr(heights_and_lens['height'], heights_and_lens['old length'])

print("new pearson correlation value with height and crack length = " + str(new_correlation) + f' with p value {new_p_value}')
print("old pearson correlation value with height and crack length = " + str(old_correlation) + f' with p value {old_p_value}')


print('here')
