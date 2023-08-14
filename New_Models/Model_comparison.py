import numpy as np
import pandas as pd
import os
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from tabulate import tabulate


label_to_predict = 'impact site y'

parent_folder_path = '/Users/jakehirst/Desktop/model_results/MODEL_COMPARISONS'
# Model_Folders = ['/Users/jakehirst/Desktop/model_results/MODEL_COMPARISONS/CNN_no_images_impact site x', 
#                  '/Users/jakehirst/Desktop/model_results/MODEL_COMPARISONS/GPR_impact site x']

def get_all_folders(parent_folder_path, label_to_predict):
    # Get all subdirectories within the parent folder
    subdirectories = [os.path.join(parent_folder_path, name) for name in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, name))]

    # Filter the subdirectories to those ending with 'impact site x'
    filtered_subdirectories = [folder_path for folder_path in subdirectories if folder_path.endswith(label_to_predict)]
    return filtered_subdirectories
    
Model_Folders = get_all_folders(parent_folder_path, label_to_predict)

All_model_metrics = {}
for i in range(len(Model_Folders)):
    folder = Model_Folders[i]
    r2 = []; adj_r2 = []; MsE = []; MaE = []; RmSE = []
    for fold in range(1,6):
        current_metrics = pd.read_csv(folder + f'/model_metrics_fold_{fold}.csv')
        r2.append(current_metrics['r^2'][0])
        adj_r2.append(current_metrics['adj_r^2'][0])
        MsE.append(current_metrics['MSE'][0])
        MaE.append(current_metrics['MAE'][0])
        RmSE.append(current_metrics['RMSE'][0])
    
    All_model_metrics[folder.split('/')[-1].split('_')[0]] = {'R Squared': np.mean(r2),
                                                              'Adjusted R Squared': np.mean(adj_r2),
                                                              'MSE': np.mean(MsE),
                                                              'MAE': np.mean(MaE),
                                                              'RMSE': np.mean(RmSE)}

# for metric in list(All_model_metrics[list(All_model_metrics.keys())[0]].keys()):
#     x = []
#     y = []
#     for model_type in All_model_metrics.keys():
#         y.append(All_model_metrics[model_type][metric])
#         x.append(model_type)
    
#     # Create the bar plot
#     plt.bar(x, y)

#     # Add labels and title
#     plt.xlabel('Model type')
#     plt.ylabel(metric)
#     plt.title(f'Comparing {metric} values across all models')
#     plt.show()
#     plt.close()


data = [['Model Type', 'R Squared', 'Adj R Squared', 'MSE', 'MAE', 'RMSE']]
for model_type in sorted(All_model_metrics.keys()):
    row = []
    row.append([model_type])
    for metric in list(All_model_metrics[list(All_model_metrics.keys())[0]].keys()):
        row.append(round(All_model_metrics[model_type][metric], 3))
    data.append(row)
    

 # Create the table
table = tabulate(data, headers="firstrow", tablefmt="fancy_grid")

# Create a figure and axes
fig, ax = plt.subplots(figsize=(10, 5))

# Hide axis
ax.axis('off')

# Create the table within the axes
table_img = ax.table(cellText=data, colLabels=None, loc='center')

# Set table properties
table_img.auto_set_font_size(False)
table_img.set_fontsize(10)
table_img.scale(1.2, 1.2)

# Save the table as a .png image
plt.title(f'Model Comparisons when predicting {label_to_predict}')
plt.savefig(parent_folder_path + f'/metric_comparisons_{label_to_predict}.png', bbox_inches='tight', dpi=300)   
    
    
        