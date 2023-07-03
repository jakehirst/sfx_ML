from GPR import *
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt



def plot_mean_r2_and_mse(subsets_and_r2s, subsets_and_mse_s, figure_folder, label_to_predict):
    #get all of the means of the r^2 values for each subset of features
    subsets = list(subsets_and_r2s.keys())
    mean_r2s = []
    for i in range(len(subsets)):
        mean_r2s.append(subsets_and_r2s[subsets[i]].mean()) 
    
    #turn the data into a dataframe and sort it based on the mean r^2 values
    data = {'subsets': subsets, 'mean_r2s': mean_r2s}
    mean_r2_df = pd.DataFrame(data)
    mean_r2_df = mean_r2_df.sort_values(by='mean_r2s').reset_index(drop=True)

    #if there are more than 10 combos
    if(len(mean_r2_df) > 10):
        #only showing top 10
        mean_r2_df = mean_r2_df[len(mean_r2_df)-10:len(mean_r2_df)]
    #plot the mean r^2 values vs the features
    plt.figure(figsize=(18, 10))
    sns.barplot(x=mean_r2_df['subsets'].to_list(), y=mean_r2_df['mean_r2s'].to_list())
    # Rotate the x-axis labels vertically
    plt.xticks(rotation='vertical',fontsize=7, fontweight='bold')
    plt.subplots_adjust(bottom=0.37)
    plt.ylabel('mean r^2 values across 5 k-folds')
    plt.xlabel('feature subsets')
    plt.title('Comparing the r^2 values for each subset of features')
    # Display the plot
    # plt.show()
    plt.savefig(figure_folder + f'/Mean_R2_vs_subfeatures_plot_{label_to_predict}.png')
    plt.close()
        
        
    ''' now do the same thing for the MSE's'''
    
    #get all of the means of the r^2 values for each subset of features
    subsets = list(subsets_and_mse_s.keys())
    mean_MSEs = []
    for i in range(len(subsets)):
        mean_MSEs.append(subsets_and_mse_s[subsets[i]].mean()) 
    
    #turn the data into a dataframe and sort it based on the mean r^2 values
    data = {'subsets': subsets, 'mean_MSEs': mean_MSEs}
    mean_MSE_df = pd.DataFrame(data)
    mean_MSE_df = mean_MSE_df.sort_values(by='mean_MSEs').reset_index(drop=True)

    #if there are more than 10 combos
    if(len(mean_MSE_df) > 10):
        #only showing top 10
        mean_MSE_df = mean_MSE_df[0:10]

    #plot the mean MSE values vs the features
    plt.figure(figsize=(18, 10))
    sns.barplot(x=mean_MSE_df['subsets'].to_list(), y=mean_MSE_df['mean_MSEs'].to_list())
    # Rotate the x-axis labels vertically
    plt.xticks(rotation='vertical',fontsize=7, fontweight='bold')
    plt.subplots_adjust(bottom=0.37)
    plt.ylabel('mean MSE values across 5 k-folds')
    plt.xlabel('feature subsets')
    plt.title('Comparing the MSE values for each subset of features')
    # Display the plot
    # plt.show()
    plt.savefig(figure_folder + f'/Mean_MSE_vs_subfeatures_plot_{label_to_predict}.png')
    plt.close()
    
    return
    
    

full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/New_Crack_Len_FULL_OG_dataframe.csv"
image_folder = '/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx/new_dataset/Visible_cracks'
all_labels = ['height', 'phi', 'theta', 
              'impact site x', 'impact site y', 'impact site z', 
              'impact site r', 'impact site phi', 'impact site theta']


    
label_to_predict = 'impact site z'
saving_folder=f'/Users/jakehirst/Desktop/model_results/NEW_DATASET_GPR_{label_to_predict}_DotProduct_kernel/'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.1)
#correlated_featureset = remove_ABAQUS_features(correlated_featureset)

#when predicting impact site x
# All_features = ['front 1 z', 'crack len', 'front 1 x', 'init phi', 'avg_ori', 'init x']
#when predicting impact site y
# All_features = ['front_0_z', 'max_kink', 'angle_btw', 'init y']
#when predicting impact site z
All_features = ['init z', 'init theta']
#when predicting height
#All_features = []

#top 10s
#when predicting impact site x
# All_features = ['angle_btw', 'front 1 z', 'max thickness', 'crack len', 'front 1 x', 'init phi', 'avg_ori', 'front 1 r', 'init r', 'init x']
#when predicting impact site y
# All_features = ['abs_val_sum_kink', 'front 0 theta', 'front 0 phi', 'abs_val_mean_kink', 'front_0_y', 'front_0_z', 'max_kink', 'angle_btw', 'init theta', 'init y']
#when predicting impact site z
#All_features = ['front 0 theta', 'init x', 'front 0 r', 'linearity', 'init phi', 'max_kink', 'front_0_x', 'init z', 'init y', 'init theta']
#when predicting height
#All_features = ['mean thickness', 'front 0 phi', 'init x', 'init y', 'max thickness', 'dist btw frts', 'linearity', 'max_kink', 'crack len', 'abs_val_sum_kink']


figure_folder = f'/Users/jakehirst/Desktop/Feature_subset_study/predicting_{label_to_predict}'
if(not os.path.exists(figure_folder)):
    os.mkdir(figure_folder)

# Get all possible subsets
feature_subsets = []
for i in range(1, len(All_features) + 1):
    feature_subsets.extend(list(combinations(All_features, i)))

subsets_and_performances = {}
subsets_and_r2s = {}
subsets_and_mse_s = {}
for subset in feature_subsets:
    print(f'\nfeatures = {list(subset)}')
    featureset = correlated_featureset[list(subset)]
    raw_images = []
    models, performances, r2s, mse_s = Kfold_Gaussian_Process_Regression(featureset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict, save_data=False)
    subsets_and_performances[str(list(subset))] = performances
    subsets_and_r2s[str(list(subset))] = r2s
    subsets_and_mse_s[str(list(subset))] = mse_s

plot_mean_r2_and_mse(subsets_and_r2s, subsets_and_mse_s, figure_folder, label_to_predict)


print('done')
