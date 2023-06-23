from sklearn.inspection import permutation_importance
from GPR import *
import matplotlib.pyplot as plt

''' getting the feature importance using my model, all of my features in my test set, and all of my labels in my test set. '''
def get_and_plot_feature_importance(models, label_to_predict, saving_folder):
    for fold_no in range(0,5):
        result_test = permutation_importance(
            models[fold_no][0], models[fold_no][2], models[fold_no][1], n_repeats=100, random_state=42, n_jobs=2
        )
        i = 0
        importances = {}
        for col in models[fold_no][2].columns:
            importance = result_test['importances_mean'][i]
            importances[col] = importance
            i +=1


        #Create a figure and axis with 1 row and 5 columns
        fig, axs = plt.subplots(1, 5, figsize=(12, 6), sharey=True)

        # Iterate through the data sets and create box plots
        for i, data in enumerate(result_test['importances']):
            axs[i].boxplot(data)
            axs[i].set_title(correlated_featureset.columns[i])

        # Set y-axis label
        axs[0].set_ylabel('Feature Importance')

        # Set plot title
        fig.suptitle(f'Feature Importance Box and whisker plots for predicting {label_to_predict} fold {fold_no+1}')
        # Adjust spacing between subplots
        fig.tight_layout()
        # Display the plot
        # plt.show()
        plt.savefig(saving_folder + f'/Feature_Importance_{label_to_predict}_fold{fold_no+1}.png')
        plt.close()

full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/FULL_OG_dataframe_with_impact_sites_and_Jimmy_RF.csv"
image_folder = '/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx/new_dataset/Visible_cracks'
all_labels = ['height', 'phi', 'theta', 'impact site x', 'impact site y', 'impact site z', 'Jimmy_impact site x', 'Jimmy_impact site y', 
              'Jimmy_impact site z', 'Jimmy_impact site r', 'Jimmy_impact site phi', 'Jimmy_impact site theta']

    
label_to_predict = 'Jimmy_impact site x'
saving_folder=f'/Users/jakehirst/Desktop/sfx/Feature_importance_study/{label_to_predict}'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, raw_images, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.05)
correlated_featureset = remove_ABAQUS_features(correlated_featureset)
top_5_features = ['Jimmy_init phi', 'Jimmy_front 1 r', 'avg_ori', 'Jimmy_init r', 'Jimmy_init x']
correlated_featureset = correlated_featureset[top_5_features]
models = Kfold_Gaussian_Process_Regression(correlated_featureset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict, save_data=True)
get_and_plot_feature_importance(models, label_to_predict, saving_folder)

label_to_predict = 'Jimmy_impact site y'
saving_folder=f'/Users/jakehirst/Desktop/sfx/Feature_importance_study/{label_to_predict}'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, raw_images, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.05)
correlated_featureset = remove_ABAQUS_features(correlated_featureset)
top_5_features = ['Jimmy_front_0_z', 'max_kink', 'angle_btw', 'Jimmy_init theta', 'Jimmy_init y']
correlated_featureset = correlated_featureset[top_5_features]
models = Kfold_Gaussian_Process_Regression(correlated_featureset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict, save_data=True)
get_and_plot_feature_importance(models, label_to_predict, saving_folder)


label_to_predict = 'Jimmy_impact site z'
saving_folder=f'/Users/jakehirst/Desktop/sfx/Feature_importance_study/{label_to_predict}'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, raw_images, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.05)
correlated_featureset = remove_ABAQUS_features(correlated_featureset)
top_5_features = ['Jimmy_front_0_x', 'Jimmy_init phi', 'Jimmy_init y', 'Jimmy_init z', 'Jimmy_init theta']
correlated_featureset = correlated_featureset[top_5_features]
models = Kfold_Gaussian_Process_Regression(correlated_featureset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict, save_data=True)
get_and_plot_feature_importance(models, label_to_predict, saving_folder)




    

