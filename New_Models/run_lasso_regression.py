from lasso_regression import *


full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/New_Crack_Len_FULL_OG_dataframe.csv"
image_folder = '/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx/new_dataset/Visible_cracks'
all_labels = ['height', 'phi', 'theta', 
              'impact site x', 'impact site y', 'impact site z', 
              'impact site r', 'impact site phi', 'impact site theta']



alpha = 0.1
    
    
label_to_predict = 'impact site x'
saving_folder=f'/Users/jakehirst/Desktop/model_results/MODEL_COMPARISONS/Lasso_Reg_{label_to_predict}/'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.05)
#correlated_featureset = remove_ABAQUS_features(correlated_featureset)
features_to_keep = ['crack len', 'init phi', 'init x']
correlated_featureset = correlated_featureset[features_to_keep]
raw_images = []
Kfold_Lasso_Regression(alpha, correlated_featureset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict)


label_to_predict = 'impact site y'
saving_folder=f'/Users/jakehirst/Desktop/model_results/MODEL_COMPARISONS/Lasso_Reg_{label_to_predict}/'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.05)
# correlated_featureset = remove_ABAQUS_features(correlated_featureset)
features_to_keep = ['max_kink', 'init y']
correlated_featureset = correlated_featureset[features_to_keep]
raw_images = []
Kfold_Lasso_Regression(alpha, correlated_featureset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict)


# label_to_predict = 'impact site z'
# saving_folder=f'/Users/jakehirst/Desktop/model_results/NEW_DATASET_Ridge_Regression_{label_to_predict}_alpha{alpha}/'
# if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
# correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.05)
# # correlated_featureset = remove_ABAQUS_features(correlated_featureset)
# top_5_features = ['front_0_x', 'init phi', 'init y', 'init z', 'init theta']
# correlated_featureset = correlated_featureset[top_5_features]
# raw_images = []
# Kfold_Lasso_Regression(alpha, correlated_featureset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict)

