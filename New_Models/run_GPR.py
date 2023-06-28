from GPR import *


full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/New_Crack_Len_FULL_OG_dataframe.csv"
# full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/FULL_OG_dataframe_with_impact_sites_and_Jimmy_RF.csv"
image_folder = '/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx/new_dataset/Visible_cracks'
all_labels = ['height', 'phi', 'theta', 
              'impact site x', 'impact site y', 'impact site z', 
              'impact site r', 'impact site phi', 'impact site theta']


    
label_to_predict = 'impact site x'
saving_folder=f'/Users/jakehirst/Desktop/model_results/NEW_DATASET_GPR_{label_to_predict}/'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.05)
#correlated_featureset = remove_ABAQUS_features(correlated_featureset)
top_5_features = ['init phi', 'front 1 r', 'avg_ori', 'init r', 'init x']
correlated_featureset = correlated_featureset[top_5_features]
raw_images = []
Kfold_Gaussian_Process_Regression(correlated_featureset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict)


label_to_predict = 'impact site y'
saving_folder=f'/Users/jakehirst/Desktop/model_results/NEW_DATASET_GPR_{label_to_predict}/'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.05)
# correlated_featureset = remove_ABAQUS_features(correlated_featureset)
top_5_features = ['front_0_z', 'max_kink', 'angle_btw', 'init theta', 'init y']
correlated_featureset = correlated_featureset[top_5_features]
raw_images = []
Kfold_Gaussian_Process_Regression(correlated_featureset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict)


label_to_predict = 'impact site z'
saving_folder=f'/Users/jakehirst/Desktop/model_results/NEW_DATASET_GPR_{label_to_predict}/'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.05)
# correlated_featureset = remove_ABAQUS_features(correlated_featureset)
top_5_features = ['front_0_x', 'init phi', 'init y', 'init z', 'init theta']
correlated_featureset = correlated_featureset[top_5_features]
raw_images = []
Kfold_Gaussian_Process_Regression(correlated_featureset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict)

# label_to_predict = 'height'
# saving_folder=f'/Users/jakehirst/Desktop/model_results/NEW_DATASET_GPR_{label_to_predict}/'
# if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
# correlated_featureset, raw_images, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.05)
# correlated_featureset = remove_ABAQUS_features(correlated_featureset)
# # top_5_features = ['Jimmy_front_0_x', 'Jimmy_init phi', 'Jimmy_init y', 'Jimmy_init z', 'Jimmy_init theta']
# # correlated_featureset = correlated_featureset[top_5_features]
# Kfold_Gaussian_Process_Regression(correlated_featureset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict)


""" plotting all the predictions for specific models """
saving_folder=f'/Users/jakehirst/Desktop/model_results/GPR_prediction_plots/'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
labels_to_predict = ['Jimmy_impact site x', 'Jimmy_impact site y', 'Jimmy_impact site z']
models_fold_to_pull = {'Jimmy_impact site x': 1,
                       'Jimmy_impact site y': 4,
                       'Jimmy_impact site z': 5
                       }

all_important_features = {'Jimmy_impact site x': eval(pd.read_csv(f'/Users/jakehirst/Desktop/model_results/GPR_{labels_to_predict[0]}/model_metrics_fold_{models_fold_to_pull[labels_to_predict[0]]}.csv')['features_used'][0]),
                          'Jimmy_impact site y': eval(pd.read_csv(f'/Users/jakehirst/Desktop/model_results/GPR_{labels_to_predict[1]}/model_metrics_fold_{models_fold_to_pull[labels_to_predict[1]]}.csv')['features_used'][0]),
                          'Jimmy_impact site z': eval(pd.read_csv(f'/Users/jakehirst/Desktop/model_results/GPR_{labels_to_predict[2]}/model_metrics_fold_{models_fold_to_pull[labels_to_predict[2]]}.csv')['features_used'][0])
                          }
full_dataset = pd.read_csv(full_dataset_pathname, index_col=[0])
plot_test_predictions_heatmap(full_dataset, labels_to_predict, all_labels, all_important_features, models_fold_to_pull, saving_folder)