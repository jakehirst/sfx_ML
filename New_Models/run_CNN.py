from CNN import *

#for mac computer
full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_data/New_Crack_Len_FULL_OG_dataframe_2023_07_14.csv"
image_folder = '/Users/jakehirst/Desktop/sfx/sfx_ML_data/images_sfx/new_dataset/Visible_cracks'

#for work computer
# full_dataset_pathname = "C:\\Users\\u1056\\sfx\\sfx_ML_data\\feature_data\\New_Crack_Len_FULL_OG_dataframe_2023_07_14.csv"
# image_folder = 'C:\\Users\\u1056\\sfx\\sfx_ML_data\\images_sfx\\Visible_cracks_new_dataset_'


''' preparing data for logistic regression on impact site using k-means clustering'''
# label_to_predict = 'impact_cluster_assignments'
# num_clusters = 5
# saving_folder=f'/Users/jakehirst/Desktop/model_results/impact_site_logi_reg_{num_clusters}_clusters/'
# if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
# correlated_featureset, raw_images, full_dataset_labels = prepare_dataset_Kmeans_cluster(full_dataset_pathname, image_folder, label_to_predict, cluster='impact_sites', num_clusters=num_clusters, saving_folder=saving_folder)
# run_kfold_Categorical_CNN(correlated_featureset, raw_images, full_dataset_labels, patience=100, max_epochs=1000, saving_folder=saving_folder)

''' preparing data for logistic regression on height using discrete binning '''
# num_bins = 2
# saving_folder=f'/Users/jakehirst/Desktop/model_results/height_logi_reg_{num_bins}_bins/'
# if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
# correlated_featureset, raw_images, full_dataset_labels = prepare_dataset_discrete_Binning(full_dataset_pathname, image_folder, 'height', num_bins=num_bins, saving_folder=None)
# run_kfold_Categorical_CNN(correlated_featureset, raw_images, full_dataset_labels, patience=100, max_epochs=1000, saving_folder=saving_folder)


all_labels = ['height', 'phi', 'theta', 
              'impact site x', 'impact site y', 'impact site z']


# lossfunc = 'mean_distance_error_phi_theta',
# lossfunc = 'mean_absolute_error', 
# lossfunc = 'mean_squared_error',
# lossfunc = 'mean_squared_logarithmic_error', - BAD FOR [phi]
# lossfunc = tf.keras.losses.CosineSimilarity(axis=1) - BAD FOR [phi, theta]
# lossfunc = tf.keras.losses.Huber()
# lossfunc = tf.keras.losses.LogCosh()

label_to_predict = 'impact site x'
saving_folder=f'/Volumes/Jake_ssd/ml_models/CNN_images_and_features_{label_to_predict}/'
# raw_images = get_images_from_dataset(full_dataset_pathname, image_folder)
correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None)
features_to_keep = ['crack len', 'init phi', 'init x']
correlated_featureset = correlated_featureset[features_to_keep]
raw_images = []
run_kfold_Regression_CNN(correlated_featureset, raw_images, full_dataset_labels, patience=50, max_epochs=500, num_outputs=1, lossfunc='mean_squared_error', saving_folder=saving_folder, use_images=False)


# label_to_predict = 'impact site y'
# saving_folder=f'C:\\Users\\u1056\\sfx\\sfx_ML_model_results\\ANN_{label_to_predict}/'
# raw_images = get_images_from_dataset(full_dataset_pathname, image_folder)
# correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None)
# features_to_keep = ['max_kink', 'init y']
# correlated_featureset = correlated_featureset[features_to_keep]
# # raw_images = []
# run_kfold_Regression_CNN(correlated_featureset, raw_images, full_dataset_labels, patience=50, max_epochs=500, num_outputs=1, lossfunc='mean_squared_error', saving_folder=saving_folder, use_images=True)


# label_to_predict = 'height'
# saving_folder=f'C:\\Users\\u1056\\sfx\\sfx_ML_model_results\\ANN_{label_to_predict}/'
# raw_images = get_images_from_dataset(full_dataset_pathname, image_folder)
# correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None)
# features_to_keep = ['crack len', 'abs_val_sum_kink']
# correlated_featureset = correlated_featureset[features_to_keep]
# # raw_images = []
# run_kfold_Regression_CNN(correlated_featureset, raw_images, full_dataset_labels, patience=50, max_epochs=500, num_outputs=1, lossfunc='mean_squared_error', saving_folder=saving_folder, use_images=True)

print('done')
