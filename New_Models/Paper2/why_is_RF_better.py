'''The theory I have is that the RF is more robust to outliers in the data. For example, if there was a test eample that had an outlier value for the max kink angle (90),
then the decision trees in the RF would classify this into its respective group instead of using the value of 90 with a weight to assign its value. In other words, it might just be
classified into the >70 group instead of doing 90*W. So if you had another example with the same features excep tthe max kink angle was 80, then you would get the exact same
prediction. So my theory is that RF's are more robust to outliers, and that is why they perform better than the ANN.

This can be measured (albiet simply) by comparing the MSE and MAE values of predictions. If the MAE is similar for the two models, AND the MSE is very different, 
then there is a discrepancy when it comes to predicting outliers.

RF's are typically poor at extrapolating because of this feature though. ANN's will have a better capability to extrapolate to out-of-domain examples because 
outlier examples will not be 'rounded' back to the value of a training example. 
'''

# from Figures import *
# from ReCalibration import *
# from sklearn.metrics import mean_squared_error, mean_absolute_error

# all_labels = ['height', 'phi', 'theta', 
#                             'impact site x', 'impact site y', 'impact site z', 
#                             'impact site r', 'impact site phi', 'impact site theta']

# model_types = ['Single RF']
# model_types = ['RF_fed_GPR']
# model_types = ['ANN', 'GPR', 'RF', 'ridge', 'Single RF', 'Single GPR', 'NN_fed_GPR', 'NN_fed_RF', 'RF_fed_GPR']
# model_types = ['Single RF', 'ANN']

# # parietal_nodes_folder = ''
# # for fold_no in range(1,2):

# all_preds = {}
# mse = {}
# mae = {}
# fold_no = 1
# for model_type in model_types:
#     print(f'**** {model_type} *****')
#     x_model_folder = f'/Volumes/Jake_ssd/Paper 2/without_transformations/UQ_bagging_models_without_transformations/impact site x/{model_type}'
#     y_model_folder = f'/Volumes/Jake_ssd/Paper 2/without_transformations/UQ_bagging_models_without_transformations/impact site y/{model_type}'

#     #defining folders to get the models and to store the results
#     # model_saving_folder = f'{model_folder}/{label_to_predict}/{model_type}/{num_models}_models/fold_{fold_no}'
#     results_saving_folder = None
    
#     Paper2_path = f'/Volumes/Jake_ssd/Paper 2/without_transformations'
#     #defining folders where the datasets are coming from (5-fold cv)
#     x_test_features_path = Paper2_path + f'/5fold_datasets/impact site x/fold{fold_no}/test_features.csv'
#     x_test_labels_path = Paper2_path + f'/5fold_datasets/impact site x/fold{fold_no}/test_labels.csv'
#     y_test_features_path = Paper2_path + f'/5fold_datasets/impact site y/fold{fold_no}/test_features.csv'
#     y_test_labels_path = Paper2_path + f'/5fold_datasets/impact site y/fold{fold_no}/test_labels.csv'


#     #defining the features that each model used (since they vary with each model)
#     # features_to_keep = ????
#     df = pd.read_csv(x_test_features_path, index_col=0)
#     all_features = df.columns
#     # all_features = all_features.drop(all_labels)
#     features_to_keep = str(all_features.drop('timestep_init').to_list())
    
#     if(model_type in ['ANN', 'RF', 'GPR','ridge']):
#         #predicting the test and train sets with the bagging models
#         test_r2_X, test_ensemble_predictions_X, test_uncertanties_X, test_labels_X = Get_predictions_and_uncertainty_with_bagging(x_test_features_path, x_test_labels_path, x_model_folder + f'/20_models/fold_{fold_no}', results_saving_folder, ast.literal_eval(features_to_keep), 'impact site x', model_type)
#         # test_r2_Y, test_ensemble_predictions_Y, test_uncertanties_Y, test_labels_Y = Get_predictions_and_uncertainty_with_bagging(y_test_features_path, y_test_labels_path, y_model_folder + f'/20_models/fold_{fold_no}', results_saving_folder, ast.literal_eval(features_to_keep), 'impact site y', model_type)
        
#     else:
#         #predicting the test and train sets with the NON bagging models
#         test_r2_X, test_ensemble_predictions_X, test_uncertanties_X, test_labels_X = Get_predictions_and_uncertainty_single_model(x_test_features_path, x_test_labels_path, x_model_folder + f'/1_models/fold_{fold_no}', results_saving_folder, ast.literal_eval(features_to_keep), 'impact site x', model_type)
#         # test_r2_Y, test_ensemble_predictions_Y, test_uncertanties_Y, test_labels_Y = Get_predictions_and_uncertainty_single_model(y_test_features_path, y_test_labels_path, y_model_folder + f'/1_models/fold_{fold_no}', results_saving_folder, ast.literal_eval(features_to_keep), 'impact site y', model_type)

#     mse[model_type] = mean_squared_error(test_labels_X, test_ensemble_predictions_X)
#     mae[model_type] = mean_absolute_error(test_labels_X, test_ensemble_predictions_X)
#     all_preds[model_type] = test_ensemble_predictions_X
# print('here')
#     # random_examples = random.sample(range(51), 30)
#     # r2 = parody_plot_with_std(test_labels_X[random_examples], test_ensemble_predictions_X[random_examples], test_uncertanties_X[random_examples], None, 'impact site x', model_type, testtrain='Test', show=True)
#     # r2 = parody_plot_with_std(test_labels_X[random_examples], test_ensemble_predictions_X[random_examples], np.ones(test_uncertanties_X[random_examples].shape) * np.mean(test_uncertanties_X), None, 'impact site x', model_type, testtrain='Test', show=True)

#     # r2 = parody_plot_with_std(test_labels_X, test_ensemble_predictions_X, test_uncertanties_X, None, 'impact site x', model_type, testtrain='Test', show=True)
#     # r2 = parody_plot_with_std(test_labels_X, test_ensemble_predictions_X, np.ones(test_uncertanties_X.shape) * np.mean(test_uncertanties_X), None, 'impact site x', model_type, testtrain='Test', show=True)


''' The theory that I have is that RF's are much better at capturing nonlinear patterns in our dataset. In the literature, this has been shown in a classification 
problem with figures showing decision boundaries when plotting the two most important features on the x and y axis, and the predicted label is shown as different 
colors. I want to do the same thing, but with my regression problem, and compare the ANN, RF, and others resulting plots.

the plots in the literture come from: https://arxiv.org/abs/2207.08815
'''
from Figures import *
from ReCalibration import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D


def make_3d_surface_plot(predictions, important_feature_1, important_feature_2, label, feature_1_name, feature_2_name, model_type):
    """
    Creates a 3D surface plot of the predictions over the important features.

    :param predictions: numpy array of the predictions
    :param important_feature_1: numpy array of important feature 1
    :param important_feature_2: numpy array of important feature 2
    :param label: Title for the plot
    :param feature_1_name: Label for the x-axis (important feature 1)
    :param feature_2_name: Label for the y-axis (important feature 2)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of the data points
    scatter = ax.scatter(important_feature_1, important_feature_2, predictions, c=predictions, cmap='viridis')
    
    # Create the surface plot
    # Create a grid to evaluate the model
    grid_x, grid_y = np.meshgrid(np.linspace(important_feature_1.min(), important_feature_1.max(), 100),
                                 np.linspace(important_feature_2.min(), important_feature_2.max(), 100))
    
    # Flatten the grid to make predictions
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    
    # If your model is not already trained, you need to fit it here, e.g.,
    # model = YourModelType() # Example: RandomForestRegressor()
    # model.fit(np.c_[important_feature_1, important_feature_2], predictions)
    
    # Replace this with the actual prediction call of your trained model
    # grid_predictions = model_type.predict(grid_points).reshape(grid_x.shape)
    
    # Plot the surface
    ax.plot_surface(grid_x, grid_y, predictions, alpha=0.3, rstride=100, cstride=100, color='b', edgecolor='none')
    
    ax.set_title(label)
    ax.set_xlabel(feature_1_name)
    ax.set_ylabel(feature_2_name)
    ax.set_zlabel('Predictions')
    
    fig.colorbar(scatter, ax=ax, label='Prediction Intensity')
    plt.show()


# def make_heatmap_plots(predictions, important_feature_1, important_feature_2, label, feature_1_name, feature_2_name):
#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(important_feature_1, important_feature_2, c=predictions, cmap='viridis', alpha=0.7)
#     plt.colorbar(scatter, label='Predictions')
#     plt.xlabel(feature_1_name)
#     plt.ylabel(feature_2_name)
#     plt.title(label)
#     plt.show()

def make_heatmap_plots(predictions, important_feature_1, important_feature_2, label, feature_1_name, feature_2_name, model_type):
    """
    Creates a scatter plot where x is important_feature_1, y is important_feature_2,
    and the color of the scatter points is determined by the value of the predictions array.
    Additionally, includes a background heatmap to show prediction intensity.
    
    :param predictions: numpy array of the predictions
    :param important_feature_1: numpy array of important feature 1
    :param important_feature_2: numpy array of important feature 2
    :param label: Title for the plot
    :param feature_1_name: Label for the x-axis (important feature 1)
    :param feature_2_name: Label for the y-axis (important feature 2)
    """
    # Create grid data for background heatmap
    xi = np.linspace(important_feature_1.min(), important_feature_1.max(), 1000)
    yi = np.linspace(important_feature_2.min(), important_feature_2.max(), 1000)
    zi = griddata((important_feature_1, important_feature_2), predictions, (xi[None, :], yi[:, None]), method='cubic')
    
    plt.figure(figsize=(10, 6))
    
    # Plot background heatmap
    plt.contourf(xi, yi, zi, levels=100, cmap='viridis', alpha=0.6)
    
    if(label == 'impact site x'): 
        min_colorbar = -50; max_colorbar=50
    elif(label == 'impact site y'):
        min_colorbar = -40; max_colorbar=40
    elif(label == 'height'):
        min_colorbar = 1; max_colorbar=5
    # Plot scatter points
    scatter = plt.scatter(important_feature_1, important_feature_2, c=predictions, cmap='viridis', edgecolor='k', vmin=min_colorbar, vmax=max_colorbar)
    plt.colorbar(scatter, label='Predictions')
    
    plt.xlabel(feature_1_name)
    plt.ylabel(feature_2_name)
    plt.title(label + ' predicting ' + model_type)
    plt.savefig(f'/Users/jakehirst/Desktop/sfx/figures/decision_boundary_heatmap/{label}_{model_type}_decision_boundary_heatmap.png')
    plt.close()
    # plt.show()

# def make_heatmap_plots(predictions, important_feature_1, important_feature_2, label, feature_1_name, feature_2_name):
#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(important_feature_1, important_feature_2, c=predictions, cmap='viridis', alpha=0.6, edgecolor='w', linewidth=0.5)
    
#     # Adding color bar to represent the predictions
#     cbar = plt.colorbar(scatter)
#     cbar.set_label('Predictions', rotation=270, labelpad=15)
    
#     plt.title(label)
#     plt.xlabel(feature_1_name)
#     plt.ylabel(feature_2_name)
    
#     # Adding grid for better readability
#     plt.grid(True, linestyle='--', alpha=0.6)
    
#     plt.show()
# from sklearn.metrics import mean_squared_error, mean_absolute_error

all_labels = ['height', 'phi', 'theta', 
                            'impact site x', 'impact site y', 'impact site z', 
                            'impact site r', 'impact site phi', 'impact site theta']

model_types = ['Single RF']
model_types = ['RF_fed_GPR']
model_types = ['ANN', 'GPR', 'RF', 'ridge', 'Single RF', 'Single GPR', 'NN_fed_GPR', 'NN_fed_RF', 'RF_fed_GPR']
model_types = ['Single RF', 'ANN']
# model_types = ['ridge']

labels = ['height', 'impact site x', 'impact site y']
# labels = ['height']

# parietal_nodes_folder = ''
# for fold_no in range(1,2):

all_preds = {}
mse = {}
mae = {}
fold_no = 1
for label in labels:
    for model_type in model_types:
        print(f'**** {model_type} *****')
        model_folder = f'/Volumes/Jake_ssd/Paper 2/without_transformations/UQ_bagging_models_without_transformations/{label}/{model_type}'

        #defining folders to get the models and to store the results
        # model_saving_folder = f'{model_folder}/{label_to_predict}/{model_type}/{num_models}_models/fold_{fold_no}'
        results_saving_folder = None
        
        Paper2_path = f'/Volumes/Jake_ssd/Paper 2/without_transformations'
        
        #defining folders where the datasets are coming from (5-fold cv)
        test_features_path = Paper2_path + f'/5fold_datasets/{label}/fold{fold_no}/test_features.csv'
        test_labels_path = Paper2_path + f'/5fold_datasets/{label}/fold{fold_no}/test_labels.csv'

        #defining the features that each model used (since they vary with each model)
        # features_to_keep = ????
        df = pd.read_csv(test_features_path, index_col=0)
        all_features = df.columns
        # all_features = all_features.drop(all_labels)
        features_to_keep = str(all_features.drop('timestep_init').to_list())
        
        if(model_type in ['ANN', 'RF', 'GPR','ridge']):
            #predicting the test and train sets with the bagging models
            test_r2, test_ensemble_predictions, test_uncertanties, test_labels = Get_predictions_and_uncertainty_with_bagging(test_features_path, test_labels_path, model_folder + f'/20_models/fold_{fold_no}', results_saving_folder, ast.literal_eval(features_to_keep), label, model_type)            
        else:
            #predicting the test and train sets with the NON bagging models
            test_r2, test_ensemble_predictions, test_uncertanties, test_labels = Get_predictions_and_uncertainty_single_model(test_features_path, test_labels_path, model_folder + f'/1_models/fold_{fold_no}', results_saving_folder, ast.literal_eval(features_to_keep), label, model_type)

        mse[model_type] = mean_squared_error(test_labels, test_ensemble_predictions)
        mae[model_type] = mean_absolute_error(test_labels, test_ensemble_predictions)
        all_preds[model_type] = test_ensemble_predictions
        
        # parody_plot_with_std(test_labels, test_ensemble_predictions, test_uncertanties, None, label, model_type, testtrain='Test', show=True)

        if(label == 'impact site x'):
            #ranking of features = 
            #['init x', 'thickness_at_init', 'linearity', 'mean thickness',
            # 'max thickness', 'init z', 'init y', 'median_thickness',
            # 'avg_prop_speed', 'angle_btw', 'sum_kink', 'avg_ori', 'max_prop_speed', 'var_thickness', 'std_thickness', 'mean_kink', 'max_kink',
            # 'abs_val_mean_kink', 'crack len', 'dist btw frts', 'median_kink', 'abs_val_sum_kink', 'std_kink', 'var_kink']
            feature_1_name = 'init x'
            feature_2_name = 'init y'
        elif(label == 'impact site y'):
            #ranking of features = 
            #['init y', 'avg_prop_speed', 'thickness_at_init', 'avg_ori', 'max_prop_speed', 'init z', 'init x', 
            # 'crack len', 'dist btw frts', 'linearity', 'mean_kink', 'angle_btw', 'median_thickness', 'mean thickness', 
            # 'max thickness', 'abs_val_mean_kink', 'std_thickness', 'var_thickness', 'sum_kink', 'max_kink', 'abs_val_sum_kink', 
            # 'std_kink', 'var_kink', 'median_kink']
            feature_1_name = 'init y'
            feature_2_name = 'avg_prop_speed'
        else:
            #ranking of features importances  = 
            #['abs_val_sum_kink', 'dist btw frts', 'crack len', 'max_prop_speed',
            # 'var_kink', 'std_kink', 'angle_btw', 'linearity', 'max_kink', 'mean_kink', 'max thickness', 'median_kink', 'median_thickness',
            # 'abs_val_mean_kink', 'mean thickness', 'std_thickness', 'var_thickness', 'init z', 'avg_ori', 'thickness_at_init', 'avg_prop_speed', 'sum_kink',
            # 'init x', 'init y']
            feature_1_name = 'abs_val_sum_kink'
            feature_2_name = 'dist btw frts'
        
        make_heatmap_plots(test_labels, df[feature_1_name].to_numpy(), df[feature_2_name].to_numpy(), label, feature_1_name, feature_2_name, 'TRUE LABEL VALUES')
        make_heatmap_plots(test_ensemble_predictions, df[feature_1_name].to_numpy(), df[feature_2_name].to_numpy(), label, feature_1_name, feature_2_name, model_type)

        # make_3d_surface_plot(test_labels, df[feature_1_name].to_numpy(), df[feature_2_name].to_numpy(), label, feature_1_name, feature_2_name, 'TRUE LABEL VALUES')
        # make_3d_surface_plot(test_ensemble_predictions, df[feature_1_name].to_numpy(), df[feature_2_name].to_numpy(), label, feature_1_name, feature_2_name, model_type)

        print('here')
        # random_examples = random.sample(range(51), 30)
        # r2 = parody_plot_with_std(test_labels_X[random_examples], test_ensemble_predictions_X[random_examples], test_uncertanties_X[random_examples], None, 'impact site x', model_type, testtrain='Test', show=True)
        # r2 = parody_plot_with_std(test_labels_X[random_examples], test_ensemble_predictions_X[random_examples], np.ones(test_uncertanties_X[random_examples].shape) * np.mean(test_uncertanties_X), None, 'impact site x', model_type, testtrain='Test', show=True)

        # r2 = parody_plot_with_std(test_labels_X, test_ensemble_predictions_X, test_uncertanties_X, None, 'impact site x', model_type, testtrain='Test', show=True)
        # r2 = parody_plot_with_std(test_labels_X, test_ensemble_predictions_X, np.ones(test_uncertanties_X.shape) * np.mean(test_uncertanties_X), None, 'impact site x', model_type, testtrain='Test', show=True)

'''The other thing that they did in the paper is they iteratively got rid of unimportant features and saw how the models performed. 
The RF seemed to be more robust to uninformative features. I can recreate this by adding random values as features in the dataset, and 
plotting the performance of the RF and ANN as we add these values. The RF should be able to completely ignore them, and the ANN should degrade in performance. '''