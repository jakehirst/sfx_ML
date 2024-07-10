'''The theory I have is that the RF is more robust to outliers in the data. For example, if there was a test eample that had an outlier value for the max kink angle (90),
then the decision trees in the RF would classify this into its respective group instead of using the value of 90 with a weight to assign its value. In other words, it might just be
classified into the >70 group instead of doing 90*W. So if you had another example with the same features excep tthe max kink angle was 80, then you would get the exact same
prediction. So my theory is that RF's are more robust to outliers, and that is why they perform better than the ANN.

This can be measured (albiet simply) by comparing the MSE and MAE values of predictions. If the MAE is similar for the two models, AND the MSE is very different, 
then there is a discrepancy when it comes to predicting outliers.

RF's are typically poor at extrapolating because of this feature though. ANN's will have a better capability to extrapolate to out-of-domain examples because 
outlier examples will not be 'rounded' back to the value of a training example. 
'''


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
from Single_UQ_models import *


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

def compare_partial_dependence_plots(labels, model_types):
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

    return

#smooths y using a gaussian kernel, and is adjusted by the lengthscale. larger lengthscale = more smoothing
def gaussian_kernel_smoothing(X, y, lengthscale):
    from scipy.spatial.distance import cdist
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.covariance import MinCovDet

    # Compute the Gaussian kernel        
    '''quote from the paper section A.4.1 : https://arxiv.org/pdf/2207.08815
    covariance matrix (sigma) is the data covariance, multiplied by the (squared) lengthscale of the Gaussian kernel smoother.
    We estimate the covariance matrix of these features through ScikitLearn’s MinCovDet, which is more robust to outliers than the empirical covariance.'''
    mcd = MinCovDet().fit(X)
    sigma = mcd.covariance_  * (lengthscale ** 2)#covariance matrix 
    inverse_sigma = np.linalg.inv(sigma)
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html mahalanobis is perfect for this equation...
    mahalanobis = cdist(X,X, 'mahalanobis', VI=inverse_sigma) #(x^* − x)^T * sigma^−1 * (x^* − x)
    K_est = np.exp( -1/2 * mahalanobis) #K = e^(-1/2 * (x^* − x)^T * sigma^−1 * (x^* − x) ** 2)
    
    #use kernel to calculate smoothed y
    y_smoothed = K_est.dot(y) # SUM_j=1_to_N(K(X_i, X_j) * Y(X_j)) ***numerator***
    y_smoothed = y_smoothed / K_est.sum(axis=1) # numerator / SUM_j=1_to_N(K(X_i, X_j))
    
    return y_smoothed

# Function to generate synthetic data to make sure my smoothing function works...
def generate_noisy_sine_data(num_samples, num_features):
    np.random.seed(42)
    X = np.random.rand(num_samples, num_features)
    noise = np.random.normal(0, 0.1, num_samples)
    y = np.sin(2 * np.pi * X[:, 0]) + noise  #Using the first feature for sine function
    return X, y

def make_side_by_side_scatter_plots(features, new_y_train, train_labels, feature_name, label_name):
    x = features[feature_name]
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # First scatter plot
    axes[0].scatter(x, new_y_train, c='r', alpha=0.5)
    axes[0].set_title(f'Smoothed {label_name}')
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel(label_name)

    # Second scatter plot
    axes[1].scatter(x, train_labels, c='b', alpha=0.5)
    axes[1].set_title(f'Original {label_name}')
    axes[1].set_xlabel(feature_name)

    # Set the same axis ranges for both plots
    all_y = np.concatenate([new_y_train, train_labels])
    axes[0].set_ylim([min(all_y)-1, max(all_y)+1])
    axes[1].set_ylim([min(all_y)-1, max(all_y)+1])
    axes[0].set_xlim([min(x)-1, max(x)+1])
    axes[1].set_xlim([min(x)-1, max(x)+1])

    # Show the plot
    plt.tight_layout()
    plt.show()
    return 

'''
Smooths the training label data at 4 different levels of smoothing, and saves it in a folder in smoothing_labels_folder
'''
def make_smooth_outputs_of_training_set(labels, smoothing_labels_folder):
    Paper2_path = f'/Volumes/Jake_ssd/Paper 2/without_transformations'
    '''repeat the following for all 5 folds'''
    for label in labels:
        for fold_no in range(1,6):
            dir = f'{smoothing_labels_folder}/data_{label}/fold_{fold_no}'
            if(not os.path.exists(dir)): os.makedirs(dir)
            
            '''Train models on the original dataset without any smoothing.'''
            test_features_path = Paper2_path + f'/5fold_datasets/{label}/fold{fold_no}/test_features.csv'
            test_labels_path = Paper2_path + f'/5fold_datasets/{label}/fold{fold_no}/test_labels.csv'
            train_features_path = Paper2_path + f'/5fold_datasets/{label}/fold{fold_no}/train_features.csv'
            train_labels_path = Paper2_path + f'/5fold_datasets/{label}/fold{fold_no}/train_labels.csv'
            
            train_labels = pd.read_csv(train_labels_path)
            train_features = pd.read_csv(train_features_path, index_col=0)
            test_labels = pd.read_csv(test_labels_path)
            test_features = pd.read_csv(test_features_path, index_col=0)
            
            #saving everything into these folders for convenience later...
            train_labels.to_csv(f'{dir}/smoothed_train_labels_L_0.csv')
            train_features.to_csv(f'{dir}/train_feats.csv')
            test_features.to_csv(f'{dir}/test_feats.csv')
            test_labels.to_csv(f'{dir}/test_labels.csv')
            
            for lengthscale in [0.5, 1, 1.5, 2]:
                new_y_train = gaussian_kernel_smoothing(train_features.to_numpy(), train_labels.to_numpy().flatten(), 1) 
                
                '''show that it works on fake data'''
                # noisy_data_x, noisy_data_y = generate_noisy_sine_data(100, 1)
                # example_new_y_train = gaussian_kernel_smoothing(noisy_data_x, noisy_data_y, .05)
                # make_side_by_side_scatter_plots(pd.DataFrame(noisy_data_x,columns = ['col1']), 
                #                                             example_new_y_train, noisy_data_y, 'col1', 'output of sine function')

                '''show that it works on the real data'''
                if(label == 'impact site x'): feat = 'init x'
                elif(label == 'impact site y'): feat = 'init y'
                else: feat = 'crack len'
                # make_side_by_side_scatter_plots(train_features, new_y_train, train_labels.to_numpy().flatten(), feat, label)
                
                y_df = pd.DataFrame(new_y_train, columns = [label])
                y_df.to_csv(f'{dir}/smoothed_train_labels_L_{lengthscale}.csv')
            
    return


def train_models_on_smoothed_labels(model_type, labels, smoothing_labels_folder):
    
    for label in labels:
        for fold_no in range(1,6):
            performances_dict = {'lengthscale': [],
                                 'r2': []}
            dir = f'{smoothing_labels_folder}/data_{label}/fold_{fold_no}'
            train_features = pd.read_csv(f'{dir}/train_feats.csv', index_col=0)
            for lengthscale in [0, 0.5, 1, 1.5, 2]:
                print(f'***** fold {fold_no} training {model_type} on {label} with lengthscale = {lengthscale} *****')
                model_folder = f'{smoothing_labels_folder}/trained_models'
                results_folder = f'{smoothing_labels_folder}/performances'
                #train model
                test_features_path = f'{smoothing_labels_folder}/data_{label}/fold_{fold_no}/test_feats.csv'
                test_labels_path = f'{smoothing_labels_folder}/data_{label}/fold_{fold_no}/test_labels.csv'
                train_labels_path = f'{smoothing_labels_folder}/data_{label}/fold_{fold_no}/smoothed_train_labels_L_{lengthscale}.csv'
                train_features_path = f'{smoothing_labels_folder}/data_{label}/fold_{fold_no}/train_feats.csv'
                
                train_feats = pd.read_csv(train_features_path, index_col=0)
                train_feats = train_feats.drop('timestep_init', axis=1) 
                train_labels = pd.read_csv(train_labels_path, index_col=0)
                with_or_without_transformations = 'without'
                hyperparam_folder = f'/Volumes/Jake_ssd/Paper 2/{with_or_without_transformations}_transformations' + f'/bayesian_optimization_{with_or_without_transformations}_transformations'

                #predicting the test and train sets with the bagging models
                if(['ANN', 'ridge', 'RF', 'GPR'].__contains__(model_type)):
                    #defining folders to get the models and to store the results
                    model_saving_folder = f'{model_folder}/{label}/{model_type}/fold_{fold_no}'
                    results_saving_folder = f'{results_folder}/{label}/{model_type}/fold_{fold_no}'
                    #train bagging model

                    make_linear_regression_models_for_ensemble(train_feats.reset_index(), 
                                                               train_labels, 
                                                               model_saving_folder, 
                                                               label, 
                                                               20, 
                                                               train_feats.columns, 
                                                               hyperparam_folder, 
                                                               model_type=model_type)
                    test_r2, test_ensemble_predictions, test_uncertanties, test_labels = Get_predictions_and_uncertainty_with_bagging(test_features_path, test_labels_path, model_saving_folder, results_saving_folder, train_feats.columns, label, model_type)
                else:
                    #defining folders to get the models and to store the results
                    model_saving_folder = f'{model_folder}/{label}/{model_type}/fold_{fold_no}'
                    results_saving_folder = f'{results_folder}/{label}/{model_type}/fold_{fold_no}'
                    #train single model
                    make_UQ_model(train_feats, 
                                  train_labels, 
                                  model_saving_folder, 
                                  label, 
                                  1, 
                                  train_feats.columns,
                                  hyperparam_folder, 
                                  model_type=model_type)
                    test_r2, test_ensemble_predictions, test_uncertanties, test_labels = Get_predictions_and_uncertainty_single_model(test_features_path, test_labels_path, model_saving_folder, results_saving_folder, train_feats.columns, label, model_type)

                performances_dict['lengthscale'].append(lengthscale)
                performances_dict['r2'].append(test_r2)
            df = pd.DataFrame.from_dict(performances_dict)
            df.to_csv(f'{smoothing_labels_folder}/performances/{label}/{model_type}/fold_{fold_no}/performances_for_all_lengthscales_{model_type}_fold_{fold_no}.csv')
    
    
    return

'''plots histograms of the performances across folds for each lengthscale and each model type.'''
# def make_performances_graphic(model_types, label, performances_folder): 
#     label_folder = performances_folder + f'/{label}'
#     for model_type in model_types:
#         performances_dict = {0.0:[], 0.5:[], 1.0:[], 1.5:[], 2.0:[]}
#         for fold_no in range(1,6):
#             performances = pd.read_csv(performances_folder + f'/{label}/{model_type}/fold_{fold_no}/performances_for_all_lengthscales_{model_type}_fold_{fold_no}.csv', index_col=0)
#             for row_num, content in performances.iterrows():
#                 performances_dict[content['lengthscale']].append(content['r2'])
#             print('here')
    
    
#     return

def make_performances_graphic(model_types, label, performances_folder): 
    performances_data = []
    label_folder = performances_folder + f'/{label}'
    
    for model_type in model_types:
        if(model_type == 'RF_fed_GPR'): legend_model_type = 'RF-fed GPR'
        elif(model_type == 'NN_fed_GPR'): legend_model_type = 'NN-fed GPR'
        elif(model_type == 'NN_fed_RF'): legend_model_type = 'NN-fed RF'
        else: legend_model_type = model_type

        performances_dict = {0.0:[], 0.5:[], 1.0:[], 1.5:[], 2.0:[]}
        
        for fold_no in range(1, 6):
            performances = pd.read_csv(performances_folder + f'/{label}/{model_type}/fold_{fold_no}/performances_for_all_lengthscales_{model_type}_fold_{fold_no}.csv', index_col=0)
            
            for row_num, content in performances.iterrows():
                performances_dict[content['lengthscale']].append(content['r2'])
        
        # Prepare the data for plotting
        for lengthscale, r2_values in performances_dict.items():
            for r2 in r2_values:
                performances_data.append({'Model Type': legend_model_type, 'Lengthscale': lengthscale, 'R2': r2})

    # Convert the data to a DataFrame
    df_performances = pd.DataFrame(performances_data)
    
    color_palette = {
        'ANN': 'tab:blue',
        'GPR': 'tab:orange',
        'RF': 'tab:green',
        'ridge': 'tab:red',
        'Single RF': 'tab:purple',
        'Single GPR': 'tab:brown',
        'NN-fed GPR': 'tab:pink',
        'NN-fed RF': 'tab:gray',
        'RF-fed GPR': 'tab:olive'
    }
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    

    # Plot the background shading
    for i in range(0, len(performances_dict), 2):
        plt.axvspan(i-0.5, i+0.5, color='lightgrey', alpha=0.5, zorder=0)
    
    sns.boxplot(x='Lengthscale', y='R2', hue='Model Type', data=df_performances, palette=color_palette, zorder=1)

    axis_size = 18
    # if(label == 'height'): plt.ylim((0,0.4))
    # elif(label == 'impact site x'): plt.ylim((0.2,0.9))
    # elif(label == 'impact site y'): plt.ylim((0.1,0.9))
    plt.ylim((0,1))
    # Add some styling to make it look nice
    plt.title(f'{label.capitalize()}', fontsize=axis_size+5, weight='bold')
    plt.xlabel('Lengthscale of the Gaussian Kernel Smoother', fontsize=axis_size, weight='bold')
    plt.ylabel('$R^2$', fontsize=axis_size+3, weight='bold')
    plt.legend(fontsize=axis_size-5)
    
    # plt.show()
    plt.savefig(f'/Volumes/Jake_ssd/Smoothing_labels/figures/{label}_smoothed_labels_figure.png')
    plt.close()



def main():
    all_labels = ['height', 'phi', 'theta', 
                            'impact site x', 'impact site y', 'impact site z', 
                            'impact site r', 'impact site phi', 'impact site theta']


    model_types = ['ANN', 'GPR', 'RF', 'ridge', 'Single RF', 'Single GPR', 'NN_fed_GPR', 'NN_fed_RF', 'RF_fed_GPR']
    # model_types = ['GPR', 'RF', 'NN_fed_GPR', 'NN_fed_RF']
    # model_types = ['NN_fed_GPR', 'NN_fed_RF']
    # model_types = ['GPR', 'Single RF', 'ANN', 'RF_fed_GPR']
    # model_types = ['NN_fed_RF']
    labels = ['impact site x', 'impact site y', 'height',]
    # labels = ['height']

    # parietal_nodes_folder = ''
    # for fold_no in range(1,2):
    
    '''Trying partial dependence plots'''
    # compare_partial_dependence_plots(labels, model_types)
    
    '''Smoothing outputs of training set.'''
    
    '''Evaluate and record their performance metrics (R² ) on the test sets.'''
    '''Define a range of lengthscales for the Gaussian kernel smoother (e.g., small, medium, large).'''
    '''For each lengthscale
            Compute the smoothed target variable using the Gaussian kernel
            Retrain both the RF and ANN models using the smoothed target variable for each lengthscale.
            Evaluate and record their performance on the original (unsmoothed) test set to ensure comparability.'''
    smoothing_labels_folder = '/Volumes/Jake_ssd/Smoothing_labels'
    # make_smooth_outputs_of_training_set(labels, smoothing_labels_folder)
    
    '''train the models and record performances'''
    # for model_type in model_types:
    #     train_models_on_smoothed_labels(model_type, labels, smoothing_labels_folder)
    
    '''plot the performances'''
    for label in labels:
        make_performances_graphic(model_types, label, smoothing_labels_folder + '/performances')

if __name__ == "__main__":
    main()
