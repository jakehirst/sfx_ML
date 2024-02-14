import os
import pandas as pd
import numpy as np
import random
import re
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize
from prepare_data import *
from sklearn.model_selection import train_test_split
from linear_regression import *
from lasso_regression import *
from ridge_regression import *
from polynomial_regression import *
from GPR import *
from CNN import *
from mastml.plots import *
from mastml.models import *
from mastml.error_analysis import *
from Backward_feature_selection import *

# full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_data/New_Crack_Len_FULL_OG_dataframe_2023_10_14.csv"
# image_folder = '/Users/jakehirst/Desktop/sfx/sfx_ML_data/images_sfx/new_dataset/Visible_cracks'
# full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_data/New_Crack_Len_FULL_OG_dataframe_2023_10_04.csv"

'''
makes a parody plot of the predictions from uncertainty model including the standard deviations
'''
def parody_plot_with_std(y_test, y_pred_test, y_pred_test_std, saving_folder, label_to_predict, model_type, testtrain='unknown_test_train'):
    # y_pred_std_test_times_2 = list(np.array(y_pred_test_std)*2)
    plt.figure(figsize=(12, 6))
    # plt.errorbar(y_test, y_pred_test, yerr=y_pred_std_test_times_2, fmt='o')
    plt.errorbar(y_test, y_pred_test, yerr=y_pred_test_std, fmt='o')
    plt.plot(y_test, y_test, c='r')
    plt.title(f'{testtrain} set {model_type} regression ensemble predicting '+f'{label_to_predict}' + ', R2=%.2f' % r2_score(y_test, y_pred_test))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(saving_folder +  f'/ensemble_UQ_parody_plot_{testtrain}_set.png')
    # plt.show()
    plt.close()
    return r2_score(y_test, y_pred_test)

''' specifically saves models for ensembling '''
def save_ensemble_model(model, fold_no, saving_folder):
    # Save the model to a file
    filename = saving_folder + f'/model_no{fold_no}.sav'
    pickle.dump(model, open(filename, 'wb'))

'''loads a single model from an ensemble to be used.'''
def load_ensemble_model(model_path):
    model = pickle.load(open(model_path, 'rb'))
    return model
    
''' 
makes num_models number of regression models and saves them into the saving_folder for later bagging ensembling. 
The training sets for each of these models are taken from the training_features and training_labels with replacement. They are the same len as the training_features
'''  
def make_linear_regression_models_for_ensemble(training_features, training_labels, model_saving_folder, label_to_predict, num_models, features_to_keep, hyperparam_folder, num_training_points=False, model_type=None): 
    models = []
    training_features = training_features[features_to_keep]
    if(not os.path.exists(model_saving_folder)): os.mkdir(model_saving_folder)
    # training_features['crack len'] = np.arange(0, 195)
    # training_labels['impact site x'] = np.arange(0, 195)

    num_samples = len(training_features)
    
    #now to train all the models and save them
    for model_num in range(num_models):
        print(f'working on model {model_num}')
        # getting a subset of the training dataset with replacement to train this model on
        sampled_index = training_labels.sample(n=num_samples, replace=True).index
        new_train_features = training_features.loc[sampled_index]
        new_train_labels = training_labels.loc[sampled_index]
        
        model = train_model(model_type, new_train_features, new_train_labels, hyperparam_folder)
        
        save_ensemble_model(model, model_num, model_saving_folder) 
  
    return

''' 
Does a 80/20 split on the dataset and saves the dataset features and labels that 
we will be using in our ensembling training sets in the data_saving_folder. 
Also keeps the test sets.
'''
def Level_1_test_train_split(full_dataset_pathname, image_folder, all_labels, label_to_predict, data_saving_folder):
    if(not os.path.exists(data_saving_folder)): os.mkdir(data_saving_folder)
    full_dataset_features, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=1)
    train_features, test_features, train_labels, test_labels = train_test_split(full_dataset_features, full_dataset_labels, test_size=0.2)
    
    # Reset index for the training and test dataframes
    train_features = train_features.reset_index(drop=True)
    train_labels = pd.DataFrame(train_labels, columns=['impact site x']).reset_index(drop=True)
    test_features = test_features.reset_index(drop=True)
    test_labels = pd.DataFrame(test_labels, columns=['impact site x']).reset_index(drop=True)
    if(not os.path.exists(data_saving_folder + '/data')): os.mkdir(data_saving_folder + '/data')
    train_features.to_csv(data_saving_folder + '/data/train_features.csv')
    train_labels.to_csv(data_saving_folder + '/data/train_labels.csv')
    test_features.to_csv(data_saving_folder + '/data/test_features.csv')
    test_labels.to_csv(data_saving_folder + '/data/test_labels.csv')

    return

'''
predicts the labels of the featureset given, and the uncertainty.
uncertainty is based on the variance of the predictions from the ensemble.
Also compares this to the true labels of the dataset.
'''
def Get_predictions_and_uncertainty_with_bagging(test_features_path, test_labels_path, model_folder, saving_folder, features_to_keep, label_to_predict, model_type):
    if(not os.path.exists(saving_folder)): os.makedirs(saving_folder)
    test_features = pd.read_csv(test_features_path)[features_to_keep]
    test_labels = pd.read_csv(test_labels_path) 
    
    # Initialize an empty list to store the loaded models
    models = []

    # Loop through the pickle files in the folder
    for filename in os.listdir(model_folder):
        if filename.endswith('.sav'):
            # Load the model from the pickle file
            with open(os.path.join(model_folder, filename), 'rb') as file:
                model = pickle.load(file)
                models.append(model)
    
    all_model_predictions = []
    for model in models:
        if(type(model) == ANNModel): 
            with torch.no_grad():
                model.eval()
                current_predictions = model(torch.FloatTensor(test_features.values).to(device)).numpy()
        else:
            current_predictions = model.predict(test_features.to_numpy())
            
        # current_predictions = model.predict(test_features.to_numpy())
        current_predictions = current_predictions.reshape(current_predictions.shape[0])
        all_model_predictions.append(current_predictions)

    all_model_predictions = np.array(all_model_predictions)
    
    ensemble_predictions = []
    ensemble_uncertanties = []
    for label_no in range(len(test_labels)):
        true_label = test_labels.iloc[label_no][0]
        mean_prediction, std_prediction = np.mean(all_model_predictions[:, label_no]), np.std(all_model_predictions[:, label_no])
        ensemble_predictions.append(mean_prediction)
        ensemble_uncertanties.append(std_prediction*2) #uncertainty will be 2 * the std of the predictions
    test_or_train = test_features_path.split('_')[-2].split('/')[-1]
    r2 = parody_plot_with_std(test_labels.to_numpy(), ensemble_predictions, ensemble_uncertanties, saving_folder, label_to_predict, model_type, testtrain=test_or_train)
    
    return r2, ensemble_predictions, ensemble_uncertanties, test_labels


''' 
Calculates a and b in the equation sigma = a * sigma_uc + b where sigma_uc is the uncalibrated uncertainties, and sigma is the calibrated uncertainties.
The calibration is calculated by minimizing the sum of the negative log likelihoods. 
'''
def get_calibration_factors(residuals, uncertainties):
    # Define the objective function
    '''linear'''
    # def objective(params, R, sigma_uc):
    #     a, b = params
    #     loss = np.sum(np.log(2*np.pi) + np.log(a*sigma_uc + b)**2 + (R**2 / (a*sigma_uc + b)**2))
    #     return .5 * loss / len(R)
    '''non-linear'''
    def objective(params, R, sigma_uc):
        a, b = params
        loss = np.sum(np.log(2*np.pi) + np.log(a*(sigma_uc**((b/2) + 1)))**2 + (R**2 / (a*(sigma_uc**((b/2) + 1)))**2))
        return .5 * loss / len(R)
    
    # Provide initial guesses for a and b
    initial_guess = [1.0, 1.0]  # You may want to adjust these based on your problem
    
    ARE = residuals
    SIGUC = uncertainties
    # Define a callback function to print out the value of the objective function
    def callback_func(xk):
        print(f'Current value of objective function: {objective(xk, ARE, SIGUC)}')

    # Call the optimizer
    # result = minimize(objective, initial_guess, args=(residuals, uncertainties), method='nelder-mead', callback=callback_func)
    result = minimize(objective, initial_guess, args=(residuals, uncertainties), method='nelder-mead')

    # Extract optimized values
    a_opt, b_opt = result.x
    return a_opt, b_opt

'''
creates plots similar to in the following paper to quickly see how accurate the UQ is
https://www.nature.com/articles/s41524-022-00794-8

1.) organize the uncertainties into bins that are normalized from 0 to 1
2.) for each bin, get the RMS of the residuals in that bin
3.) plot each bin's normalized uncertainty vs the normalized RMS 

'''
def make_RVE_plots(label_to_predict, model_type, test_predictions, test_uncertanties, test_true_labels, train_predictions, train_uncertanties, train_true_labels, saving_folder, num_bins=10):
    def normalize_array(arr, min_value, max_value):
        # Initialize an empty list to store normalized values
        normalized_arr = []
        # Iterate through the array and normalize each value
        for num in arr:
            # normalized_value = (num - min_value) / (max_value - min_value)
            normalized_value = num / max_value
            normalized_arr.append(normalized_value)
        return normalized_arr
    
    def bin_uncertainties_and_residuals(normalized_uncertainties, normalized_residuals, num_bins):
        # Calculate the bin edges based on percentiles
        bin_edges = np.percentile(normalized_uncertainties, np.linspace(0, 100, num_bins + 1))
        # Use numpy.histogram to bin the uncertainties
        hist, _ = np.histogram(normalized_uncertainties, bins=bin_edges)

        # Calculate the average uncertainty for each bin
        average_uncertainties = []
        RMS_residuals = []
        for i in range(num_bins):
            bin_indices = np.where((normalized_uncertainties >= bin_edges[i]) & (normalized_uncertainties <= bin_edges[i + 1]))
            bin_uncertainties = [normalized_uncertainties[i] for i in bin_indices[0]] 
            bin_residuals = [normalized_residuals[i] for i in bin_indices[0]] 
            if len(bin_uncertainties) > 0:
                average_uncertainty = np.mean(bin_uncertainties)
                RMS_residual = np.sqrt(np.mean(np.array(bin_residuals) ** 2))
                average_uncertainties.append(average_uncertainty)
                RMS_residuals.append(RMS_residual)
            else:
                average_uncertainties.append(0.0)


        average_uncertainties_col = np.array(average_uncertainties).reshape(-1, 1)

        # Create a linear regression model
        model = LinearRegression()
        # Fit the model to the data
        model.fit(average_uncertainties_col, RMS_residuals)
        # Predict the y-values based on the fitted line
        y_pred = model.predict(average_uncertainties_col)
        # Get the slope (coefficient) of the fitted line and round to two decimal places
        slope = round(model.coef_[0], 2)
        # Get the intercept of the fitted line and round to two decimal places
        intercept = round(model.intercept_, 2)
        


        # hist contains the count of uncertainties in each bin
        # print("Histogram:", hist)
        # print("Average bin Uncertainties:", average_uncertainties)
        # print("average bin residuals: ", RMS_residuals)
    
        return average_uncertainties, RMS_residuals, slope, intercept, average_uncertainties, y_pred
    
    test_true_labels = test_true_labels._values.T.tolist()[0]
    test_residuals = np.abs(np.array(test_true_labels) - np.array(test_predictions))
    train_true_labels = train_true_labels._values.T.tolist()[0]
    train_residuals = np.abs(np.array(train_true_labels) - np.array(train_predictions))
    a_cal, b_cal = get_calibration_factors(np.array(train_residuals), np.array(train_uncertanties)) #COMMENT pending...
    CAL_train_uncertainties = a_cal * np.array(train_uncertanties) + b_cal #COMMENT pending...
    CAL_test_uncertainties = a_cal * np.array(test_uncertanties) + b_cal #COMMENT pending...
    
    #normalizing the residuals and the uncertainties by making the TRAIN residuals/uncertaintites range from 0 to 1.
    test_normalized_residuals = normalize_array(test_residuals.tolist(), min(train_residuals.tolist()), max(train_residuals.tolist()))
    test_normalized_uncertainties = normalize_array(test_uncertanties, min(train_uncertanties), max(train_uncertanties))
    train_normalized_residuals = normalize_array(train_residuals.tolist(), min(train_residuals.tolist()), max(train_residuals.tolist()))
    train_normalized_uncertainties = normalize_array(train_uncertanties, min(train_uncertanties), max(train_uncertanties))
    CAL_train_normalized_uncertainties = normalize_array(CAL_train_uncertainties, min(CAL_train_uncertainties), max(CAL_train_uncertainties)) #COMMENT pending...
    CAL_test_normalized_uncertainties = normalize_array(CAL_test_uncertainties, min(CAL_train_uncertainties), max(CAL_train_uncertainties)) #COMMENT pending...
    

    train_average_uncertainties, train_RMS_residuals, train_slope, train_intercept, train_average_uncertainties, train_y_pred = bin_uncertainties_and_residuals(train_normalized_uncertainties, train_normalized_residuals, num_bins)
    CAL_train_average_uncertainties, CAL_train_RMS_residuals, CAL_train_slope, CAL_train_intercept, CAL_train_average_uncertainties, CAL_train_y_pred = bin_uncertainties_and_residuals(CAL_train_normalized_uncertainties, train_normalized_residuals, num_bins)
    test_average_uncertainties, test_RMS_residuals, test_slope, test_intercept, test_average_uncertainties, test_y_pred = bin_uncertainties_and_residuals(test_normalized_uncertainties, test_normalized_residuals, num_bins)
    CAL_test_average_uncertainties, CAL_test_RMS_residuals, CAL_test_slope, CAL_test_intercept, CAL_test_average_uncertainties, CAL_test_y_pred = bin_uncertainties_and_residuals(CAL_test_normalized_uncertainties, test_normalized_residuals, num_bins)

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.scatter(train_average_uncertainties, train_RMS_residuals, c='grey', label=f'Binned RvE slope = {train_slope}')
    ax1.scatter(CAL_train_average_uncertainties, CAL_train_RMS_residuals, c='blue', label=f'Calibrated Binned RvEslope = {CAL_train_slope}')
    ax1.plot(train_average_uncertainties, train_y_pred, color='grey')
    ax1.plot(CAL_train_average_uncertainties, CAL_train_y_pred, color='blue')
    ax1.plot([0,1], [0,1], c='red', label='Ideal fitted line')
    ax1.set_title(f'Train set Residual vs Error (RvE) plots for {num_bins} bins')
    ax1.legend()

    
    ax2.scatter(test_average_uncertainties, test_RMS_residuals, c='grey', label=f'Binned RvE slope = {test_slope}')
    ax2.scatter(CAL_test_average_uncertainties, CAL_test_RMS_residuals, c='blue', label=f'Calibrated Binned RvEslope = {CAL_test_slope}')
    ax2.plot(test_average_uncertainties, test_y_pred, color='grey')
    ax2.plot(CAL_test_average_uncertainties, CAL_test_y_pred, color='blue')
    ax2.plot([0,1], [0,1], c='red', label='Ideal fitted line')
    ax2.set_title(f'Test set Residual vs Error (RvE) plots for {num_bins} bins')
    ax2.legend()
    plt.xlabel('Uncertainties')
    plt.ylabel('RMS residuals')
    plt.tight_layout()
    # plt.show()
    plt.savefig(saving_folder + f'/RVE_plot_{num_bins}_bins_{model_type}.png')
    plt.close()
    
    train_r2 = parody_plot_with_std(train_true_labels, train_predictions, train_uncertanties, saving_folder, label_to_predict, model_type, testtrain='Train')

    return train_intercept, train_slope, CAL_train_intercept, CAL_train_slope, train_intercept, test_slope, CAL_test_intercept, CAL_test_slope

''' Plots the residuals and the uncertainties on the x and y axis respectively, giving an idea for how they correlate to eachother. '''
def plot_residuals_vs_uncertainties(predictions, uncertainites, true_labels, saving_folder, model_type, label_to_predict):
    true_labels = true_labels.to_numpy().reshape((true_labels.shape[0],))
    predictions = np.array(predictions)
    residuals = np.abs(predictions - true_labels)
    
    plt.scatter(residuals, np.array(uncertainites))
    plt.xlabel('residuals')
    plt.ylabel('uncertainties')
    plt.title("Residuals vs uncertainties")
    plt.show()
    # plt.savefig(saving_folder + f'/{model_type}_RvE_plot_{num_bins}_bins.png')
    # plt.close()
    
    return

''' creates the 5-fold cross validation datasets for each label to predict in labels to predict in the path given. '''
def make_5_fold_datasets(saving_folder, full_dataset_pathname, image_folder, normalize=True):
    all_labels = ['height', 'phi', 'theta', 
            'impact site x', 'impact site y', 'impact site z', 
            'impact site r', 'impact site phi', 'impact site theta']

    labels_to_predict = ['impact site x', 'impact site y', 'height']

    for label_to_predict in labels_to_predict:
        make_dirs(f'{saving_folder}/{label_to_predict}')
        # if(not os.path.exists(f'{saving_folder}/{label_to_predict}')): os.mkdir(f'{saving_folder}/{label_to_predict}')
        # full_dataset_features, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=1)
        data = pd.read_csv(full_dataset_pathname)
        
        full_dataset_features = data.drop(all_labels, axis=1)
        if(normalize == True):
            # Zero-center the data
            data_centered = full_dataset_features - data.mean()
            # Normalize to the range [-1, 1]
            full_dataset_features = data_centered / data_centered.abs().max()
        full_dataset_labels = data[label_to_predict]
        
        rnge = range(1, len(full_dataset_features)+1)
        kf5 = KFold(n_splits=5, shuffle=True)
        fold_no = 1
        for train_index, test_index in kf5.split(rnge):
            if(not os.path.exists(f'{saving_folder}/{label_to_predict}/fold{fold_no}')): os.mkdir(f'{saving_folder}/{label_to_predict}/fold{fold_no}')
            
            train_df = full_dataset_features.iloc[train_index]
            y_train = full_dataset_labels.iloc[train_index]
            test_df = full_dataset_features.iloc[test_index]
            y_test = full_dataset_labels.iloc[test_index]
            
            # y_train = pd.DataFrame(y_train, columns=[label_to_predict])
            # y_test = pd.DataFrame(y_test, columns=[label_to_predict])

            train_df.to_csv(f'{saving_folder}/{label_to_predict}/fold{fold_no}/train_features.csv', index=False)
            y_train.to_csv(f'{saving_folder}/{label_to_predict}/fold{fold_no}/train_labels.csv', index=False)
            test_df.to_csv(f'{saving_folder}/{label_to_predict}/fold{fold_no}/test_features.csv', index=False)
            y_test.to_csv(f'{saving_folder}/{label_to_predict}/fold{fold_no}/test_labels.csv', index=False)

            fold_no += 1


''' makes all of the directories in a directory path if they dont exist. '''
def make_dirs(directory_path):
    splits = directory_path.split('/')
    current_dir = ''
    for i in range(1, len(splits)):
        current_dir = current_dir + f'/{splits[i]}'
        if(not os.path.exists(current_dir)): os.mkdir(current_dir)
    return



'''
Helper function for make_calibration_plots(). This code was taken directly (and then slighly edited) from 
https://github.com/ulissigroup/uncertainty_benchmarking/blob/master/NN_ensemble/assess_ensemble.ipynb
which is from the paper Methods for comparing uncertainty quantifications for material property predictions by Tran et al.
'''
def calculate_density(percentile, predictions, true_values, uncertainties):
    # Define a normalized bell curve we'll be using to calculate calibration
    norm = stats.norm(loc=0, scale=1)
    residuals = predictions - true_values.reshape(-1)
    stdevs = np.array(uncertainties) / 2 #COMMENT need to fix this because the uncertainties that I plot are 2 std's
    
    '''
    Calculate the fraction of the residuals that fall within the lower
    `percentile` of their respective Gaussian distributions, which are
    defined by their respective uncertainty estimates.
    '''
    # Find the normalized bounds of this percentile
    upper_bound = norm.ppf(percentile)

    # Normalize the residuals so they all should fall on the normal bell curve
    normalized_residuals = residuals.reshape(-1) / stdevs.reshape(-1)

    # Count how many residuals fall inside here
    num_within_quantile = 0
    for resid in normalized_residuals:
        if resid <= upper_bound:
            num_within_quantile += 1

    # Return the fraction of residuals that fall within the bounds
    density = num_within_quantile / len(residuals)
    return density



''' making calibration plots. This code was taken directly (and then slighly edited) from 
    https://github.com/ulissigroup/uncertainty_benchmarking/blob/master/NN_ensemble/assess_ensemble.ipynb
    which is from the paper Methods for comparing uncertainty quantifications for material property predictions by Tran et al.
    
    It makes the calibration plots, but also calculates and returns the miscalibration area.
'''
def make_calibration_plots(model_name, predictions, true_values, uncertainties, saving_folder):
    # %matplotlib inline
    import numpy as np
    from matplotlib import pyplot as plt
    import seaborn as sns
    from shapely.geometry import Polygon, LineString
    from shapely.ops import polygonize, unary_union
    from tqdm import tqdm_notebook
    import tqdm
    from tqdm import notebook
    
    predicted_pi = np.linspace(0, 1, 100)
    # observed_pi = [calculate_density(quantile, predictions, true_values, uncertainties)
    #             for quantile in tqdm_notebook(predicted_pi, desc='Calibration')]
    # observed_pi = [calculate_density(quantile, predictions, true_values, uncertainties)
    #             for quantile in notebook(predicted_pi, desc='Calibration')]
    observed_pi = [calculate_density(quantile, predictions, true_values, uncertainties)
                for quantile in predicted_pi]

    calibration_error = ((predicted_pi - observed_pi)**2).sum()
    print('Calibration error = %.2f' % calibration_error)
    
    # Set figure defaults
    width = 4  # Because it looks good
    fontsize = 12
    rc = {'figure.figsize': (width, width),
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'legend.fontsize': fontsize}
    sns.set(rc=rc)
    sns.set_style('ticks')
    
    # Plot settings
    figsize = (width, width)

    # Plot the calibration curve
    fig_cal = plt.figure(figsize=figsize)
    ax_ideal = sns.lineplot(x=[0, 1], y=[0, 1], label='ideal')
    _ = ax_ideal.lines[0].set_linestyle('--')
    ax_gp = sns.lineplot(x=predicted_pi, y=observed_pi, label=model_name)
    ax_fill = plt.fill_between(predicted_pi, predicted_pi, observed_pi,
                            alpha=0.2, label='miscalibration area')
    _ = ax_ideal.set_xlabel('Expected cumulative distribution')
    _ = ax_ideal.set_ylabel('Observed cumulative distribution')
    _ = ax_ideal.set_xlim([0, 1])
    _ = ax_ideal.set_ylim([0, 1])

    # Calculate the miscalibration area.
    polygon_points = []
    for point in zip(predicted_pi, observed_pi):
        polygon_points.append(point)
    for point in zip(reversed(predicted_pi), reversed(predicted_pi)):
        polygon_points.append(point)
    polygon_points.append((predicted_pi[0], observed_pi[0]))
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy # original data
    ls = LineString(np.c_[x, y]) # closed, non-simple
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    polygon_area_list =[poly.area for poly in polygonize(mls)]
    miscalibration_area = np.asarray(polygon_area_list).sum()

    # Annotate the plot with the miscalibration area
    plt.text(x=0.95, y=0.05,
            s='Miscalibration area = %.2f' % miscalibration_area,
            verticalalignment='bottom',
            horizontalalignment='right',
            fontsize=fontsize)
    plt.savefig(saving_folder +  f'/calibration_plot.png')
    # plt.show()
    plt.close()
    return miscalibration_area, calibration_error
    
    

''' making sharpness plots. This code was taken directly (and then slighly edited) from 
    https://github.com/ulissigroup/uncertainty_benchmarking/blob/master/NN_ensemble/assess_ensemble.ipynb
    which is from the paper Methods for comparing uncertainty quantifications for material property predictions by Tran et al.
    
    It makes the sharpness plots, but also calculates and returns the sharpness and dispersion values.
'''
def plot_sharpness_curve(stdevs, saving_folder):
    width = 4
    figsize = (width, width)
    fontsize = 12
     # Plot sharpness curve
    xlim = [0, max(stdevs)]
    fig_sharp = plt.figure(figsize=figsize)
    # ax_sharp = sns.histplot(stdevs, kde=False, norm_hist=True)
    ax_sharp = sns.histplot(stdevs, kde=False, stat="density", binwidth=0.4, )
    ax_sharp.set_xlim(xlim)
    ax_sharp.set_xlabel('Predicted standard deviation')
    ax_sharp.set_ylabel('Normalized frequency')
    ax_sharp.set_yticklabels([])
    ax_sharp.set_yticks([])

    # Calculate and report sharpness/dispersion
    sharpness = np.sqrt(np.mean(stdevs**2))
    _ = ax_sharp.axvline(x=sharpness, label='sharpness', c='r')
    dispersion = np.sqrt(((stdevs - stdevs.mean())**2).sum() / (len(stdevs)-1)) / stdevs.mean()
    if sharpness < (xlim[0] + xlim[1]) / 2:
        text = '\n  Sharpness = %.2f \n  C$_v$ = %.2f' % (sharpness, dispersion)
        h_align = 'left'
    else:
        text = '\nSharpness = %.2f  \nC$_v$ = %.2f  ' % (sharpness, dispersion)
        h_align = 'right'
    _ = ax_sharp.text(x=sharpness, y=ax_sharp.get_ylim()[1],
                    s=text,
                    verticalalignment='top',
                    horizontalalignment=h_align,
                    fontsize=fontsize)
    
    plt.savefig(saving_folder + f'/sharpness_plot.png')
    # plt.show()
    plt.close()
    return sharpness, dispersion

