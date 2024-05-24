import sys
sys.path.append('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/New_Models')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bagging_models import *
from Single_UQ_models import *
import ast
from mastml.plots import *
from mastml.models import *
from mastml.error_analysis import *
from sklearn.isotonic import IsotonicRegression

#COMMENT linear recalibration by Palmer et al.
'''gets linear recalibration factors as described by Palmer et al.'''
def get_linear_recalibration_factors(train_residuals, train_uncertanties):
    cf = CorrectionFactors(train_residuals, pd.Series(train_uncertanties))
    a, b = cf.nll()
    print(f'a = {a} b = {b}')
    return a, b

'''using the calibration factors a and b (which are already determined) scale the uncertainties appropriately'''
def linear_calibrate_uncertainties(a, b, test_uncertainties):
    calibrated_test_uncertainties = pd.Series(a * np.array(test_uncertainties) + b, name='test_model_errors')
    return calibrated_test_uncertainties


'''KULESHOV ET AL RECALIBRATION TECHNIQUE (my version)'''

'''
This turns each prediction (which is represented as a gaussian distribution) into 
a numpy array that defines quantiles bounds for each confidence interval percentage between 0 and 100% with steps of 1%.
Example: for predictions = [1] and stds = [1]
CI_ranges[0][68] = 1  (aka the 68% confidence interval is between 0 and 2 (mean +- CI_ranges[0][68]))
CI_ranges[0][95] = 2
This is basically coming from a two-tailed test
'''
def get_CI_ranges_from_prediction_distributions(uncertainties, percentiles):
    stds = uncertainties / 2
    #test case
    # predictions = np.array([2, 100, 150, 200])  # mean values
    # stds = np.array([1, 10, 15, 20])            # standard deviations
    CI_ranges = []    
    z_scores = norm.ppf(1 - (1-(percentiles))/2)

    for std in stds:
        CI_range = z_scores * std
        CI_ranges.append(CI_range)        
    return CI_ranges


'''
This gets the actual percentage of true values that lie within each confidence interval (from 0 to 100%)
'''
def get_true_percentage_inside_ranges(CI_ranges, predictions, labels):
    total_in_ranges = np.zeros(len(CI_ranges[0])) #this will store the empirical CDFs for each prediction kinda...
    residuals = np.abs(predictions - labels)

    for i in range(len(predictions)):
        for j in range(len(CI_ranges[i])):
            if(residuals[i] < CI_ranges[i][j]):
                total_in_ranges[j] += 1
    
    percentage_inside_ranges = total_in_ranges / len(predictions) #percentage of values that actually lie within each confidence interval
    return percentage_inside_ranges

'''
Now make an isotonic regression model that fits the true frequencies 
'''
def make_isotonic_regression_model(true_percentage_inside_ranges, ideal_percentiles):
    # ideal_percentiles = np.linspace(0,1, 101)
    # isotonic_model = IsotonicRegression(out_of_bounds='clip',y_min=0, y_max=0.99999)
    isotonic_model = IsotonicRegression(out_of_bounds='clip')
    isotonic_model.fit(ideal_percentiles, true_percentage_inside_ranges)
    
    return isotonic_model

'''
helper function for adjust_CI_ranges. This interpolates the points between the maximum value and the last true value so that you dont end up with a CI_range like [0, 2, 5.4, 8, 100, 100, 100]'''
def interpolate_end_of_arr(arr, largest_num):
    # Reverse array to find the last value before the sequence of 100s starts
    reversed_arr = arr[::-1]
    first_100_index = len(arr) - 1 - reversed_arr.index(largest_num) if largest_num in arr else len(arr)

    # Find the last non-100 value before the first 100
    last_unique_before_100_index = first_100_index - 1
    while last_unique_before_100_index >= 0 and arr[last_unique_before_100_index] == largest_num:
        last_unique_before_100_index -= 1

    if last_unique_before_100_index < 0 or first_100_index == len(arr):  # No interpolation needed
        return arr

    # Number of elements to interpolate over (including the last unique and excluding the first 100)
    num_elements = first_100_index - last_unique_before_100_index

    # Starting value for interpolation
    start_value = arr[last_unique_before_100_index]

    # Interpolation values between start_value and 100 (inclusive)
    if num_elements > 1:
        interpolation_values = np.linspace(start_value, largest_num, num_elements + 1)
    else:
        interpolation_values = [start_value, largest_num]

    # Update the array with interpolated values from start_value to 100
    new_arr = arr[:last_unique_before_100_index + 1] + interpolation_values[1:].tolist()
    return np.array(new_arr)

'''
Given the CI_ranges of the uncalibrated model, we use the isotonic regression model (the calibrator)
to adjust the CI_ranges. uncal_CI_ranges.
'''
def adjust_CI_ranges(uncal_CI_ranges, calibrator, ideal_percentages, label):
    from scipy.interpolate import interp1d
    # New inputs
    # new_percentages = np.linspace(0,.9999,101) #TODO go back to this if 1% intervals do not work up
    # new_percentages = np.linspace(0,1, 101)
    true_percentage_inside_ranges = calibrator.predict(ideal_percentages)

    adjusted_CI_ranges = []
    for i in range(len(uncal_CI_ranges)):
        CI_range = uncal_CI_ranges[i]
        # Creating a linear interpolation function based on the given points
        interp_func = interp1d(true_percentage_inside_ranges, CI_range, kind='linear', fill_value='extrapolate')
        
        # Estimating outputs for the new inputs using the interpolation function
        new_outputs = interp_func(ideal_percentages)
        #COMMENT sometimes the new CI_ranges can be nan at the least and inf at the most, so we adjust by just replaceing with
        #COMMENT 0 and 100, respectively if the label is impact site, and 0 and 6 if the label is fall height.
        
        if(new_outputs[-1] > 10000): 
            print('here')
        
        if(label == 'height'):
            #replace with 0 and 6
            new_outputs = np.nan_to_num(new_outputs, nan=0.0, posinf=6.0, neginf=0.0)
            new_outputs = interpolate_end_of_arr(new_outputs.tolist(), 6.0)
        else:
            #replace with 0 and 100.
            new_outputs = np.nan_to_num(new_outputs, nan=0.0, posinf=100.0, neginf=0.0)
            new_outputs = interpolate_end_of_arr(new_outputs.tolist(), 100.0)


        adjusted_CI_ranges.append(new_outputs)
        
    return adjusted_CI_ranges


'''
Calculates the area between the ideal CDF and the empirical CDF lines in the calibration plots using trapezoids.
'''
def calculate_miscal_area(true_percentage_inside_ranges, ideal_percentiles):
    miscal_area = np.trapz(np.abs(true_percentage_inside_ranges - ideal_percentiles), ideal_percentiles)
    return miscal_area

'''
This gets the average of the CI_range of the 95% confidence interval. 
This is similar to the way that Kuleshov et al. calculated it, but instead of using the variance, we are using the quantile range of the 68% CI.
It measures how spread out the predictions are on average.

If you are working with the uncalibrated Gaussian distribution, you have to first get the 
CI_ranges using get_CI_ranges_from_prediction_distributions(uncertainties, percentiles).
'''
def calculate_sharpness(CI_ranges, percentiles):
    sixty_8th_percentile_idx = int((len(percentiles) - 1 ) * .68)
    sharpness = np.mean(np.array(CI_ranges)[:, sixty_8th_percentile_idx])
    return sharpness



def calculate_dispersion(CI_ranges, percentiles):
    sixty_8th_percentile_idx = int((len(percentiles) - 1 ) * .68)
    all_68th_percentiles = np.array(CI_ranges)[:, sixty_8th_percentile_idx]
    avg_68th_percentile = np.mean(all_68th_percentiles)
    
    sum_square_diff = (np.sum((all_68th_percentiles - avg_68th_percentile) ** 2)) / (len(percentiles) - 1)
    
    Cv = np.sqrt(sum_square_diff) / avg_68th_percentile
    
    return Cv


# def calibration_plot(ideal_percentiles, test_true_percentage_inside_ranges_UNCALIBRATED, uncalibrated_miscal_area, test_true_percentage_inside_ranges_CALIBRATED=None, calibrated_miscal_area=None):
#     plt.plot(ideal_percentiles, test_true_percentage_inside_ranges_UNCALIBRATED, c='blue', label=f'Miscalibration area = {uncalibrated_miscal_area.round(3)}')
#     if(test_true_percentage_inside_ranges_CALIBRATED != None and calibrated_miscal_area != None):
#         plt.plot(ideal_percentiles, test_true_percentage_inside_ranges_CALIBRATED, c='cyan', label=f'Miscalibration area (calibrated) = {calibrated_miscal_area.round(3)}')
#     plt.plot(ideal_percentiles, ideal_percentiles, c='red')
#     plt.legend(fontsize=10)
#     plt.title('before and after calibration TEST SET', fontsize=10)
#     plt.show()
#     return

def calibration_plot(ideal_percentiles, test_true_percentage_inside_ranges_UNCALIBRATED, uncalibrated_miscal_area, test_true_percentage_inside_ranges_CALIBRATED=None, calibrated_miscal_area=None):
    # Create the figure and the axes
    fig, ax = plt.subplots()
    
    # Plot the uncalibrated line
    ax.plot(ideal_percentiles, test_true_percentage_inside_ranges_UNCALIBRATED, c='blue')
    
    # Check if calibrated data is provided
    if test_true_percentage_inside_ranges_CALIBRATED is not None and calibrated_miscal_area is not None:
        ax.plot(ideal_percentiles, test_true_percentage_inside_ranges_CALIBRATED, c='cyan')
    
    # Plot the ideal (diagonal) line with dashed style
    ax.plot(ideal_percentiles, ideal_percentiles, c='red', linestyle='--')
    
    # Shade the region between the blue and red lines where blue is above red
    miscal_area = np.maximum(test_true_percentage_inside_ranges_UNCALIBRATED - ideal_percentiles, 0)
    ax.fill_between(ideal_percentiles, ideal_percentiles, test_true_percentage_inside_ranges_UNCALIBRATED, color='yellow', alpha=0.3, label=f'Miscalibration area = {uncalibrated_miscal_area.round(3)}')

    # Set legend with smaller font
    ax.legend(fontsize='small')

    ax.set_xlabel('Expected cumulative distribution', fontsize='small')
    ax.set_ylabel('True cumulative distribution', fontsize='small')

    # Set the title with smaller font
    ax.set_title('Calibration plot', fontsize='small')

    # Set smaller font sizes for axes
    ax.tick_params(axis='both', which='major', labelsize='small')

    plt.show()
    
    return

'''uses the CI_ranges to plot the uncertainties of the impact sites as ellipses.'''
def plot_impact_site_with_uncertainty_2D_with_ellipse(CIs_wanted, x_pred, x_CIs, y_pred, y_CIs, x_true, y_true, number_of_percenages, saving_path=None):
    from matplotlib.patches import Ellipse
    def plot_confidence_ellipses(ax, x_pred, y_pred, x_CIs, y_CIs, confidence_intervals):
        """
        Adds confidence interval ellipses to the provided axes object.
        
        :param ax: The matplotlib axes to add ellipses to.
        :param x_pred: The x coordinate of the predicted mean.
        :param y_pred: The y coordinate of the predicted mean.
        :param x_std: The standard deviation in the x direction.
        :param y_std: The standard deviation in the y direction.
        :param confidence_intervals: A list of confidence interval percentages.
        """
        # colors = list(mcolors.CSS4_COLORS.values())  # Get a list of color names
        # np.random.shuffle(colors)  # Shuffle the list to randomize color selection
        # colors = ['red','yellow','green', 'orange', 'blue', 'purple', 'black', 'cyan', 'red', 'green']
        colors = ['black','black','black', 'black', 'black', 'black', 'black', 'black', 'black', 'black']
        
        # For a normal distribution, the number of standard deviations for a given confidence interval can be found
        # by using the inverse of the cumulative distribution function (CDF), often referred to as the z-value.
        # The z-value times the standard deviation gives you the radius of the confidence interval for that percentage.
        for i, confidence in enumerate(confidence_intervals):
            index = int((confidence /100) * (number_of_percenages - 1))
            width = x_CIs[index]
            height = y_CIs[index]
            # z_value = np.abs(np.array([stats.norm.ppf((1 + (confidence / 100)) / 2)]))
            # width = z_value * x_std * 2  # 2x for the diameter
            # height = z_value * y_std * 2  # 2x for the diameter
            # ellipse = Ellipse((x_pred, y_pred), width, height, edgecolor='blue', facecolor='none', label=f'{confidence}% CI')
            # ax.add_patch(ellipse) 
            ellipse_color = colors[i % len(colors)]  # Use modulo to loop over colors if necessary
            ellipse = Ellipse((x_pred, y_pred), width, height, edgecolor=ellipse_color, facecolor='none', label=f'{confidence}% Confidence interval', linewidth=3)
            ax.add_patch(ellipse)
              
    #material basis vectors for RPA bone
    Material_X = np.array([-0.87491124, -0.44839274,  0.18295974])
    Material_Y = np.array([ 0.23213791, -0.71986519, -0.65414532])
    Material_Z = np.array([ 0.42502036, -0.5298472,   0.7339071 ])
    #Center of mass of the RPA bone in abaqus basis
    CM = np.array([106.55,72.79,56.64])
    # #Ossification center of the RPA bone in abaqus basis
    OC = np.array([130.395996,46.6063,98.649696])

    # Assuming convert_coordinates_to_new_basis is a function you've defined elsewhere
    # and it correctly converts 3D coordinates to 2D in this new context.

    # Load parietal bone node locations (already done in your original code)
    parietal_node_location_df = pd.read_csv('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/parital_node_locations.csv')
    RPA_x = parietal_node_location_df['RPA nodes x']
    RPA_y = parietal_node_location_df['RPA nodes y']
    RPA_z = parietal_node_location_df['RPA nodes z']
    
    # No need to convert Z coordinates or use them in plotting
    RPA_x, RPA_y, RPA_z = convert_coordinates_to_new_basis(Material_X, Material_Y, Material_Z, CM, RPA_x, RPA_y, RPA_z)

    # Your predictive model code remains the same, just ensure it doesn't include Z predictions


    # Existing code for loading data and converting coordinates remains the same...

    # Create a 2D figure
    fig, ax = plt.subplots(figsize=(20, 16))

    # Scatter plot for the parietal bone nodes
    ax.scatter(RPA_x, RPA_y, s=100,  c='grey', alpha=.15, label='Parietal bone', zorder=1)

    point_size = 100

    # Scatter plot for the mean predicted and true impact locations
    ax.scatter(x_pred, y_pred, c='cyan', label='Mean prediction', s=point_size*2, zorder=3)
    ax.scatter(x_true, y_true, c='orange', label='True impact location', s=point_size*2, zorder=3)

    # Add confidence interval ellipses
    plot_confidence_ellipses(ax, x_pred, y_pred, x_CIs, y_CIs, CIs_wanted)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('GPR prediction for impact site in right parietal bone', fontweight='bold')
    ax.set_ylim(-60, 60)
    ax.set_xlim(-80, 80)
    ax.legend()

    # Show or save the plot
    if saving_path is None:
        plt.show()
    else:
        plt.savefig(saving_path)
        plt.close()
    return

'''plots the calibration curves of all folds at once just like in Nemani et al. .'''
def plot_calibration_for_all_folds(all_percentages_inside_ranges, ideal_percentiles, model_type, label_to_predict):
    uncalibrated_percentages = np.array(all_percentages_inside_ranges['uncalibrated'])
    calibrated_percentages = np.array(all_percentages_inside_ranges['calibrated'])
    
    # Create the figure and the axes
    fig, ax = plt.subplots()
    
    for i in range(len(uncalibrated_percentages)):
        # Plot the uncalibrated line
        ax.plot(ideal_percentiles, uncalibrated_percentages[i], c='blue', alpha=0.5)
        # Check if calibrated data is provided
        ax.plot(ideal_percentiles, calibrated_percentages[i], c='cyan', alpha=0.5)
    
    # Plot the ideal (diagonal) line with dashed style
    ax.plot(ideal_percentiles, ideal_percentiles, c='red', linestyle='--')
    
    # Shade the region between the blue and red lines where blue is above red
    # miscal_area = np.maximum(test_true_percentage_inside_ranges_UNCALIBRATED - ideal_percentiles, 0)
    # ax.fill_between(ideal_percentiles, ideal_percentiles, test_true_percentage_inside_ranges_UNCALIBRATED, color='yellow', alpha=0.3, label=f'Miscalibration area = {uncalibrated_miscal_area.round(3)}')

    # Set legend with smaller font
    # ax.legend(fontsize='small')

    ax.set_xlabel('Expected cumulative distribution', fontsize='small')
    ax.set_ylabel('True cumulative distribution', fontsize='small')

    # Set the title with smaller font
    ax.set_title(f'{model_type}, {label_to_predict} \nCalibration plot', fontsize='small')

    # Set smaller font sizes for axes
    ax.tick_params(axis='both', which='major', labelsize='small')

    plt.savefig(f'/Users/jakehirst/Desktop/sfx/Presentations_and_Papers/Paper 2/Figures/calibration_plots_for_all_folds/{label_to_predict}_{model_type}.png')
    plt.close()
    return



def main():
    recalibration_type = 'Kuleshov'
    
    labels_to_predict = ['impact site x', 'impact site y', 'height']
    # labels_to_predict = ['impact site y', 'height']
    # labels_to_predict = ['impact site x']
    labels_to_predict = ['impact site x', 'impact site y']



    all_labels = ['height', 'phi', 'theta', 
                                'impact site x', 'impact site y', 'impact site z', 
                                'impact site r', 'impact site phi', 'impact site theta']

    with_or_without_transformations = 'without'

    Paper2_path = f'/Volumes/Jake_ssd/Paper 2/recalibrations/without_transformations'
    model_folder = Paper2_path + f'/UQ_bagging_models_{with_or_without_transformations}_transformations'
    data_folder = Paper2_path + '/5fold_datasets'
    results_folder = Paper2_path + '/Compare_Code_5_fold_ensemble_results'
    hyperparam_folder = Paper2_path + f'/bayesian_optimization_{with_or_without_transformations}_transformations'

    if(with_or_without_transformations == 'with'):
        full_dataset_pathname = "/Volumes/Jake_ssd/Paper 1/Paper_1_results_WITH_feature_engineering/dataset/feature_transformations_2023-11-16/height/HEIGHTALL_TRANSFORMED_FEATURES.csv"
        backward_feat_selection_results_folder = '/Volumes/Jake_ssd/Paper 1/Paper_1_results_WITH_feature_engineering/results'
    else:
        # full_dataset_pathname = "/Volumes/Jake_ssd/Paper 1/Paper_1_results_no_feature_engineering/dataset/New_Crack_Len_FULL_OG_dataframe_2023_11_16.csv"
        full_dataset_pathname = "/Volumes/Jake_ssd/Paper 2/New_Crack_Len_FULL_OG_dataframe_2024_02_22.csv"
        backward_feat_selection_results_folder = Paper2_path + '/Paper_2_results_WITHOUT_feature_engineering/results' 
        df = pd.read_csv(full_dataset_pathname, index_col=0)
        all_features = df.columns
        all_features = all_features.drop(all_labels)
        all_features = str(all_features.drop('timestep_init').to_list())
        print('all features = ' + str(all_features))
        
    model_types = ['ANN', 'GPR', 'RF', 'ridge', 'Single RF', 'Single GPR', 'NN_fed_GPR', 'NN_fed_RF', 'RF_fed_GPR']
    # model_types = ['ridge', 'NN_fed_GPR']
    model_types = ['Single RF']
    # model_types = ['GPR']
    # model_types = ['ridge']

    predictions = {'impact site x': {}, 'impact site y': {}, 'height': {}}
    results = {}
    for label_to_predict in labels_to_predict:
        results[label_to_predict] = {}
        for model_type in model_types:
            results[label_to_predict][model_type] = {}
            results[label_to_predict][model_type]['uncalibrated'] = {'miscal_area': [], 'sharpness': [], 'dispersion': [], 'R^2': []}
            results[label_to_predict][model_type]['calibrated'] = {'miscal_area': [], 'sharpness': [], 'dispersion': [], 'R^2': []}
            print(f'PREDICTING {label_to_predict} USING {model_type}')
            
            all_percentages_inside_ranges = {'uncalibrated': [], 'calibrated': []}
            
            for fold_no in range(1,6):
                predictions[label_to_predict][f'fold {fold_no}'] = {}
                print(f'Working on fold {fold_no}...')


                #defining folders where the datasets are coming from (5-fold cv)
                test_features_path = Paper2_path + f'/5fold_datasets/{label_to_predict}/fold{fold_no}/test_features.csv'
                test_labels_path = Paper2_path + f'/5fold_datasets/{label_to_predict}/fold{fold_no}/test_labels.csv'
                calibration_features_path = Paper2_path + f'/5fold_datasets/{label_to_predict}/fold{fold_no}/calibration_features.csv'
                calibration_labels_path = Paper2_path + f'/5fold_datasets/{label_to_predict}/fold{fold_no}/calibration_labels.csv'
                # r2, current_predictions, uncertainties, test_labels = Get_predictions_and_uncertainty(test_features_path, test_labels_path, model_saving_folder, results_saving_folder, all_features, label_to_predict, model_type)
                
                
                #predicting the test and train sets with the bagging models
                if(['ANN', 'ridge', 'RF', 'GPR'].__contains__(model_type)):
                    #defining folders to get the models and to store the results
                    model_saving_folder = f'{model_folder}/{label_to_predict}/{model_type}/uncalibrated/fold_{fold_no}'
                    results_saving_folder = f'{results_folder}/{label_to_predict}/{model_type}/uncalibrated/fold_{fold_no}'
                    test_r2, test_ensemble_predictions, test_uncertanties, test_labels = Get_predictions_and_uncertainty_with_bagging(test_features_path, test_labels_path, model_saving_folder, results_saving_folder, ast.literal_eval(all_features), label_to_predict, model_type)
                    calibration_r2, calibration_ensemble_predictions, calibration_uncertanties, calibration_labels = Get_predictions_and_uncertainty_with_bagging(calibration_features_path, calibration_labels_path, model_saving_folder, results_saving_folder, ast.literal_eval(all_features), label_to_predict, model_type)
                else:
                    #defining folders to get the models and to store the results
                    model_saving_folder = f'{model_folder}/{label_to_predict}/{model_type}/uncalibrated/fold_{fold_no}'
                    results_saving_folder = f'{results_folder}/{label_to_predict}/{model_type}/uncalibrated/fold_{fold_no}'
                    test_r2, test_ensemble_predictions, test_uncertanties, test_labels = Get_predictions_and_uncertainty_single_model(test_features_path, test_labels_path, model_saving_folder, results_saving_folder, ast.literal_eval(all_features), label_to_predict, model_type)
                    calibration_r2, calibration_ensemble_predictions, calibration_uncertanties, calibration_labels = Get_predictions_and_uncertainty_single_model(calibration_features_path, calibration_labels_path, model_saving_folder, results_saving_folder, ast.literal_eval(all_features), label_to_predict, model_type)
                
                predictions[label_to_predict][f'fold {fold_no}']['predictions'] = test_ensemble_predictions
                predictions[label_to_predict][f'fold {fold_no}']['true_vals'] = test_labels
                
                #defining the residual errors of the predictions
                calibration_residuals = pd.Series(np.abs(calibration_labels - calibration_ensemble_predictions))
                test_residuals = pd.Series(np.abs(test_labels - test_ensemble_predictions))
                
                if(recalibration_type =='linear'):
                    linear_a, linear_b = get_linear_recalibration_factors(calibration_residuals, calibration_uncertanties)
                    calibrated_linear_train_uncertainties = linear_calibrate_uncertainties(linear_a, linear_b, calibration_uncertanties)
                    calibrated_linear_test_uncertainties = linear_calibrate_uncertainties(linear_a, linear_b, test_uncertanties)
                elif(recalibration_type == 'Kuleshov'):
                    number_of_percenages = 1001 
                    ideal_percentiles = np.linspace(0,0.9999,number_of_percenages) #COMMENT using a ceiling of 1.0 does not work... dont know why. Using higher amount of ideal_percentiles leads to better results... dont know why either.
                    
                    cal_CI_ranges = get_CI_ranges_from_prediction_distributions(calibration_uncertanties, ideal_percentiles)
                    true_percentage_inside_ranges = get_true_percentage_inside_ranges(cal_CI_ranges, calibration_ensemble_predictions, calibration_labels)
                    calibrator = make_isotonic_regression_model(true_percentage_inside_ranges, ideal_percentiles)
                    
                    '''see if it worked on calibration set (should be exactly right)'''
                    # adjusted_cal_CI_ranges = adjust_CI_ranges(cal_CI_ranges, calibrator)
                                    
                    # cal_CI_ranges_CALIBRATED, new_ideal_percentages = adjust_CI_ranges(cal_CI_ranges, calibrator, ideal_percentiles)
                    # cal_true_percentage_inside_ranges_CALIBRATED = get_true_percentage_inside_ranges(cal_CI_ranges_CALIBRATED, calibration_ensemble_predictions, calibration_labels)

                    # uncalibrated_miscal_area = calculate_miscal_area(true_percentage_inside_ranges, ideal_percentiles)
                    # calibrated_miscal_area = calculate_miscal_area(cal_true_percentage_inside_ranges_CALIBRATED, ideal_percentiles)
                    
                    # plt.plot(ideal_percentiles, true_percentage_inside_ranges, c='blue', label=f'uncalibrated, miscal_area = {uncalibrated_miscal_area}')
                    # plt.plot(new_ideal_percentages, cal_true_percentage_inside_ranges_CALIBRATED, c='cyan', label=f'calibrated, miscal_area = {calibrated_miscal_area}')
                    # plt.plot(ideal_percentiles, ideal_percentiles, c='red')
                    # plt.legend()
                    # plt.title('before and after calibration CALIBRATION SET')
                    # plt.show()
                    '''see if it worked on calibration set (should be exactly right)'''

                    test_CI_ranges = get_CI_ranges_from_prediction_distributions(test_uncertanties, ideal_percentiles)
                    test_true_percentage_inside_ranges_UNCALIBRATED = get_true_percentage_inside_ranges(test_CI_ranges, test_ensemble_predictions, test_labels)
                    
                    adjusted_test_CI_ranges = adjust_CI_ranges(test_CI_ranges, calibrator, ideal_percentiles, label_to_predict)
                    test_true_percentage_inside_ranges_CALIBRATED = get_true_percentage_inside_ranges(adjusted_test_CI_ranges, test_ensemble_predictions, test_labels)

                    all_percentages_inside_ranges['uncalibrated'].append(test_true_percentage_inside_ranges_UNCALIBRATED)
                    all_percentages_inside_ranges['calibrated'].append(test_true_percentage_inside_ranges_CALIBRATED)
                    
                    predictions[label_to_predict][f'fold {fold_no}']['CI_ranges'] = test_CI_ranges
                    predictions[label_to_predict][f'fold {fold_no}']['recalibrated CI_ranges'] = adjusted_test_CI_ranges
                    
                    uncalibrated_miscal_area = calculate_miscal_area(test_true_percentage_inside_ranges_UNCALIBRATED, ideal_percentiles)
                    calibrated_miscal_area = calculate_miscal_area(test_true_percentage_inside_ranges_CALIBRATED, ideal_percentiles)
                    results[label_to_predict][model_type]['uncalibrated']['miscal_area'].append(uncalibrated_miscal_area)
                    results[label_to_predict][model_type]['calibrated']['miscal_area'].append(calibrated_miscal_area)
                    
                    uncalibrated_sharpness = calculate_sharpness(test_CI_ranges, ideal_percentiles)
                    calibrated_sharpness = calculate_sharpness(adjusted_test_CI_ranges, ideal_percentiles)
                    results[label_to_predict][model_type]['uncalibrated']['sharpness'].append(uncalibrated_sharpness)
                    results[label_to_predict][model_type]['calibrated']['sharpness'].append(calibrated_sharpness)
                    
                    uncalibrated_Cv = calculate_dispersion(test_CI_ranges, ideal_percentiles)
                    calibrated_Cv = calculate_dispersion(adjusted_test_CI_ranges, ideal_percentiles)
                    results[label_to_predict][model_type]['uncalibrated']['dispersion'].append(uncalibrated_Cv)
                    results[label_to_predict][model_type]['calibrated']['dispersion'].append(calibrated_Cv)
                    
                    results[label_to_predict][model_type]['uncalibrated']['R^2'].append(test_r2)
                    results[label_to_predict][model_type]['calibrated']['R^2'].append(test_r2)
                    
                    # calibration_plot(ideal_percentiles, test_true_percentage_inside_ranges_UNCALIBRATED, uncalibrated_miscal_area)
                    # calibration_plot(ideal_percentiles, test_true_percentage_inside_ranges_CALIBRATED, calibrated_miscal_area)

                    # parody_plot_with_std(test_labels, test_ensemble_predictions, test_uncertanties, None, 'impact site x', model_type, testtrain='Test', show=True)
                    print('next fold...')
                    
                else:
                    print('invalid recalibration type')
                    return

            plot_calibration_for_all_folds(all_percentages_inside_ranges, ideal_percentiles, model_type, label_to_predict)

    
    '''showing difference between uncalibrated and calibrated impact site predictions.'''
    fold_no = 1
    example_num = 20
    x_pred = predictions['impact site x'][f'fold {fold_no}']['predictions'][example_num]
    y_pred = predictions['impact site y'][f'fold {fold_no}']['predictions'][example_num]
    x_CIs = predictions['impact site x'][f'fold {fold_no}']['CI_ranges'][example_num]
    y_CIs = predictions['impact site y'][f'fold {fold_no}']['CI_ranges'][example_num]
    x_calibrated_CIs = predictions['impact site x'][f'fold {fold_no}']['recalibrated CI_ranges'][example_num]
    y_calibrated_CIs = predictions['impact site y'][f'fold {fold_no}']['recalibrated CI_ranges'][example_num]
    x_true = predictions['impact site x'][f'fold {fold_no}']['true_vals'][example_num]
    y_true = predictions['impact site y'][f'fold {fold_no}']['true_vals'][example_num]
    
    print('done!!')
    plot_impact_site_with_uncertainty_2D_with_ellipse([10, 20, 30, 40, 50, 60, 70, 80, 90, 99], x_pred, x_CIs, y_pred, y_CIs, x_true, y_true, number_of_percenages, saving_path=None)
    plot_impact_site_with_uncertainty_2D_with_ellipse([10, 20, 30, 40, 50, 60, 70, 80, 90, 99], x_pred, x_calibrated_CIs, y_pred, y_calibrated_CIs, x_true, y_true, number_of_percenages, saving_path=None)

    '''showing difference between uncalibrated and calibrated impact site predictions.'''
    
    
    avg_results = {}
    # Assuming results is a pre-defined dictionary containing the model performance metrics.
    # Assuming model_types is a list containing the different types of models.

    metrics = ['R^2','miscal_area', 'sharpness', 'dispersion']
    for label_to_predict in labels_to_predict:
        for metric in metrics:
            # Set the figure size
            plt.figure(figsize=(12, 6))
            avg_results[metric] = {}
            bar_width = 0.4  # width of the bars

            # Collect the average metrics for each model type and condition (calibrated/uncalibrated)
            for i, model_type in enumerate(model_types):
                uncal_avg_metric = np.average(results[label_to_predict][model_type]['uncalibrated'][metric])
                cal_avg_metric = np.average(results[label_to_predict][model_type]['calibrated'][metric])
                avg_results[metric][model_type] = {'uncalibrated': uncal_avg_metric, 'calibrated': cal_avg_metric}
                
                # Calculate the position of each bar
                uncalibrated_pos = i - bar_width / 2
                calibrated_pos = i + bar_width / 2

                # Plot each bar
                plt.bar(uncalibrated_pos, uncal_avg_metric, width=bar_width, label='Uncalibrated' if i==0 else "", color='blue')
                plt.bar(calibrated_pos, cal_avg_metric, width=bar_width, label='Calibrated' if i==0 else "", color='green')

            # Set x-ticks to be in the middle of the grouped bars
            plt.xticks(range(len(model_types)), model_types, fontsize=11)

            # Add title and labels
            plt.title(f'{str(metric).capitalize()}\nComparison of Calibrated vs Uncalibrated for {label_to_predict.title()}', fontsize=16)
            plt.xlabel('Model Type', fontsize=14)
            plt.ylabel(str(metric).capitalize(), fontsize=14)

            # Add legend
            plt.legend()
            
            # plt.show()
            
            # Show plot
            plt.savefig(f'/Volumes/Jake_ssd/Paper 2/recalibrations/without_transformations/Compare_Code_5_fold_ensemble_results/{label_to_predict}/cal_vs_uncal_{metric}.png')
            plt.close()

        comparison_path = f'/Volumes/Jake_ssd/Paper 2/recalibrations/without_transformations/Compare_Code_5_fold_ensemble_results/{label_to_predict}/cal_vs_uncal_results'
        if(not os.path.exists(comparison_path)): os.makedirs(comparison_path)
        r2_df = pd.DataFrame.from_dict(avg_results['R^2'], orient='index')
        miscal_area_df = pd.DataFrame.from_dict(avg_results['miscal_area'], orient='index')
        sharpness_df = pd.DataFrame.from_dict(avg_results['sharpness'], orient='index')
        dispersion_df = pd.DataFrame.from_dict(avg_results['dispersion'], orient='index')
        r2_df.to_csv(comparison_path + f'/R2_cal_vs_uncal.csv')
        miscal_area_df.to_csv(comparison_path + f'/miscal_area_cal_vs_uncal.csv')
        sharpness_df.to_csv(comparison_path + f'/sharpness_cal_vs_uncal.csv')
        dispersion_df.to_csv(comparison_path + f'/dispersion_cal_vs_uncal.csv')
        
        print('here')
        #'/Volumes/Jake_ssd/Paper 2/recalibrations/without_transformations/Compare_Code_5_fold_ensemble_results/impact site x/ridge/calibrated/impact site x_ridge_calibrated_results.csv'

    print('done')

    
    

if __name__ == "__main__":
    main()