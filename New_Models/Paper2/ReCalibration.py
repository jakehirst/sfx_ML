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



#COMMENT nonlinear recalibration by Palmer et al.
# '''Nonlinear recalibration as described by Palmer et al.'''
# def get_nonlinear_recalibration_factors():
#     a, b = get_calibration_factors(train_residuals, train_ensemble_uncertanties)
#     print(f'a = {a} b = {b}')
#     calibrated_train_uncertainties = pd.Series(a * (train_ensemble_uncertanties**((b/2) + 1)), name='train_model_errors')
#     calibrated_test_uncertainties = pd.Series(a * (test_ensemble_uncertanties**((b/2) + 1)), name='test_model_errors')
#     return a, b

# '''using the calibration factors a and b (which are already determined) scale the uncertainties appropriately'''
# def nonlinear_calibrate_uncertainties(a, b, test_uncertainties):
#     calibrated_test_uncertainties = pd.Series(a * (test_uncertainties**((b/2) + 1)), name='test_model_errors')
#     return calibrated_test_uncertainties


# model_path = '/Volumes/Jake_ssd/Paper 2/without_transformations/UQ_bagging_models_without_transformations/impact site y/Single RF/1_models/fold_1/model_no1.sav'
# with open(os.path.join(model_folder, filename), 'rb') as file:
#     model = pickle.load(file)



'''
This function trains an isotonic regression model for recalibration like discussed by Kuleshov et al. 
The isotonic regression model is R in the equations of section 3 of the paper.
Input the predictions, uncertainties, and true values from the CALIBRATION dataset.
'''
# def fit_isotonic_regression_model(predictions, uncertainties, true_values):
#     #make an array of varying confidence intervals from 0% to 100%.
#     expected_percentages = np.linspace(0,1,100)
#     #calculate the actual percentage of true values that are in the p confidence interval for all p from 0.01 to 0.99
#     actual_percentages = np.array([calculate_density(p, predictions, true_values, uncertainties) for p in expected_percentages])
    
#     isotonic_model = IsotonicRegression(out_of_bounds='clip')

#     # isotonic_model.fit(actual_percentages, expected_percentages)#COMMENT this is the code from before
#     isotonic_model.fit(expected_percentages, actual_percentages)
#     #proof that this worked:
#     # plt.scatter(actual_percentages, expected_percentages, c='b', label='Before calibration')
#     # preds = isotonic_model.predict(actual_percentages)
#     # plt.scatter(expected_percentages, preds, c='r', label='After calibration')
#     # plt.legend()
#     # plt.show()
#     '''
#     We use isotonic regression here because it maintains the order of the uncertainties. 
#     Relatively large uncertainty estimates are still relatively large, 
#     and relatively small uncertainty estimates are still relatively small.
#     '''
#     return isotonic_model

'''
helper function that uses the isotonic regression recalibration model to adjust and return uncertainties.
The uncalibrated uncertainties are 2 * standard deviation of prediction (where the prediction is a Gaussian distribution)
The calibrated uncertainties also represent 2 * standard deviation of the prediction, resulting in a differently shaped Gaussian distribution.
The mean_predictions are not included here because the uncalibrated and calibrated predictions will have the same mean.
'''
# def use_isotonic_model_to_adjust_uncertainties(uncalibrated_uncertainties, recalibration_model):
#     # Convert uncalibrated uncertainties (which are 2 * std dev) to a percentile representation.
#     # Assuming the uncertainties are normally distributed, use the CDF to find the percentile 
#     # that corresponds to each uncertainty level.
#     norm = stats.norm(loc=0, scale=1)
#     uncalibrated_percentiles = norm.cdf(np.array(uncalibrated_uncertainties) / 2)  # Divide by 2 to get to std dev
    
#     # Apply the recalibration model to adjust these percentiles
#     calibrated_percentiles = recalibration_model.predict(uncalibrated_percentiles)
    
#     # Convert the calibrated percentiles back to uncertainties (2 * std dev)
#     # Use the PPF (inverse CDF) to find the corresponding standard deviations, then double them.
#     calibrated_uncertainties = 2 * norm.ppf(calibrated_percentiles)
    
#     return calibrated_uncertainties

'''try from before'''
# def use_isotonic_model_to_adjust_uncertainties(uncalibrated_uncertainties, recalibration_model, mean_predictions, true_values):
#     norm = stats.norm(loc=0, scale=1)
#     # Assuming uncalibrated_uncertainties are 2 * std, convert to percentiles
#     uncalibrated_percentiles = norm.cdf(np.array(uncalibrated_uncertainties) / 2)
    
#     # Apply the recalibration model to adjust these percentiles
#     calibrated_percentiles = recalibration_model.predict(uncalibrated_percentiles)
    
#     # Clip the calibrated percentiles to avoid extreme values that lead to 'inf' in ppf
#     safe_calibrated_percentiles = np.clip(calibrated_percentiles, 0.001, 0.999)
    
#     # Convert back to uncertainties (2 * std dev) using the PPF on the clipped percentiles
#     calibrated_uncertainties = 2 * norm.ppf(safe_calibrated_percentiles)
    
#     return calibrated_uncertainties



# def fit_isotonic_regression_model(predictions, uncertainties, true_values):
#     """
#     Train an isotonic regression model for recalibration.
#     The input uncertainties are assumed to be 2 * sta ndard deviation.
#     """
#     uncertainties = np.array(uncertainties)
#     predictions = np.array(predictions)
#     # Compute the cumulative probabilities for the observed values under the model's predictions
#     predicted_probs = norm.cdf(true_values, loc=predictions, scale=uncertainties / 2)
    
#     # Empirical CDF of the predicted probabilities
#     sorted_indices = np.argsort(predicted_probs)
#     sorted_probs = predicted_probs[sorted_indices]
#     empirical_cdf = np.arange(1, len(true_values) + 1) / len(true_values)
    
#     # Fit the isotonic regression model
#     isotonic_model = IsotonicRegression(out_of_bounds='clip')
#     isotonic_model.fit(sorted_probs, empirical_cdf[sorted_indices])  # Ensure matching with sorted indices
    
#     return isotonic_model


'''
This function takes in the predictions and uncertainties of an uncalibrated model, then fits an isotonic regression model to the uncertainties.
Then the uncertainties will be adjusted via the use_isotonic_model_to_adjust_uncertainties() method.
'''
# def train_isotonic_model_and_adjust_uncertainties(mean_predictions, uncertainties, true_values):
#     recalibration_model = fit_isotonic_regression_model(mean_predictions, uncertainties, true_values)
#     calibrated_uncertainties = use_isotonic_model_to_adjust_uncertainties(uncertainties, recalibration_model, mean_predictions, true_values)
#     return calibrated_uncertainties, recalibration_model

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
    # percentiles = np.linspace(0, 100, 101) / 100 #TODO this is defining the uncalibrated confidence intervals in 1% increments (I dont think theres enough data to do so)
    # percentiles = np.linspace(0, 100, 51) / 100 #TODO GO BACK TO THIS WHEN FINISHED WITH SIMPLE CASE
    # percentiles = np.linspace(0,99.99, 5) / 100
    
    z_scores = norm.ppf(1 - (1-(percentiles))/2)

    for std in stds:
        CI_range = z_scores * std
        CI_ranges.append(CI_range)        
    return CI_ranges, percentiles


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
    isotonic_model = IsotonicRegression(out_of_bounds='clip')
    isotonic_model.fit(ideal_percentiles, true_percentage_inside_ranges)
    
    return isotonic_model



def adjust_CI_ranges(cal_CI_ranges, calibrator, old_percentages):
    from scipy.interpolate import interp1d
    # New inputs
    new_percentages = np.linspace(0,.9999,101)
    true_percentage_inside_ranges = calibrator.predict(old_percentages)

    adjusted_CI_ranges = []
    for i in range(len(cal_CI_ranges)):
        CI_range = cal_CI_ranges[i]
        # Creating a linear interpolation function based on the given points
        interp_func = interp1d(true_percentage_inside_ranges, CI_range, kind='linear', fill_value='extrapolate')
        
        # Estimating outputs for the new inputs using the interpolation function
        new_outputs = interp_func(new_percentages)
        adjusted_CI_ranges.append(new_outputs)

    return adjusted_CI_ranges, new_percentages



def calculate_miscal_area(true_percentage_inside_ranges, ideal_percentiles):
    miscal_area = np.trapz(np.abs(true_percentage_inside_ranges - ideal_percentiles), ideal_percentiles)
    return miscal_area



def main():
    recalibration_type = 'linear'
    recalibration_type = 'Kuleshov'
    
    label_to_predict = 'impact site x'
    labels_to_predict = ['impact site x', 'impact site y', 'height']


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
    # model_types = ['Single GPR', 'NN_fed_GPR', 'NN_fed_RF', 'RF_fed_GPR']
    # model_types = ['Single RF']
    # model_types = ['ridge']

    how_many_are_benefitting = 0
    models_not_benefitting = []
    labels_not_benefitting = []
    total = 0
    for label_to_predict in labels_to_predict:
        for model_type in model_types:
            print(f'PREDICTING {label_to_predict} USING {model_type}')
            for fold_no in range(1,6):


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
                
                #defining the residual errors of the predictions
                calibration_residuals = pd.Series(np.abs(calibration_labels - calibration_ensemble_predictions))
                test_residuals = pd.Series(np.abs(test_labels - test_ensemble_predictions))
                
                if(recalibration_type =='linear'):
                    linear_a, linear_b = get_linear_recalibration_factors(calibration_residuals, calibration_uncertanties)
                    calibrated_linear_train_uncertainties = linear_calibrate_uncertainties(linear_a, linear_b, calibration_uncertanties)
                    calibrated_linear_test_uncertainties = linear_calibrate_uncertainties(linear_a, linear_b, test_uncertanties)
                elif(recalibration_type == 'Kuleshov'):
                    ideal_percentiles = np.linspace(0,99.99, 20) / 100
                    ideal_percentiles = np.linspace(0,.9999,101)
                    cal_CI_ranges, ideal_percentiles = get_CI_ranges_from_prediction_distributions(calibration_uncertanties, ideal_percentiles)
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

                    test_CI_ranges, ideal_percentiles = get_CI_ranges_from_prediction_distributions(test_uncertanties, ideal_percentiles)
                    test_true_percentage_inside_ranges_UNCALIBRATED = get_true_percentage_inside_ranges(test_CI_ranges, test_ensemble_predictions, test_labels)
                    
                    adjusted_test_CI_ranges, new_test_ideal_percentages = adjust_CI_ranges(test_CI_ranges, calibrator, ideal_percentiles)
                    test_true_percentage_inside_ranges_CALIBRATED = get_true_percentage_inside_ranges(adjusted_test_CI_ranges, test_ensemble_predictions, test_labels)

                    uncalibrated_miscal_area = calculate_miscal_area(test_true_percentage_inside_ranges_UNCALIBRATED, ideal_percentiles)
                    calibrated_miscal_area = calculate_miscal_area(test_true_percentage_inside_ranges_CALIBRATED, ideal_percentiles)
                    
                    if(uncalibrated_miscal_area > calibrated_miscal_area):
                        how_many_are_benefitting += 1
                        total += 1
                    else:
                        models_not_benefitting.append(model_type)
                        labels_not_benefitting.append(label_to_predict)
                        total += 1 
                    print(f'percentage benefitting = {how_many_are_benefitting/total}')
                    # plt.plot(ideal_percentiles, test_true_percentage_inside_ranges_UNCALIBRATED, c='blue', label=f'uncalibrated, miscal_area = {uncalibrated_miscal_area}')
                    # plt.plot(new_test_ideal_percentages, test_true_percentage_inside_ranges_CALIBRATED, c='cyan', label=f'calibrated, miscal_area = {calibrated_miscal_area}')
                    # plt.plot(ideal_percentiles, ideal_percentiles, c='red')
                    # plt.legend(fontsize=10)
                    # plt.title('before and after calibration TEST SET', fontsize=10)
                    # plt.show()
                    # print('here')
                    
                    
                    
                else:
                    print('invalid recalibration type')
                    return
        
    print('done!!')


if __name__ == "__main__":
    main()