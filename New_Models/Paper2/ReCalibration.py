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
def fit_isotonic_regression_model(predictions, uncertainties, true_values):
    #make an array of varying confidence intervals from 0% to 100%.
    expected_percentages = np.linspace(0,1,100)
    #calculate the actual percentage of true values that are in the p confidence interval for all p from 0.01 to 0.99
    actual_percentages = np.array([calculate_density(p, predictions, true_values, uncertainties) for p in expected_percentages])
    
    isotonic_model = IsotonicRegression(out_of_bounds='clip')
    # isotonic_model.fit(actual_percentages, np.full_like(actual_percentages, true_percentages))
    isotonic_model.fit(expected_percentages, actual_percentages)
    #proof that this worked:
    # plt.scatter(actual_percentages, true_percentages, c='b', label='Before calibration')
    # preds = isotonic_model.predict(true_percentages)
    # plt.scatter(actual_percentages, preds, c='r', label='After calibration')
    # plt.legend()
    # plt.show()
    '''
    We use isotonic regression here because it maintains the order of the uncertainties. 
    Relatively large uncertainty estimates are still relatively large, 
    and relatively small uncertainty estimates are still relatively small.
    '''
    return isotonic_model


def main():
    recalibration_type = 'linear'
    recalibration_type = 'Kuleshov'
    
    label_to_predict = 'impact site x'


    all_labels = ['height', 'phi', 'theta', 
                                'impact site x', 'impact site y', 'impact site z', 
                                'impact site r', 'impact site phi', 'impact site theta']


    with_or_without_transformations = 'without'

    Paper2_path = f'/Volumes/Jake_ssd/Paper 2/{with_or_without_transformations}_transformations'
    model_folder = Paper2_path + f'/UQ_bagging_models_{with_or_without_transformations}_transformations'
    data_folder = Paper2_path + '/5fold_datasets'
    results_folder = Paper2_path + '/Compare_Code_5_fold_ensemble_results'
    hyperparam_folder = Paper2_path + f'/bayesian_optimization_{with_or_without_transformations}_transformations'

    image_folder = '/Users/jakehirst/Desktop/sfx/sfx_ML_data/images_sfx/new_dataset/Visible_cracks'

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

        print(all_features)
        
    model_types = ['Single RF', 'Single GPR', 'NN_fed_GPR', 'NN_fed_RF', 'RF_fed_GPR']
    model_types = ['Single GPR', 'NN_fed_GPR', 'NN_fed_RF', 'RF_fed_GPR']


    for model_type in model_types:
        for fold_no in range(1,6):
            #defining folders to get the models and to store the results
            model_saving_folder = f'{model_folder}/{label_to_predict}/{model_type}/1_models/fold_{fold_no}'
            results_saving_folder = f'{results_folder}/{label_to_predict}/{model_type}/1_models/fold_{fold_no}'

            #defining folders where the datasets are coming from (5-fold cv)
            test_features_path = Paper2_path + f'/5fold_datasets/{label_to_predict}/fold{fold_no}/test_features.csv'
            test_labels_path = Paper2_path + f'/5fold_datasets/{label_to_predict}/fold{fold_no}/test_labels.csv'
            train_features_path = Paper2_path + f'/5fold_datasets/{label_to_predict}/fold{fold_no}/train_features.csv'
            train_labels_path = Paper2_path + f'/5fold_datasets/{label_to_predict}/fold{fold_no}/train_labels.csv'
            # r2, current_predictions, uncertainties, test_labels = Get_predictions_and_uncertainty(test_features_path, test_labels_path, model_saving_folder, results_saving_folder, all_features, label_to_predict, model_type)
            
            
            #predicting the test and train sets with the bagging models
            test_r2, test_ensemble_predictions, test_uncertanties, test_labels = Get_predictions_and_uncertainty(test_features_path, test_labels_path, model_saving_folder, results_saving_folder, ast.literal_eval(all_features), label_to_predict, model_type)
            train_r2, train_ensemble_predictions, train_uncertanties, train_labels = Get_predictions_and_uncertainty(train_features_path, train_labels_path, model_saving_folder, results_saving_folder, ast.literal_eval(all_features), label_to_predict, model_type)
            #defining the residual errors of the predictions
            train_labels_arr = train_labels.to_numpy().T[0]
            train_predictions_arr = np.array(train_ensemble_predictions)
            test_labels_arr = test_labels.to_numpy().T[0]
            test_predictions_arr = np.array(test_ensemble_predictions)
            train_residuals = pd.Series(np.abs(train_labels_arr - train_predictions_arr))
            test_residuals = pd.Series(np.abs(test_labels_arr - test_predictions_arr))
            
            if(recalibration_type =='linear'):
                linear_a, linear_b = get_linear_recalibration_factors(train_residuals, train_uncertanties)
                calibrated_linear_train_uncertainties = linear_calibrate_uncertainties(linear_a, linear_b, train_uncertanties)
                calibrated_linear_test_uncertainties = linear_calibrate_uncertainties(linear_a, linear_b, test_uncertanties)
            elif(recalibration_type == 'Kuleshov'):
                calibration_model = fit_isotonic_regression_model(train_ensemble_predictions, train_uncertanties, train_labels.to_numpy().flatten())
                #TODO fix the code below to use the calibration model
                print('here')
                #plotting residuals and uncertainties
                # test_residuals = np.abs(predictions - true_values)
                new_stds = calibration_model.predict(std_devs)
                
                plt.scatter(normalized_stds, residuals, c='b')
                plt.scatter(new_stds, residuals, c='r')
                plt.xlabel('Uncertainty')
                plt.ylabel('residual')
                # calibrated_linear_train_uncertainties = linear_calibrate_uncertainties(linear_a, linear_b, train_uncertanties)
                # calibrated_linear_test_uncertainties = linear_calibrate_uncertainties(linear_a, linear_b, test_uncertanties)
            else:
                print('invalid recalibration type')
                return
            
            saving_folder = 'none'
            
            #first checking that the training data improves
            parody_plot_with_std(train_labels_arr, train_predictions_arr, train_uncertanties, saving_folder, label_to_predict, model_type, testtrain='unknown_test_train', show=True)
            parody_plot_with_std(train_labels_arr, train_predictions_arr, calibrated_linear_train_uncertainties, saving_folder, label_to_predict, model_type, testtrain='unknown_test_train', show=True)

            #now checking that the testing data improves
            parody_plot_with_std(test_labels_arr, test_predictions_arr, test_uncertanties, saving_folder, label_to_predict, model_type, testtrain='unknown_test_train', show=True)
            parody_plot_with_std(test_labels_arr, test_predictions_arr, calibrated_linear_test_uncertainties, saving_folder, label_to_predict, model_type, testtrain='unknown_test_train', show=True)

            
            print('here')

if __name__ == "__main__":
    main()