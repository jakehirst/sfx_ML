'''The theory I have is that the RF is more robust to outliers in the data. For example, if there was a test eample that had an outlier value for the max kink angle (90),
then the decision trees in the RF would classify this into its respective group instead of using the value of 90 with a weight to assign its value. In other words, it might just be
classified into the >70 group instead of doing 90*W. So if you had another example with the same features excep tthe max kink angle was 80, then you would get the exact same
prediction. So my theory is that RF's are more robust to outliers, and that is why they perform better than the ANN.

This can be measured (albiet simply) by comparing the MSE and MAE values of predictions. If the MAE is similar for the two models, AND the MSE is very different, 
then there is a discrepancy when it comes to predicting outliers.

RF's are typically poor at extrapolating because of this feature though. ANN's will have a better capability to extrapolate to out-of-domain examples because 
outlier examples will not be 'rounded' back to the value of a training example. 
'''

from Figures import *
from ReCalibration import *
from sklearn.metrics import mean_squared_error, mean_absolute_error

all_labels = ['height', 'phi', 'theta', 
                            'impact site x', 'impact site y', 'impact site z', 
                            'impact site r', 'impact site phi', 'impact site theta']

model_types = ['Single RF']
model_types = ['RF_fed_GPR']
model_types = ['ANN', 'GPR', 'RF', 'ridge', 'Single RF', 'Single GPR', 'NN_fed_GPR', 'NN_fed_RF', 'RF_fed_GPR']
model_types = ['Single RF', 'ANN']

# parietal_nodes_folder = ''
# for fold_no in range(1,2):

all_preds = {}
mse = {}
mae = {}
fold_no = 1
for model_type in model_types:
    print(f'**** {model_type} *****')
    x_model_folder = f'/Volumes/Jake_ssd/Paper 2/without_transformations/UQ_bagging_models_without_transformations/impact site x/{model_type}'
    y_model_folder = f'/Volumes/Jake_ssd/Paper 2/without_transformations/UQ_bagging_models_without_transformations/impact site y/{model_type}'

    #defining folders to get the models and to store the results
    # model_saving_folder = f'{model_folder}/{label_to_predict}/{model_type}/{num_models}_models/fold_{fold_no}'
    results_saving_folder = None
    
    Paper2_path = f'/Volumes/Jake_ssd/Paper 2/without_transformations'
    #defining folders where the datasets are coming from (5-fold cv)
    x_test_features_path = Paper2_path + f'/5fold_datasets/impact site x/fold{fold_no}/test_features.csv'
    x_test_labels_path = Paper2_path + f'/5fold_datasets/impact site x/fold{fold_no}/test_labels.csv'
    y_test_features_path = Paper2_path + f'/5fold_datasets/impact site y/fold{fold_no}/test_features.csv'
    y_test_labels_path = Paper2_path + f'/5fold_datasets/impact site y/fold{fold_no}/test_labels.csv'


    #defining the features that each model used (since they vary with each model)
    # features_to_keep = ????
    df = pd.read_csv(x_test_features_path, index_col=0)
    all_features = df.columns
    # all_features = all_features.drop(all_labels)
    features_to_keep = str(all_features.drop('timestep_init').to_list())
    
    if(model_type in ['ANN', 'RF', 'GPR','ridge']):
        #predicting the test and train sets with the bagging models
        test_r2_X, test_ensemble_predictions_X, test_uncertanties_X, test_labels_X = Get_predictions_and_uncertainty_with_bagging(x_test_features_path, x_test_labels_path, x_model_folder + f'/20_models/fold_{fold_no}', results_saving_folder, ast.literal_eval(features_to_keep), 'impact site x', model_type)
        # test_r2_Y, test_ensemble_predictions_Y, test_uncertanties_Y, test_labels_Y = Get_predictions_and_uncertainty_with_bagging(y_test_features_path, y_test_labels_path, y_model_folder + f'/20_models/fold_{fold_no}', results_saving_folder, ast.literal_eval(features_to_keep), 'impact site y', model_type)
        
    else:
        #predicting the test and train sets with the NON bagging models
        test_r2_X, test_ensemble_predictions_X, test_uncertanties_X, test_labels_X = Get_predictions_and_uncertainty_single_model(x_test_features_path, x_test_labels_path, x_model_folder + f'/1_models/fold_{fold_no}', results_saving_folder, ast.literal_eval(features_to_keep), 'impact site x', model_type)
        # test_r2_Y, test_ensemble_predictions_Y, test_uncertanties_Y, test_labels_Y = Get_predictions_and_uncertainty_single_model(y_test_features_path, y_test_labels_path, y_model_folder + f'/1_models/fold_{fold_no}', results_saving_folder, ast.literal_eval(features_to_keep), 'impact site y', model_type)

    mse[model_type] = mean_squared_error(test_labels_X, test_ensemble_predictions_X)
    mae[model_type] = mean_absolute_error(test_labels_X, test_ensemble_predictions_X)
    all_preds[model_type] = test_ensemble_predictions_X
print('here')
    # random_examples = random.sample(range(51), 30)
    # r2 = parody_plot_with_std(test_labels_X[random_examples], test_ensemble_predictions_X[random_examples], test_uncertanties_X[random_examples], None, 'impact site x', model_type, testtrain='Test', show=True)
    # r2 = parody_plot_with_std(test_labels_X[random_examples], test_ensemble_predictions_X[random_examples], np.ones(test_uncertanties_X[random_examples].shape) * np.mean(test_uncertanties_X), None, 'impact site x', model_type, testtrain='Test', show=True)

    # r2 = parody_plot_with_std(test_labels_X, test_ensemble_predictions_X, test_uncertanties_X, None, 'impact site x', model_type, testtrain='Test', show=True)
    # r2 = parody_plot_with_std(test_labels_X, test_ensemble_predictions_X, np.ones(test_uncertanties_X.shape) * np.mean(test_uncertanties_X), None, 'impact site x', model_type, testtrain='Test', show=True)

