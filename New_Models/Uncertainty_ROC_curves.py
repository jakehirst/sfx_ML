from GPR import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

'''
This code is a way to use a concept similar to the ROC curve to try and decide on what a good combination for accuracy and UQ is.

Normally, the ROC curve plots the True Positive and False Positive rate over all possible thresholds, where the theshold is a 
value for logistic regression for example where the predictions are split into true and false. For example, the true value array
and prediction array are as follows:

    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    y_predict = np.array([0.2, 0.3, 0.6, 0.8, 0.4, 0.7, 0.1, 0.9, 0.85, 0.5])

The threshold value will start at say 0.05. In this case, all of the predictions will be labeled as 1 (true), so our
true positive rate will be very high but the false positive rate will be very high as well. As we increase the threshold,
we eventually get to a threshold value of 0.5. At this point, all of the y_predict will be correctly classified, so our true positive
rate will be high, and our false positive rate will be low. Keep increasing the threshold, and our false positive rate will continue to
be very low, but the True positive rate will decrease eventually to zero.

In our case, we will be replacing these thresholds with varying values of uncertainty. We will be varying the uncertainty that classifies the 
predictions into confident (positive) or unconfident (negative) and the accuracy (True or False) will be determined by whether the mean 
of the prediction is closer than some value "max_D". Since a single prediction has two uncertainties (one in the x direction and one in the
y direction) the real uncertainty of a prediction will be based on the euclidean distance of the uncertainty from a 100% certain prediction
(the uncertainty in the x and y direction are both 0. The uncertainties will be varied from the minimum standard deviation of the predictions, 
to the maximum standard deviation of the predictions.

'''

'''
obviously gets the euclidean distance between two points in 2D space
'''
def euclidean_distance(x_T, y_T, x_P, y_P):
    return np.sqrt((x_T - x_P)**2 + (y_T - y_P)**2)

''' 
gets an array that classifies each prediction into an accurate predicition or an inaccurate prediction based on the maximum 
distance allowed for an accurate prediction (max_D)
'''
def get_accuracy_array(x_true, y_true, x_pred_mean, y_pred_mean, max_D):
    accuracy_array = []
    for i in range(len(x_true)):
        d = euclidean_distance(x_true[i], y_true[i], x_pred_mean[i], y_pred_mean[i])
        if(d <= max_D): #comparing the distance between the true value and prediction
            accuracy_array.append(1) #this means that the prediction is accurate
        else:
            accuracy_array.append(0) #this means that the predicition is inaccurate
    return accuracy_array

''' 
gets an array that shows the confidence of each prediction based on the euclidean distance of the standard deviation 
in the x and y directions from a perfectly confident prediction (x,y) = (0,0).
'''
def get_confidence_array(x_pred_std, y_pred_std):
    confidence_array = []
    for i in range(len(x_pred_std)):
        confidence_array.append(euclidean_distance(0, 0, x_pred_std[i], y_pred_std[i]))
    return confidence_array


''' 
returns the true_x, true_y, x_pred, x_std, y_pred and y_std of either the test or training dataset of a certain fold in the k-fold cross validation. 
'''
def get_predictions(x_models, y_models, train_or_test, fold_no):
    (x_model, x_true_test, x_test_df, x_true_train, x_train_df) = x_models[fold_no]
    (y_model, y_true_test, y_test_df, y_true_train, y_train_df) = y_models[fold_no]
    if(train_or_test == "Train"):
        x_pred_mean, x_pred_std = x_model.predict(x_train_df.to_numpy(), return_std=True)
        y_pred_mean, y_pred_std = y_model.predict(y_train_df.to_numpy(), return_std=True)
        return x_true_train, y_true_train, x_pred_mean, x_pred_std, y_pred_mean, y_pred_std
    elif(train_or_test == "Test"):
        x_pred_mean, x_pred_std = x_model.predict(x_test_df.to_numpy(), return_std=True)
        y_pred_mean, y_pred_std = y_model.predict(y_test_df.to_numpy(), return_std=True)
        return x_true_test, y_true_test, x_pred_mean, x_pred_std, y_pred_mean, y_pred_std
    else: 
        print("must be predicting either the train or test set.")
    return


''' 
binarizes the confidence_array into confident or unconfident based on the std_thresh. If the std is less or equal to the threshold, 
the prediction is considered confident (1) if it is over the threshold, it is considered unconfident (0).
'''
def get_binary_confidence_arr(confidence_array, std_thresh):
    return np.where(np.array(confidence_array) <= std_thresh, 1, 0)

'''
gets the sensitivity of predictions. High sensitivity in this case means that most of your accurate predicitons are also confident.
low sensitivity means that most of your accurate predicitons are not confident.
'''
def get_sensitivity(binary_confidence_array, accuracy_array):
    def count_ones(arr):
        return np.sum(arr == 1)
    def count_matching_ones(arr1, arr2):
        return np.sum(np.logical_and(arr1 == 1, arr2 == 1))
    total_positive = count_ones(np.array(accuracy_array))#prediction values are actually accurate no matter the confidence
    correctly_classified_positive = count_matching_ones(binary_confidence_array, np.array(accuracy_array))#prediction values that are confident AND accurate
    sensitivity = correctly_classified_positive / total_positive
    return sensitivity

'''
gets the specificity of predictions. High specificity in this case means that most of your inaccurate predicitons are also unconfident.
low specificity means that most of your inaccurate predicitons are confident.
'''
def get_specificity(binary_confidence_array, accuracy_array):
    def count_zeros(arr):
        return np.sum(arr == 0)
    def count_matching_zeros(arr1, arr2):
        return np.sum(np.logical_and(arr1 == 0, arr2 == 0))
    total_negative = count_zeros(np.array(accuracy_array))#prediction values are actually inaccurate no matter the confidence
    correctly_classified_negative = count_matching_zeros(binary_confidence_array, np.array(accuracy_array))#prediction values that are unconfident AND inaccurate
    specificity = correctly_classified_negative / total_negative
    return specificity

"""
AUC is the area under the curve, which can be a good estimate of how robust the ROC curve is no matter the threshold value.
AUC is not a good metric for imbalanced datasets. For example if you have 3 patients with a disease and 200 patients without, AUC is
not a reliable metric.
"""
def calculate_auc(sensitivities, specificities):
    auc = np.trapz(sensitivities, x=1 - specificities)
    return auc

""" 
makes the ROC curve with the accuracies being the actual (true) value whether the prediction mean was close or not (1 = close, 0 = not close)
and whether the prediction was confident or not (1 = confident, 0 = unconfident)
"""
def make_roc_curve(accuracy_array, confidence_array, max_D, fold_no, roc_saving_folder):
    if(not os.path.exists(roc_saving_folder + f'/fold_{fold_no}')): os.mkdir(roc_saving_folder + f'/fold_{fold_no}')
    def count_ones(arr):
        return np.sum(arr == 1)
    std_threshold_arr = np.linspace(min(confidence_array), max(confidence_array), 200)
    sensitivities = []
    specificities = []
    plt.figure(figsize=(8,8))
    for std_thresh in std_threshold_arr:
        binary_confidence_array = get_binary_confidence_arr(confidence_array, std_thresh)
        sensitivity = get_sensitivity(binary_confidence_array, accuracy_array)
        specificity = get_specificity(binary_confidence_array, accuracy_array)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        plt.scatter(1-specificity, sensitivity, color='darkorange')

    auc = calculate_auc(np.array(sensitivities), np.array(specificities))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve max_D = {max_D}, threshold_range = {np.round(min(confidence_array), 2)},{np.round(max(confidence_array), 2)} \n AUC = {auc} \n # of accurate predictions = {count_ones(np.array(accuracy_array))}')
    # plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(roc_saving_folder + f'/fold_{fold_no}/max_D_{max_D}')
    plt.close()
    
    return


















full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/New_Crack_Len_FULL_OG_dataframe.csv"
# full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/FULL_OG_dataframe_with_impact_sites_and_Jimmy_RF.csv"
image_folder = '/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx/new_dataset/Visible_cracks'
all_labels = ['height', 'phi', 'theta', 
              'impact site x', 'impact site y', 'impact site z', 
              'impact site r', 'impact site phi', 'impact site theta']


label_to_predict = 'impact site x'
saving_folder=f'/Users/jakehirst/Desktop/model_results/GPR_RBF_kernel_scaled{label_to_predict}/'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.005)
#correlated_featureset = remove_ABAQUS_features(correlated_featureset)
# features_to_keep = ['crack len', 'init x']
# correlated_featureset = correlated_featureset[features_to_keep]

correlated_featureset = correlated_featureset.loc[:, ~correlated_featureset.columns.str.contains('front')]
raw_images = []
x_models, x_performances, x_r2s, x_mse_s = Kfold_Gaussian_Process_Regression(correlated_featureset, full_dataset_labels, important_features, saving_folder, label_to_predict)


label_to_predict = 'impact site y'
saving_folder=f'/Users/jakehirst/Desktop/model_results/GPR_RBF_kernel_scaled{label_to_predict}/'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.005)
#correlated_featureset = remove_ABAQUS_features(correlated_featureset)
# features_to_keep = ['max_kink', 'init y']
# correlated_featureset = correlated_featureset[features_to_keep]
correlated_featureset = correlated_featureset.loc[:, ~correlated_featureset.columns.str.contains('front')]
raw_images = []
y_models, y_performances, y_r2s, y_mse_s = Kfold_Gaussian_Process_Regression(correlated_featureset, full_dataset_labels, important_features, saving_folder, label_to_predict)





roc_saving_folder = '/Users/jakehirst/Desktop/sfx/ROC_plots/Test_set4'
if(not os.path.exists(roc_saving_folder)): os.mkdir(roc_saving_folder)
for fold_no in range(0,5):  
    x_true, y_true, x_pred_mean, x_pred_std, y_pred_mean, y_pred_std = get_predictions(x_models, y_models, 'Test', fold_no)
    confidence_array = get_confidence_array(x_pred_std, y_pred_std)
    for max_D in range(1, 20):
        accuracy_array = get_accuracy_array(x_true, y_true, x_pred_mean, y_pred_mean, max_D)
        make_roc_curve(accuracy_array, confidence_array, max_D, fold_no, roc_saving_folder)

    print('here')
    
roc_saving_folder = '/Users/jakehirst/Desktop/sfx/ROC_plots/Train_set4'
if(not os.path.exists(roc_saving_folder)): os.mkdir(roc_saving_folder)
for fold_no in range(0,5):  
    x_true, y_true, x_pred_mean, x_pred_std, y_pred_mean, y_pred_std = get_predictions(x_models, y_models, 'Train', fold_no)
    confidence_array = get_confidence_array(x_pred_std, y_pred_std)
    for max_D in range(1, 20):
        accuracy_array = get_accuracy_array(x_true, y_true, x_pred_mean, y_pred_mean, max_D)
        make_roc_curve(accuracy_array, confidence_array, max_D, fold_no, roc_saving_folder)

    print('here')
print('done')

