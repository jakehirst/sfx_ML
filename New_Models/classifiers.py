import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from Logistic_regression import *
from Random_forest_classification import *
from SVM_classification import *


'''bins the labels and loads the features. returns the binned labels and the features as pandas dataframes'''
def bin_data(label_to_predict, train_features_path, test_features_path, train_labels_path, test_labels_path, num_bins=3):
    '''removes all columns with Nan'''
    def remove_columns_with_nan(df):
        cols_with_nan = df.columns[df.isna().any()].tolist()
        df_cleaned = df.drop(columns=cols_with_nan)
        return df_cleaned, cols_with_nan
    
    train_features = pd.read_csv(train_features_path)
    test_features = pd.read_csv(test_features_path)
    train_labels = pd.read_csv(train_labels_path)
    test_labels = pd.read_csv(test_labels_path)
    
    train_features, removed_columns = remove_columns_with_nan(train_features)
    test_features = test_features.drop(columns=removed_columns)
    '''getting the binning edges for the label predicted'''
    if(label_to_predict == 'height'):
        bin_edges = np.linspace(1.0, 5.0, num_bins + 1)
    elif(label_to_predict == 'impact site x'):
        bin_edges = np.linspace(-60.0, 60.0, num_bins + 1)

    # Bin the values and convert to categorical type
    train_labels_binned = train_labels.copy()
    train_labels_binned[label_to_predict] = pd.cut(train_labels[label_to_predict], bins=bin_edges, labels=False, include_lowest=True)
    test_labels_binned = test_labels.copy()
    test_labels_binned[label_to_predict] = pd.cut(test_labels[label_to_predict], bins=bin_edges, labels=False, include_lowest=True)
    return train_features, test_features, train_labels_binned, test_labels_binned

def make_confusion_matrix(y, X, model, train_test, save='', show=False):
    cm = confusion_matrix(y, model.predict(X))
    disp = plot_confusion_matrix(model, X, y)
    disp.ax_.set_title(f'Confusion {train_test} set')
    disp.ax_.set_xlabel('Predicted Label')
    disp.ax_.set_ylabel('True Label')
    if(not show):
        plt.savefig(save)
    else:
        plt.show()
        
        
        
        
        
        

        
        
# data_folder = '/Volumes/Jake_ssd/Backward_feature_selection/5fold_datasets'
# label_to_predict = 'height'
# label_to_predict = 'impact site x'
# model_types = ['logistic_regression', 'RF_classifier', 'SVM_classifier']
# # model_types = ['SVM_classifier']
# num_bins = 10
# results_folder = f'/Volumes/Jake_ssd/Classification/{label_to_predict}'

# for model_type in model_types:
#     print(f'\n$$$$$$ MODEL TYPE = {model_type} $$$$$$')
#     for fold_no in range(1,6):
#         print(f'fold = {fold_no}')
#         train_features_path = f'{data_folder}/{label_to_predict}/fold{fold_no}/train_features.csv'
#         test_features_path = f'{data_folder}/{label_to_predict}/fold{fold_no}/test_features.csv'
#         train_labels_path = f'{data_folder}/{label_to_predict}/fold{fold_no}/train_labels.csv'
#         test_labels_path = f'{data_folder}/{label_to_predict}/fold{fold_no}/test_labels.csv'
        
#         train_features, test_features, train_labels, test_labels = bin_data(label_to_predict, train_features_path, test_features_path, train_labels_path, test_labels_path, num_bins=num_bins)
#         '''remove all of the timestep_init features'''
#         train_features = train_features.loc[:, ~train_features.columns.str.contains('timestep_init')]
#         test_features = test_features.loc[:, ~test_features.columns.str.contains('timestep_init')]
        
        

        
#         if(model_type == 'logistic_regression'):
#             model, y_pred_train, y_pred_test = make_log_regression_model(train_features, train_labels, test_features, test_labels, 1000, L1=0, L2=1)
#         elif(model_type == 'RF_classifier'):
#             model, y_pred_train, y_pred_test = make_random_forest_classifier(train_features, train_labels, test_features, test_labels, n_estimators=5000, max_depth=10, bootstrap=True)
#         elif(model_type == 'SVM_classifier'):
#             model, y_pred_train, y_pred_test = make_non_linear_svm_model(train_features, train_labels, test_features, test_labels, C=0.3, kernel='rbf', degree=3, random_state=None)
#         make_confusion_matrix(train_labels, train_features, model, 'train', show=True)
#         make_confusion_matrix(test_labels, test_features, model, 'test', show=True)