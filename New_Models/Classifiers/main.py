from classifiers import *

'''First, we need to define the path of where to get the dataset, and define other parameters that we will need'''
import sys
sys.path.append('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/New_Models')
sys.path.append('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/New_Models/Paper2')

from Bagging_models import *
from ReCalibration import *
from Backward_feature_selection import *
import ast
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


# def bin_labels(data_folder, labels, num_bins):
#     all_labels = {}
#     '/Volumes/Jake_ssd/classifiers/5fold_datasets/impact site x/fold1/test_labels.csv'
#     for label in labels:
#         if(label == 'impact site x'):
#             min_value = -55 ; max_value = 45
#         elif(label == 'impact site y'):
#             min_value = -35 ; max_value = 45
#         elif(label == 'height'):
#             min_value = 1 ; max_value = 5
#         else: print('wrong label... ')
        
#         for fold in range(1,6):
#             test_labels = pd.read_csv(f'/Volumes/Jake_ssd/classifiers/5fold_datasets/{label}/fold{fold}/test_labels.csv')
#             train_labels = pd.read_csv(f'/Volumes/Jake_ssd/classifiers/5fold_datasets/{label}/fold{fold}/train_labels.csv')
#             binned_test_labels = pd.cut(test_labels[label], bins=np.linspace(min_value, max_value, num_bins + 1), labels=False, include_lowest=True)
#             binned_train_labels = pd.cut(train_labels[label], bins=np.linspace(min_value, max_value, num_bins + 1), labels=False, include_lowest=True)
            
#     return

def bin_labels(train_labels, test_labels, label, num_bins):
    all_labels = {}
    '/Volumes/Jake_ssd/classifiers/5fold_datasets/impact site x/fold1/test_labels.csv'
    if(label == 'impact site x'):
        min_value = -55 ; max_value = 45
    elif(label == 'impact site y'):
        min_value = -35 ; max_value = 45
    elif(label == 'height'):
        min_value = 1 ; max_value = 5
    else: print('wrong label... ')
        
    binned_test_labels = pd.cut(test_labels[label], bins=np.linspace(min_value, max_value, num_bins + 1), labels=False, include_lowest=True)
    binned_train_labels = pd.cut(train_labels[label], bins=np.linspace(min_value, max_value, num_bins + 1), labels=False, include_lowest=True)
            
    return binned_test_labels, binned_train_labels
            
            
def main():
    all_labels = ['height', 'phi', 'theta', 
                            'impact site x', 'impact site y', 'impact site z', 
                            'impact site r', 'impact site phi', 'impact site theta']

    labels_to_predict = ['impact site x', 'impact site y', 'height']
    labels_to_predict = ['height']
    labels_to_predict = ['impact site x']

    with_or_without_transformations = 'without'

    classifier_path = f'/Volumes/Jake_ssd/classifiers'
    Paper2_path = f'/Volumes/Jake_ssd/Paper 2/recalibrations/{with_or_without_transformations}_transformations'

    if(not os.path.exists(classifier_path)): os.makedirs(classifier_path)
    model_folder = classifier_path + f'/classification_models_{with_or_without_transformations}_transformations'
    data_folder = classifier_path + '/5fold_datasets'
    results_folder = classifier_path + '/Compare_Code_5_fold_ensemble_results'
    # hyperparam_folder = Paper2_path + f'/bayesian_optimization_{with_or_without_transformations}_transformations'
    hyperparam_folder = f'/Volumes/Jake_ssd/Paper 2/{with_or_without_transformations}_transformations' + f'/bayesian_optimization_{with_or_without_transformations}_transformations'


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
    


    '''Only have to uncomment this if the 5 fold datasets have not been made or need to be remade'''
    # data = pd.read_csv(full_dataset_pathname, index_col=0)
    # make_5_fold_datasets(data_folder, data, all_labels)
    



    model_types = ['XGBoost', 'RF']
    for model_type in model_types:
        print(f'\nMODEL = {model_type}')
        for fold_no in range(1,6):
            print(fold_no)
            for label_to_predict in labels_to_predict:
                X_train = pd.read_csv(f'/Volumes/Jake_ssd/classifiers/5fold_datasets/{label_to_predict}/fold{fold_no}/train_features.csv')
                X_test = pd.read_csv(f'/Volumes/Jake_ssd/classifiers/5fold_datasets/{label_to_predict}/fold{fold_no}/test_features.csv')
                X_train = X_train.drop('timestep_init', axis=1)
                X_test = X_test.drop('timestep_init', axis=1)
                y_train = pd.read_csv(f'/Volumes/Jake_ssd/classifiers/5fold_datasets/{label_to_predict}/fold{fold_no}/train_labels.csv')
                y_test = pd.read_csv(f'/Volumes/Jake_ssd/classifiers/5fold_datasets/{label_to_predict}/fold{fold_no}/test_labels.csv')
                
                '''bin the labels'''
                y_test_binned, y_train_binned = bin_labels(y_train, y_test, label_to_predict, 3)
                
                if(model_type == 'RF'):
                    train_random_forest_classifier(X_train, y_train_binned, X_test, y_test_binned, n_estimators=100, max_depth=5, random_state=None, bootstrap=True)
                
                elif(model_type == 'XGBoost'):
                    train_gradient_boosting_classifier(X_train, y_train_binned, X_test, y_test_binned, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None)
                
                # print('here')
if __name__ == "__main__":
    main()