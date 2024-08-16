
# from New_Models.Classifiers.classifier_lib import *

'''First, we need to define the path of where to get the dataset, and define other parameters that we will need'''
import sys
sys.path.append('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/New_Models')
sys.path.append('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/New_Models/Paper2')
sys.path.append('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML')

from New_Models.Classifiers.classifier_lib import *
from Bagging_models import *
from ReCalibration import *
from Backward_feature_selection import *
import ast
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor



            
            
def main():
    all_labels = ['height', 'phi', 'theta', 
                            'impact site x', 'impact site y', 'impact site z', 
                            'impact site r', 'impact site phi', 'impact site theta']

    labels_to_predict = ['impact site x', 'impact site y', 'height']
    labels_to_predict = ['impact site y']
    labels_to_predict = ['height']

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
    



    model_types = ['RF']
    num_bins = 5
    for model_type in model_types:
        print(f'\nMODEL = {model_type}')
        test_accuracies = []
        train_accuracies = []
        for fold_no in range(1,6):
            print(fold_no)
            for label_to_predict in labels_to_predict:
                print(label_to_predict)
                X_train = pd.read_csv(f'/Volumes/Jake_ssd/classifiers/5fold_datasets/{label_to_predict}/fold{fold_no}/train_features.csv')
                X_test = pd.read_csv(f'/Volumes/Jake_ssd/classifiers/5fold_datasets/{label_to_predict}/fold{fold_no}/test_features.csv')
                X_train = X_train.drop('timestep_init', axis=1)
                X_test = X_test.drop('timestep_init', axis=1)
                y_train = pd.read_csv(f'/Volumes/Jake_ssd/classifiers/5fold_datasets/{label_to_predict}/fold{fold_no}/train_labels.csv')
                y_test = pd.read_csv(f'/Volumes/Jake_ssd/classifiers/5fold_datasets/{label_to_predict}/fold{fold_no}/test_labels.csv')
                
                '''bin the labels'''
                y_test_binned = bin_labels(y_test.to_numpy(), label_to_predict, num_bins)
                y_train_binned = bin_labels(y_train.to_numpy(), label_to_predict, num_bins)

                # y_test_binned, y_train_binned = bin_labels(y_train, y_test, label_to_predict, 2)
                
                if(model_type == 'RF'):
                    model, y_pred_train = train_random_forest_classifier(X_train, y_train_binned, n_estimators=100, max_depth=5, random_state=None, bootstrap=True)
                
                elif(model_type == 'XGBoost'):
                    search_space = {
                                        'n_estimators': Integer(20, 200),
                                        'learning_rate': Real(0.01, 1.0, prior='log-uniform'),
                                        # 'max_depth': Integer(1, 10),
                                        # 'min_samples_split': Integer(2, 100),
                                        # 'min_samples_leaf': Integer(1, 100),
                                        # 'subsample': Real(0.5, 1.0)
                                    }
                    # Initialize the Bayesian optimization
                    bayes_search = BayesSearchCV(
                        estimator=GradientBoostingClassifier(random_state=0),
                        search_spaces=search_space,
                        n_iter=32,  # Number of parameter settings that are sampled
                        cv=3,  # 3-fold cross-validation
                        n_jobs=-1,  # Use all available cores
                        verbose=1,
                        scoring='accuracy',
                        random_state=0
                    )
                    # Perform the search
                    bayes_search.fit(X_train, y_train_binned)
                    best_params = bayes_search.best_params_
                    print(f'Best parameters: {best_params}')
                    model, y_pred_train = train_gradient_boosting_classifier(X_train, y_train_binned, n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'], max_depth=3, random_state=None)
                test_accuracies.append(accuracy_score(y_test_binned, model.predict(X_test)))
                train_accuracies.append(accuracy_score(y_train_binned, model.predict(X_train)))
                # threshold = 0.9
                # # Get the probability predictions
                # proba_predictions = model.predict_proba(X_test)
                # # Get the class predictions
                # class_predictions = model.predict(X_test)
                # print(f'original len = {len(class_predictions)}')
                # og_accuracy = accuracy_score(y_test_binned, class_predictions)
                # print(f'original accuracy = {og_accuracy}')
                # # Find the indices where the highest probability exceeds the threshold
                # confident_indices = np.max(proba_predictions, axis=1) > threshold
                # # Filter the predictions and labels based on these indices
                # filtered_class_predictions = class_predictions[confident_indices]
                # filtered_true_labels = y_test_binned[confident_indices]
                # print(f'filtered len =  {len(filtered_class_predictions)}')
                # # Calculate accuracy only for the confident predictions
                # accuracy = accuracy_score(filtered_true_labels, filtered_class_predictions)
                # print(f"Accuracy for predictions with confidence greater than {threshold}: {accuracy:.2f}")
                
                # print('here')
            print(f'\n\n*** current *** average test_accruacy = {np.average(test_accuracies)}')
        print(f'\n\naverage test_accruacy = {np.average(test_accuracies)}')
        print(f'average train_accruacy = {np.average(train_accuracies)}')

if __name__ == "__main__":
    main()