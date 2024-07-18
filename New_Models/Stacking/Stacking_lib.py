'''First, we need to define the path of where to get the dataset, and define other parameters that we will need'''
import sys
sys.path.append('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/New_Models')
sys.path.append('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/New_Models/Paper2')
sys.path.append('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/New_Models/Classifiers')

from classifiers import *
from Bagging_models import *
from ReCalibration import *
from Backward_feature_selection import *
import ast



def make_UQ_model(training_features, training_labels, model_saving_folder, label_to_predict, num_models, features_to_keep, hyperparam_folder, num_training_points=False, model_type=None): 
    models = []
    training_features = training_features[features_to_keep]
    current_label = training_labels.columns[0]
    if(not os.path.exists(model_saving_folder)): os.mkdir(model_saving_folder)

    if(model_type == 'Single RF'):
        hp_folder = f'/Volumes/Jake_ssd/Paper 2/{with_or_without_transformations}_transformations_trial_2/bayesian_optimization_{with_or_without_transformations}_transformations'
        depth, features, samples_leaf, samples_split, estimators = get_best_hyperparameters_RF(label_to_predict=training_labels.columns[0], hyperparameter_folder=hp_folder)
        model =  RandomForestRegressor(max_depth=depth, max_features=features, 
                                       min_samples_leaf = samples_leaf, min_samples_split = samples_split, n_estimators=estimators, random_state=42)
        model.fit(training_features, training_labels)
        
    elif(model_type == 'Single GPR'):

        kernel = ConstantKernel(constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale_bounds=(1e2, 1e6)) + WhiteKernel(noise_level_bounds=(1e-10, 1e+3)) 
        # model = GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=200)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=200) #COMMENT removed random state

        model.fit(training_features.to_numpy(), training_labels.to_numpy())
        save_ensemble_model(model, 1, model_saving_folder) 

    elif(model_type == 'NN_fed_GPR'):
        # c, length_scale, noise_level = get_best_hyperparameters_NN_fed_GPR(label_to_predict=training_labels.columns[0], hyperparameter_folder=hyperparam_folder)
        model = NN_fed_GPR()
        model.fit(training_features, training_labels, hyperparam_folder)

    elif(model_type == 'RF_fed_GPR'):
        # c, length_scale, noise_level = get_best_hyperparameters_NN_fed_GPR(label_to_predict=training_labels.columns[0], hyperparameter_folder=hyperparam_folder)
        model = RF_fed_GPR()
        model.fit(training_features, training_labels, hyperparam_folder)
        
    elif(model_type == 'NN_fed_RF'):
        # c, length_scale, noise_level = get_best_hyperparameters_NN_fed_GPR(label_to_predict=training_labels.columns[0], hyperparameter_folder=hyperparam_folder)
        model = NN_fed_RF()
        model.fit(training_features, training_labels, hyperparam_folder, num_optimization_tries=100, hyperparam_folder=f'/Volumes/Jake_ssd/Paper 2/without_transformations/optimized_hyperparams/NN_fed_RF/{current_label}')
        save_ensemble_model(model, 1, model_saving_folder) 

    save_ensemble_model(model, 1, model_saving_folder) 
    # save_ensemble_model(model, 1, '/Users/jakehirst/Desktop/TEST FOLDER')
    
    return 


def make_predictions(model_type, model_folder, label_to_predict, num_models, fold_no, validation_features_path, validation_labels_path, results_folder, features_to_keep):
    '''If the model is an ensemble type, then we train it using the code below'''
    if(model_type in ['ANN', 'RF', 'GPR', 'ridge']):
        model_saving_folder = f'{model_folder}/{label_to_predict}/{model_type}/{num_models}_models/fold_{fold_no}'
        results_saving_folder = f'{results_folder}/{label_to_predict}/{model_type}/{num_models}_models/fold_{fold_no}'
        #make ensemble prediction
        r2, predictions, uncertanties, labels = Get_predictions_and_uncertainty_with_bagging(validation_features_path, 
                                                                                                validation_labels_path, 
                                                                                                model_saving_folder, 
                                                                                                results_saving_folder, 
                                                                                                features_to_keep, 
                                                                                                label_to_predict, 
                                                                                                model_type)
    
    elif(model_type in ['NN_fed_GPR', 'NN_fed_RF', 'RF_fed_GPR', 'Single GPR', 'Single RF']):
        model_saving_folder = f'{model_folder}/{label_to_predict}/{model_type}/1_models/fold_{fold_no}'
        results_saving_folder = f'{results_folder}/{label_to_predict}/{model_type}/1_models/fold_{fold_no}'
        #make union based prediction
        r2, predictions, uncertanties, labels = Get_predictions_and_uncertainty_single_model(validation_features_path, 
                                                                                                validation_labels_path, 
                                                                                                model_saving_folder, 
                                                                                                results_saving_folder, 
                                                                                                features_to_keep, 
                                                                                                label_to_predict, 
                                                                                                model_type)
    # #TODO now we need to implement the classification predictions   
    # elif:
    return r2, predictions, uncertanties, labels
    
'''combining all the base model predictions, and then making a meta prediciton using the meta_model'''
def make_meta_predictions(meta_model, base_model_outputs):
    X_test = base_model_outputs.to_numpy()
    return meta_model.predict(X_test)

'''training the meta model on features which are the outputs of the base models'''
def train_meta_model(meta_model_type, base_model_outputs, labels):
    X_val = base_model_outputs.to_numpy()
    y_val = np.array(labels)
    
    if(meta_model_type == 'linear'):
        meta_model = LinearRegression()
        meta_model.fit(X_val, y_val)

    else:
        print('model not defined')
        
    
    return meta_model