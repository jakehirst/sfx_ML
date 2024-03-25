import os
import pickle
import pandas as pd
import numpy as np
from Bagging_models import *

'''
Helper function predicts the labels of the featureset given, and the uncertainty.
Also compares this to the true labels of the dataset.
'''
def Get_predictions_and_uncertainty_single_model(test_features_path, test_labels_path, model_folder, saving_folder, features_to_keep, label_to_predict, model_type):
    if(saving_folder == None): print('not saving parody plot')
    elif(not os.path.exists(saving_folder)): os.makedirs(saving_folder)
    test_features = pd.read_csv(test_features_path)[features_to_keep]
    test_labels = pd.read_csv(test_labels_path) 

    filename = model_folder + '/model_no1.sav'
    with open(os.path.join(model_folder, filename), 'rb') as file:
        model = pickle.load(file)
    
    # Loop through the pickle files in the folder
    # for filename in os.listdir(model_folder):
    #     if filename.endswith('.sav'):
    #         # Load the model from the pickle file
    #         with open(os.path.join(model_folder, filename), 'rb') as file:
    #             model = pickle.load(file)
    #             models.append(model)
    
    all_model_predictions = []
    # for model in models:
    if(model_type == 'Single RF'): 
        
        tree_predictions = []
        # Iterate over all trees in the random forest
        for tree in model.estimators_:
            # Predict using the current tree
            tree_pred = tree.predict(test_features.to_numpy())
            # Append the predictions to the list
            tree_predictions.append(tree_pred)

        # Convert the list to a NumPy array for easier manipulation if needed
        tree_predictions = np.array(tree_predictions)
        
        # current_predictions = model.predict(test_features.to_numpy()) #COMMENT this is the same as the average of all the individual trees
        current_predictions = np.mean(tree_predictions, axis=0)
        single_pred_stds = np.std(tree_predictions, axis=0)
        
    elif(model_type == 'Single GPR'):
        current_predictions, single_pred_stds = model.predict(test_features.to_numpy(), return_std=True)
        # current_predictions = model.predict(test_features.to_numpy())
            
        # current_predictions = model.predict(test_features.to_numpy())
        current_predictions = current_predictions.reshape(current_predictions.shape[0])
        all_model_predictions.append(current_predictions)
    
    elif(model_type == 'NN_fed_GPR'):
        current_predictions, single_pred_stds = model.predict(test_features.to_numpy())
        # current_predictions = model.predict(test_features.to_numpy())
            
        # current_predictions = model.predict(test_features.to_numpy())
        current_predictions = current_predictions.reshape(current_predictions.shape[0])
        all_model_predictions.append(current_predictions)
        
    elif(model_type == 'NN_fed_RF'):
        current_predictions, single_pred_stds = model.predict(test_features.to_numpy())
        # current_predictions = model.predict(test_features.to_numpy())
            
        # current_predictions = model.predict(test_features.to_numpy())
        current_predictions = current_predictions.reshape(current_predictions.shape[0])
        all_model_predictions.append(current_predictions)
    
    elif(model_type == 'RF_fed_GPR'):
        current_predictions, single_pred_stds = model.predict(test_features.to_numpy())
        # current_predictions = model.predict(test_features.to_numpy())
            
        # current_predictions = model.predict(test_features.to_numpy())
        current_predictions = current_predictions.reshape(current_predictions.shape[0])
        all_model_predictions.append(current_predictions)

    all_model_predictions = np.array(all_model_predictions)
    
    # ensemble_predictions = []
    # ensemble_uncertanties = []
    # for label_no in range(len(test_labels)):
    #     true_label = test_labels.iloc[label_no][0]
    #     mean_prediction, std_prediction = np.mean(all_model_predictions[:, label_no]), np.std(all_model_predictions[:, label_no])
    #     ensemble_predictions.append(mean_prediction)
    #     ensemble_uncertanties.append(std_prediction*2) #uncertainty will be 2 * the std of the predictions
    
    uncertainties = single_pred_stds * 2
    test_or_train = test_features_path.split('_')[-2].split('/')[-1]
    r2 = parody_plot_with_std(test_labels.to_numpy(), current_predictions, uncertainties, saving_folder, label_to_predict, model_type, testtrain=test_or_train)
    
    return r2, np.array(current_predictions), np.array(uncertainties), np.array(test_labels).flatten()



# '''
# Helper function predicts the labels of the featureset given, and the uncertainty.
# Also compares this to the true labels of the dataset.
# '''
# def Get_predictions_and_uncertainty(test_features_path, test_labels_path, model_folder, saving_folder, features_to_keep, label_to_predict, model_type):
#     if(not os.path.exists(saving_folder)): os.makedirs(saving_folder)
#     test_features = pd.read_csv(test_features_path)[features_to_keep]
#     test_labels = pd.read_csv(test_labels_path) 

#     filename = model_folder + '/model_no1.sav'
#     with open(os.path.join(model_folder, filename), 'rb') as file:
#         model = pickle.load(file)
    
#     # Loop through the pickle files in the folder
#     # for filename in os.listdir(model_folder):
#     #     if filename.endswith('.sav'):
#     #         # Load the model from the pickle file
#     #         with open(os.path.join(model_folder, filename), 'rb') as file:
#     #             model = pickle.load(file)
#     #             models.append(model)
    
#     all_model_predictions = []
#     # for model in models:
#     if(model_type == 'Single RF'): 
        
#         tree_predictions = []
#         # Iterate over all trees in the random forest
#         for tree in model.estimators_:
#             # Predict using the current tree
#             tree_pred = tree.predict(test_features.to_numpy())
#             # Append the predictions to the list
#             tree_predictions.append(tree_pred)

#         # Convert the list to a NumPy array for easier manipulation if needed
#         tree_predictions = np.array(tree_predictions)
        
#         # current_predictions = model.predict(test_features.to_numpy()) #COMMENT this is the same as the average of all the individual trees
#         current_predictions = np.mean(tree_predictions, axis=0)
#         single_pred_stds = np.std(tree_predictions, axis=0)
        
#     elif(model_type == 'Single GPR'):
#         current_predictions, single_pred_stds = model.predict(test_features.to_numpy(), return_std=True)
#         # current_predictions = model.predict(test_features.to_numpy())
            
#         # current_predictions = model.predict(test_features.to_numpy())
#         current_predictions = current_predictions.reshape(current_predictions.shape[0])
#         all_model_predictions.append(current_predictions)
    
#     elif(model_type == 'NN_fed_GPR'):
#         current_predictions, single_pred_stds = model.predict(test_features.to_numpy())
#         # current_predictions = model.predict(test_features.to_numpy())
            
#         # current_predictions = model.predict(test_features.to_numpy())
#         current_predictions = current_predictions.reshape(current_predictions.shape[0])
#         all_model_predictions.append(current_predictions)
        
#     elif(model_type == 'NN_fed_RF'):
#         current_predictions, single_pred_stds = model.predict(test_features.to_numpy())
#         # current_predictions = model.predict(test_features.to_numpy())
            
#         # current_predictions = model.predict(test_features.to_numpy())
#         current_predictions = current_predictions.reshape(current_predictions.shape[0])
#         all_model_predictions.append(current_predictions)
    
#     elif(model_type == 'RF_fed_GPR'):
#         current_predictions, single_pred_stds = model.predict(test_features.to_numpy())
#         # current_predictions = model.predict(test_features.to_numpy())
            
#         # current_predictions = model.predict(test_features.to_numpy())
#         current_predictions = current_predictions.reshape(current_predictions.shape[0])
#         all_model_predictions.append(current_predictions)

#     all_model_predictions = np.array(all_model_predictions)
    
#     # ensemble_predictions = []
#     # ensemble_uncertanties = []
#     # for label_no in range(len(test_labels)):
#     #     true_label = test_labels.iloc[label_no][0]
#     #     mean_prediction, std_prediction = np.mean(all_model_predictions[:, label_no]), np.std(all_model_predictions[:, label_no])
#     #     ensemble_predictions.append(mean_prediction)
#     #     ensemble_uncertanties.append(std_prediction*2) #uncertainty will be 2 * the std of the predictions
    
#     uncertainties = single_pred_stds * 2
#     test_or_train = test_features_path.split('_')[-2].split('/')[-1]
#     r2 = parody_plot_with_std(test_labels.to_numpy(), current_predictions, uncertainties, saving_folder, label_to_predict, model_type, testtrain=test_or_train)
    
#     return r2, current_predictions, uncertainties, test_labels