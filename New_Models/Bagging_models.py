import os
import pandas as pd
import numpy as np
import random
from prepare_data import *
from sklearn.model_selection import train_test_split
from linear_regression import *
from lasso_regression import *
from ridge_regression import *
from polynomial_regression import *
from GPR import *
from CNN import *



'''
makes a parody plot of the predictions from GPR including the standard deviations
'''
def parody_plot_with_std(y_test, y_pred_test, y_pred_test_std, saving_folder, label_to_predict):
    plt.figure()
    plt.errorbar(y_test, y_pred_test, yerr=y_pred_test_std, fmt='o')
    plt.plot(y_test, y_test, c='r')
    plt.title('Linear Regression ensemble predicting '+f'{label_to_predict}' + ', R2=%.2f' % r2_score(y_test, y_pred_test))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(saving_folder +  f'/ensemble_UQ_parody_plot.png')
    # plt.show()
    plt.close()
    return r2_score(y_test, y_pred_test)

''' specifically saves models for ensembling '''
def save_ensemble_model(model, fold_no, saving_folder):
    # Save the model to a file
    filename = saving_folder + f'/model_no{fold_no}.sav'
    pickle.dump(model, open(filename, 'wb'))
    
''' 
makes num_models number of regression models and saves them into the saving_folder for later bagging ensembling. 
The training sets for each of these models are taken from the training_features and training_labels with replacement. They are the same len as the training_features
'''  
def make_linear_regression_models_for_ensemble(training_features, training_labels, model_saving_folder, label_to_predict, num_models, features_to_keep, num_training_points=False, model_type=None): 
    models = []
    training_features = training_features[features_to_keep]
    if(not os.path.exists(model_saving_folder)): os.mkdir(model_saving_folder)
    
    num_samples = len(training_features)
    
    #now to train all the models and save them
    for model_num in range(num_models):
        # getting a subset of the training dataset with replacement to train this model on
        sampled_index = training_labels.sample(n=num_samples, replace=True).index
        new_train_features = training_features.loc[sampled_index]
        new_train_labels = training_labels.loc[sampled_index]
        
        if(model_type == None):
            print("need to choose a model type")
        elif(model_type == 'linear'):
            model = LinearRegression() 
        elif(model_type == 'lasso'):
            a = 0.1
            model = Lasso(alpha=a)
        elif(model_type == 'ridge'):
            a = 0.1
            model = Ridge(alpha=a)
        elif(model_type == 'poly2'):
            degree = 2
            model = make_pipeline(PolynomialFeatures(degree),LinearRegression())
        elif(model_type == 'poly3'):
            degree = 3
            model = make_pipeline(PolynomialFeatures(degree),LinearRegression())
        elif(model_type == 'poly4'):
            degree = 4
            model = make_pipeline(PolynomialFeatures(degree),LinearRegression())  
        elif(model_type == 'GPR'):
            # kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * RBF()
            # kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * RBF(length_scale=1e1, length_scale_bounds=(1e-6, 1e3))  + WhiteKernel(noise_level=2, noise_level_bounds=(1e-2, 1e2))
            kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * RBF() + WhiteKernel(noise_level=1)
            model = GaussianProcessRegressor(kernel=kernel, random_state=0, alpha=50, n_restarts_optimizer=25)    
        elif(model_type == 'ANN'):
            n = int(0.2 * len(new_train_features))  # 20% of the length of the list
            val_indexes = random.sample(range(len(new_train_features)), n)
            train_indexes = [i for i in range(len(new_train_features)) if i not in val_indexes]

            train_set = new_train_features.iloc[train_indexes]
            train_labels = training_labels.iloc[train_indexes]
            val_set = new_train_features.iloc[val_indexes]
            val_labels = training_labels.iloc[val_indexes]
            # raw_images = []
            model = make_1D_CNN_for_ensemble(train_set, val_set, train_labels, val_labels, patience=100, max_epochs=2000)

            
        if(model_type == 'ANN'):
            save_ensemble_model(model, model_num, model_saving_folder)
            # model.save(model_saving_folder + f"/trained_model_fold_{model_num}.h5")
            # print('do something now')
        else:
            model.fit(new_train_features.to_numpy(), new_train_labels)
            y_pred_train  = model.predict(new_train_features.to_numpy())
            save_ensemble_model(model, model_num, model_saving_folder) 
        # collect_and_save_metrics(y_test, y_pred_test, train_df.__len__(), len(train_df.columns), full_dataset.columns.to_list(), fold_no, saving_folder)
        # collect_and_save_metrics(y_train, y_pred_train, y_test, y_pred_test, list(train_df.columns), fold_no, saving_folder)
        # plot_test_predictions_heatmap(y_test, y_pred_test, y_pred_test_std, fold_no, saving_folder)
        # parody_plot(y_test, y_pred_test, fold_no, saving_folder, label_to_predict, model_type='Linear Regression')

    return

''' 
Does a 80/20 split on the dataset and saves the dataset features and labels that 
we will be using in our ensembling training sets in the data_saving_folder. 
Also keeps the test sets.
'''
def Level_1_test_train_split(full_dataset_pathname, image_folder, all_labels, label_to_predict, data_saving_folder):
    if(not os.path.exists(data_saving_folder)): os.mkdir(data_saving_folder)
    full_dataset_features, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=1)
    train_features, test_features, train_labels, test_labels = train_test_split(full_dataset_features, full_dataset_labels, test_size=0.2)
    
    # Reset index for the training and test dataframes
    train_features = train_features.reset_index(drop=True)
    train_labels = pd.DataFrame(train_labels, columns=['impact site x']).reset_index(drop=True)
    test_features = test_features.reset_index(drop=True)
    test_labels = pd.DataFrame(test_labels, columns=['impact site x']).reset_index(drop=True)
    if(not os.path.exists(data_saving_folder + '/data')): os.mkdir(data_saving_folder + '/data')
    train_features.to_csv(data_saving_folder + '/data/train_features.csv')
    train_labels.to_csv(data_saving_folder + '/data/train_labels.csv')
    test_features.to_csv(data_saving_folder + '/data/test_features.csv')
    test_labels.to_csv(data_saving_folder + '/data/test_labels.csv')

    return

'''
predicts the labels of the featureset given, and the uncertainty.
Also compares this to the true labels of the dataset.
'''
def Get_predictions_and_uncertainty_with_bagging(test_features_path, test_labels_path, model_folder, saving_folder, features_to_keep, label_to_predict):
    if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
    test_features = pd.read_csv(test_features_path, index_col=0)[features_to_keep]
    test_labels = pd.read_csv(test_labels_path, index_col=0) 
    
    # Initialize an empty list to store the loaded models
    models = []

    # Loop through the pickle files in the folder
    for filename in os.listdir(model_folder):
        if filename.endswith('.sav'):
            # Load the model from the pickle file
            with open(os.path.join(model_folder, filename), 'rb') as file:
                model = pickle.load(file)
                models.append(model)
    
    all_model_predictions = []
    for model in models:
        current_predictions = model.predict(test_features.to_numpy())
        current_predictions = current_predictions.reshape(current_predictions.shape[0])
        all_model_predictions.append(current_predictions)

    all_model_predictions = np.array(all_model_predictions)
    
    ensemble_predictions = []
    ensemble_uncertanties = []
    for label_no in range(len(test_labels)):
        true_label = test_labels.iloc[label_no][0]
        mean_prediction, var_prediction = np.mean(all_model_predictions[:, label_no]), np.var(all_model_predictions[:, label_no])
        ensemble_predictions.append(mean_prediction)
        ensemble_uncertanties.append(var_prediction)
    r2 = parody_plot_with_std(test_labels.to_numpy(), ensemble_predictions, ensemble_uncertanties, saving_folder, label_to_predict)
    
    return r2









model_type = 'ANN'
full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_data/New_Crack_Len_FULL_OG_dataframe_2023_07_14.csv"
image_folder = '/Users/jakehirst/Desktop/sfx/sfx_ML_data/images_sfx/new_dataset/Visible_cracks'
all_labels = ['height', 'phi', 'theta', 
              'impact site x', 'impact site y', 'impact site z', 
              'impact site r', 'impact site phi', 'impact site theta']




''' ************* impact site x ************'''
label_to_predict = 'impact site x'
data_saving_folder = f'/Volumes/Jake_ssd/ensembling_models/{label_to_predict}'

Level_1_test_train_split(full_dataset_pathname, image_folder, all_labels, label_to_predict, data_saving_folder)
training_features = pd.read_csv('/Volumes/Jake_ssd/ensembling_models/impact site x/data/train_features.csv', index_col=0)
training_labels = pd.read_csv('/Volumes/Jake_ssd/ensembling_models/impact site x/data/train_labels.csv', index_col=0)


model_saving_folder = f'/Volumes/Jake_ssd/ensembling_models/impact site x/{model_type}'
features_to_keep = ['crack len', 'init phi', 'init x']
make_linear_regression_models_for_ensemble(training_features, training_labels, model_saving_folder, label_to_predict, 50, features_to_keep, model_type=model_type)

test_features_path = '/Volumes/Jake_ssd/ensembling_models/impact site x/data/test_features.csv'
test_labels_path = '/Volumes/Jake_ssd/ensembling_models/impact site x/data/test_labels.csv'
model_folder = f'/Volumes/Jake_ssd/ensembling_models/impact site x/{model_type}'
saving_folder = f'/Volumes/Jake_ssd/model_results/impact site x/{model_type}'
r2 = Get_predictions_and_uncertainty_with_bagging(test_features_path, test_labels_path, model_folder, saving_folder, features_to_keep, label_to_predict)





''' ************* impact site y ************'''
label_to_predict = 'impact site y'
data_saving_folder = f'/Volumes/Jake_ssd/ensembling_models/{label_to_predict}'
Level_1_test_train_split(full_dataset_pathname, image_folder, all_labels, label_to_predict, data_saving_folder)
training_features = pd.read_csv('/Volumes/Jake_ssd/ensembling_models/impact site y/data/train_features.csv', index_col=0)
training_labels = pd.read_csv('/Volumes/Jake_ssd/ensembling_models/impact site y/data/train_labels.csv', index_col=0)


model_saving_folder = f'/Volumes/Jake_ssd/ensembling_models/impact site y/{model_type}'
features_to_keep = ['max_kink', 'init y', 'angle_btw']
make_linear_regression_models_for_ensemble(training_features, training_labels, model_saving_folder, label_to_predict, 50, features_to_keep, model_type=model_type)

test_features_path = '/Volumes/Jake_ssd/ensembling_models/impact site y/data/test_features.csv'
test_labels_path = '/Volumes/Jake_ssd/ensembling_models/impact site y/data/test_labels.csv'
model_folder = f'/Volumes/Jake_ssd/ensembling_models/impact site y/{model_type}'
saving_folder = f'/Volumes/Jake_ssd/model_results/impact site y/{model_type}'
r2 = Get_predictions_and_uncertainty_with_bagging(test_features_path, test_labels_path, model_folder, saving_folder, features_to_keep, label_to_predict)
print(f'r2 = {r2}')
print('\n')






''' ************* height ************'''
label_to_predict = 'height'
data_saving_folder = f'/Volumes/Jake_ssd/ensembling_models/{label_to_predict}'

Level_1_test_train_split(full_dataset_pathname, image_folder, all_labels, label_to_predict, data_saving_folder)
training_features = pd.read_csv('/Volumes/Jake_ssd/ensembling_models/height/data/train_features.csv', index_col=0)
training_labels = pd.read_csv('/Volumes/Jake_ssd/ensembling_models/height/data/train_labels.csv', index_col=0)


model_saving_folder = f'/Volumes/Jake_ssd/ensembling_models/height/{model_type}'
features_to_keep = ['abs_val_sum_kink']
make_linear_regression_models_for_ensemble(training_features, training_labels, model_saving_folder, label_to_predict, 50, features_to_keep, model_type=model_type)

test_features_path = '/Volumes/Jake_ssd/ensembling_models/height/data/test_features.csv'
test_labels_path = '/Volumes/Jake_ssd/ensembling_models/height/data/test_labels.csv'
model_folder = f'/Volumes/Jake_ssd/ensembling_models/height/{model_type}'
saving_folder = f'/Volumes/Jake_ssd/model_results/height/{model_type}'
Get_predictions_and_uncertainty_with_bagging(test_features_path, test_labels_path, model_folder, saving_folder, features_to_keep, label_to_predict)
