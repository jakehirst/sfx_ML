'''First, we need to define the path of where to get the dataset, and define other parameters that we will need'''
import sys
sys.path.append('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/New_Models')
from NN_fed_GPR import *
from NN_fed_RF import *
from RF_fed_GPR import *
from Bagging_models import *
from Backward_feature_selection import *
from Single_UQ_models import *
import ast
import ipywidgets as widgets
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns


'''fourth to last cell in ipynb'''
def make_datasets(cluster_t, num_clusterz, data_with_clusters):
    '''make RF regression model with this, and see how it performs when predicting height'''
    all_labels = ['height', 'phi', 'theta', 
                                'impact site x', 'impact site y', 'impact site z', 
                                'impact site r', 'impact site phi', 'impact site theta']
    labels_to_predict = ['height']
    model_types = ['Single RF']

    with_or_without_transformations = 'without'


    Paper2_path = f'/Volumes/Jake_ssd/with_clusters/{cluster_t}_{num_clusterz}_clusters'
    model_folder = Paper2_path + f'/UQ_bagging_models_{with_or_without_transformations}_transformations'
    data_folder = Paper2_path + '/5fold_datasets'
    results_folder = Paper2_path + '/Compare_Code_5_fold_ensemble_results'
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
        all_features = data_with_clusters.columns
        all_features = all_features.drop(all_labels)
        all_features = str(all_features.drop('timestep_init').to_list()) #get rid of timestep_init feature,, its cheating...
        print(all_features)
        
        
    '''Only have to uncomment this if the 5 fold datasets have not been made or need to be remade'''
    make_5_fold_datasets(data_folder, data_with_clusters, all_labels=all_labels)

    print('ALL_TRANSFORMED_FEATURES' in full_dataset_pathname)
    
    return labels_to_predict, model_types, backward_feat_selection_results_folder, all_features, data_folder, model_folder, results_folder, hyperparam_folder, Paper2_path

'''third to last cell in ipynb'''
def get_features_to_keep(labels_to_predict, model_types, full_dataset_pathname, backward_feat_selection_results_folder, all_features):
    '''get the appropriate features that each model will use based on backward feature elimination'''
    all_features_to_keep = {}

    min_features = 1 #minimum number of features you want to select from BFS (backward feature selection)
    max_features = 25 #maximum number of features you want to select from BFS
    for label in labels_to_predict:
        all_features_to_keep[label] = {}
        for model_type in model_types:
            
            if('ALL_TRANSFORMED_FEATURES' in full_dataset_pathname):
                print('true')
                model_type_hyperparam = model_type.removeprefix('Single ')
                #TODO use code below if using feature selection
                best_features = get_best_features(backward_feat_selection_results_folder, label, model_type_hyperparam, min_features, max_features)
                all_features_to_keep[label][model_type] = best_features
            
            else:
                print('using just the basic features')
                #TODO use code below if NOT using feature selection
                all_features_to_keep[label][model_type] = all_features

    print(all_features_to_keep)
    return all_features_to_keep


'''second to last cell in ipynb'''
def make_all_models(model_types, labels_to_predict, data_folder, model_folder, results_folder, all_features_to_keep, hyperparam_folder):
    '''Now we will make all of the models'''
    def make_UQ_model(training_features, training_labels, model_saving_folder, label_to_predict, num_models, features_to_keep, hyperparam_folder, num_training_points=False, model_type=None): 
        models = []
        training_features = training_features[features_to_keep]
        current_label = training_labels.columns[0]
        if(not os.path.exists(model_saving_folder)): os.mkdir(model_saving_folder)

        if(model_type == 'Single RF'):
            # hp_folder = f'/Volumes/Jake_ssd/Paper 2/{with_or_without_transformations}_transformations_trial_2/bayesian_optimization_{with_or_without_transformations}_transformations'
            # depth, features, samples_leaf, samples_split, estimators = get_best_hyperparameters_RF(label_to_predict=training_labels.columns[0], hyperparameter_folder=hp_folder)
            # model =  RandomForestRegressor(max_depth=depth, max_features=features, 
            #                                min_samples_leaf = samples_leaf, min_samples_split = samples_split, n_estimators=estimators, random_state=42)
            # model.fit(training_features, training_labels)
            param_space = {
                'n_estimators': (1000, 5000),  #Higher for generalization, lower for overfitting
                'max_depth': (4,25),  #Higher for overfitting, lower for generaliztion
                'min_samples_split': (5, 30),  #Higher for generalization, lower for overfitting
                'min_samples_leaf': (1, 25), #Higher for generalization, lower for overfitting
                # 'max_features': (1, X_train.shape[1]), #Higher for overfitting, lower for generalization
                'max_features': (3,25), #Higher for overfitting, lower for generalization
            }

            # Define the model you want to use
            model = RandomForestRegressor(random_state=42)

            # Setup Bayesian optimization
            opt = BayesSearchCV(
                estimator=model,
                search_spaces=param_space,
                n_iter=50,  # Number of parameter settings that are sampled. 100 tries.
                n_jobs=-1,  # Number of jobs to run in parallel. Set to -1 to use all available cores.
                cv=5,  # Number of folds in cross-validation
                verbose=1,
                random_state=42
            )

            # Perform the optimization
            opt.fit(training_features, training_labels.values.ravel())

            # Best parameter set found
            print("Best parameters found: ", opt.best_params_)

            # Best model
            model = opt.best_estimator_
            
            
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
        
        return 



    for fold_no in range(1,6):
        for model_type in model_types:
            for label_to_predict in labels_to_predict:
                print(f'\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\ Predicting {label_to_predict} using {model_type} $$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n')
                
                # all_labels = ['height', 'phi', 'theta', 
                #             'impact site x', 'impact site y', 'impact site z', 
                #             'impact site r', 'impact site phi', 'impact site theta']

                print(f'{data_folder}/{label_to_predict}/fold{fold_no}/train_features.csv')
                training_features = pd.read_csv(f'{data_folder}/{label_to_predict}/fold{fold_no}/train_features.csv').reset_index(drop=True)
                training_labels = pd.read_csv(f'{data_folder}/{label_to_predict}/fold{fold_no}/train_labels.csv').reset_index(drop=True)

                model_saving_folder = f'{model_folder}/{label_to_predict}/{model_type}/1_models/fold_{fold_no}'
                if(not os.path.exists(model_saving_folder)):
                    os.makedirs(model_saving_folder)
                    
                results_saving_folder = f'{results_folder}/{label_to_predict}/{model_type}/1_models/fold_{fold_no}'
                if(not os.path.exists(results_saving_folder)):
                    os.makedirs(results_saving_folder)
                # make_dirs(model_saving_folder)
                # make_dirs(results_saving_folder)

                '''TODO gotta find out what features to use for each label before testing on new dataset'''
                features_to_keep = ast.literal_eval(all_features_to_keep[label_to_predict][model_type])
                print(features_to_keep)
                make_UQ_model(training_features, training_labels, model_saving_folder, label_to_predict, 1, features_to_keep, hyperparam_folder, model_type=model_type)
                # make_linear_regression_models_for_ensemble(training_features, training_labels, model_saving_folder, label_to_predict, num_models, features_to_keep, hyperparam_folder, model_type=model_type)

'''last cell in ipynb'''
def evaluate_models(model_types, labels_to_predict, model_folder, results_folder, Paper2_path, all_features_to_keep):
    '''Now we will evaluate the performance of the bagging models'''
    for model_type in model_types:
        print(f'MODEL TYPE = {model_type}')
        for label_to_predict in labels_to_predict:
            print(f'LABEL = {label_to_predict}')
            performance_data = []
            for fold_no in range(1,6):
                print(f'fold {fold_no}')
                
                #defining folders to get the models and to store the results
                model_saving_folder = f'{model_folder}/{label_to_predict}/{model_type}/1_models/fold_{fold_no}'
                results_saving_folder = f'{results_folder}/{label_to_predict}/{model_type}/1_models/fold_{fold_no}'
                
                #defining folders where the datasets are coming from (5-fold cv)
                test_features_path = Paper2_path + f'/5fold_datasets/{label_to_predict}/fold{fold_no}/test_features.csv'
                test_labels_path = Paper2_path + f'/5fold_datasets/{label_to_predict}/fold{fold_no}/test_labels.csv'
                train_features_path = Paper2_path + f'/5fold_datasets/{label_to_predict}/fold{fold_no}/train_features.csv'
                train_labels_path = Paper2_path + f'/5fold_datasets/{label_to_predict}/fold{fold_no}/train_labels.csv'

                #defining the features that each model used (since they vary with each model)
                features_to_keep = ast.literal_eval(all_features_to_keep[label_to_predict][model_type])
                
                #predicting the test and train sets with the bagging models
                test_r2, test_ensemble_predictions, test_ensemble_uncertanties, test_labels = Get_predictions_and_uncertainty_single_model(test_features_path, test_labels_path, model_saving_folder, results_saving_folder, features_to_keep, label_to_predict, model_type)
                train_r2, train_ensemble_predictions, train_ensemble_uncertanties, train_labels = Get_predictions_and_uncertainty_single_model(train_features_path, train_labels_path, model_saving_folder, results_saving_folder, features_to_keep, label_to_predict, model_type)

                #defining the residual errors of the predictions
                train_labels_arr = train_labels
                train_predictions_arr = np.array(train_ensemble_predictions)
                test_labels_arr = test_labels
                test_predictions_arr = np.array(test_ensemble_predictions)
                train_residuals = pd.Series(np.abs(train_labels_arr - train_predictions_arr))
                test_residuals = pd.Series(np.abs(test_labels_arr - test_predictions_arr))

                a = 0
                b = 0
                '''getting calibration factors *** linear'''
                # cf = CorrectionFactors(train_residuals, pd.Series(train_ensemble_uncertanties))
                # a, b = cf.nll()
                # print(f'a = {a} b = {b}')
                # calibrated_train_uncertainties = pd.Series(a * np.array(train_ensemble_uncertanties) + b, name='train_model_errors')
                # calibrated_test_uncertainties = pd.Series(a * np.array(test_ensemble_uncertanties) + b, name='test_model_errors')
                
                '''getting calibration factors *** Nonlinear'''
                # a, b = get_calibration_factors(train_residuals, train_ensemble_uncertanties)
                # print(f'a = {a} b = {b}')
                # calibrated_train_uncertainties = pd.Series(a * (train_ensemble_uncertanties**((b/2) + 1)), name='train_model_errors')
                # calibrated_test_uncertainties = pd.Series(a * (test_ensemble_uncertanties**((b/2) + 1)), name='test_model_errors')

                '''
                Calculating and plotting performance metrics as outlined in section 2.3 of Tran et al. (https://dx.doi.org/10.1088/2632-2153/ab7e1a)
                
                Models should be compared in terms of 
                1st - accuracy (R^2) 
                2nd - calibration (miscalibration area)
                3rd - sharpness
                4th - dispersion
                '''
                #CALIBRATION plots and miscalibration area
                #This tells us how 'honest' our uncertainty values are. A perfect calibration plot would mean for a given confidence interval in our prediction
                #(say 90%), we can expect with 90% certainty that the true value falls within that confidence interval.
                miscalibration_area, calibration_error = make_calibration_plots(model_type, test_predictions_arr, test_labels_arr, test_ensemble_uncertanties, results_saving_folder)
                #SHARPNESS plots and value
                #Models can be calibrated, but all have very dull uncertainty values (they all have large uncertainties). To ensure UQ is meaningful, models
                #should a be sharp (i.e. uncertainties should be as small as possible.)
                #Sharpness is essentially calculated as the average of predicted standard deviations. #COMMENT Low sharpness values are better.
                stdevs = np.array(test_ensemble_uncertanties)/2 #right now, i multiply the stds by 2 to make it look better in parity plots... but this needs the raw std.
                sharpness, dispersion = plot_sharpness_curve(stdevs, results_saving_folder)
                #DISPERSION value
                #Models can be calibrated and sharp, but even so, if they are all similar uncertainties, then this does not tell us much. To ensure more 
                #meaningful UQ, having a large dispersion of uncertainties is valuable. 
                #Dispersion is calculated using equation 4 of the paper, which is called the coefficient of variation (Cv). #COMMENT High dispersion (Cv) values are better.
                
                '''using their library to make an rve plot'''
                # train_intercept, train_slope, CAL_train_intercept, CAL_train_slope, train_intercept, test_slope, CAL_test_intercept, CAL_test_slope = make_RVE_plots(label_to_predict, model_type, test_ensemble_predictions, test_ensemble_uncertanties, test_labels, train_ensemble_predictions, train_ensemble_uncertanties, train_labels, results_saving_folder, num_bins=15)
                
                '''collecting the performance data from this model'''
                # performance_data.append([15, fold_no, train_r2, test_r2, a, b, train_intercept, train_slope, CAL_train_intercept, CAL_train_slope, train_intercept, test_slope, CAL_test_intercept, CAL_test_slope, miscalibration_area, calibration_error])
                performance_data.append([fold_no, train_r2, test_r2, miscalibration_area, calibration_error, sharpness, dispersion])
                
            # columns = ['num bins', 'fold_no', 'train R2', 'test R2',  'a', 'b', 'train_intercept', 'train_slope', 'CAL_train_intercept', 'CAL_train_slope', 'train_intercept', 'test_slope', 'CAL_test_intercept', 'CAL_test_slope', 'miscal_area', 'cal_error']
            columns = ['fold_no', 'train R2', 'test R2', 'miscal_area', 'cal_error', 'sharpness', 'dispersion']
            df = pd.DataFrame(columns=columns)
            for row in performance_data:
                df.loc[len(df)] = row
            average_row = df.mean()
            df = df.append(average_row, ignore_index=True)
                
            results_saving_folder = f'{results_folder}/{label_to_predict}/{model_type}/1_models'
            df.to_csv(results_saving_folder + f'/{label_to_predict}_{model_type}_1results.csv', index=False)
            
            
          
            
def add_cluster_one_hot_vectors(data, num_clusters, cluster_type='init_cluster'):
    """
    Performs k-means clustering on the specified coordinates in the DataFrame,
    adds the cluster as one-hot vectors to the DataFrame.
    
    Parameters:
    - data: pandas DataFrame containing the dataset.
    - num_clusters: int, number of clusters to use in k-means.
    - cluster_type: str, either 'init_cluster' or 'impact_cluster' to specify which
      coordinates to cluster on.
    """
    if cluster_type == 'init_cluster':
        coordinates = ['init x', 'init y', 'init z']
    elif cluster_type == 'impact_cluster':
        coordinates = ['impact site x', 'impact site y', 'impact site z']
    else:
        raise ValueError("cluster_type must be either 'init_cluster' or 'impact_cluster'")
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data[coordinates])
    data[cluster_type] = kmeans.labels_

    # One-hot encode the cluster labels
    onehot_encoder = OneHotEncoder(sparse=False)
    cluster_labels_onehot = onehot_encoder.fit_transform(data[[cluster_type]])
    
    # Add the one-hot encoded vectors back to the dataframe
    columns = [f'{cluster_type}_{i}' for i in range(num_clusters)]
    for i, column in enumerate(columns):
        data[column] = cluster_labels_onehot[:, i]

    # Optionally, drop the original cluster label column
    data.drop([cluster_type], axis=1, inplace=True)
    
    return data


def visualize_clusters(data, cluster_type):

       # Now, plotting
       cluster_columns = [col for col in data.columns if f'{cluster_type}_' in col]
       data[cluster_type] = data[cluster_columns].idxmax(axis=1)

       # Extract the cluster number from the column names (assuming the format 'init_cluster_X')
       data[cluster_type] = data[cluster_type].apply(lambda x: int(x.split('_')[-1]))

       # Now, you can plot
       plt.figure(figsize=(10, 8))
       plt.scatter(data['init x'], data['init y'], c=data[cluster_type], cmap='viridis', alpha=0.7)
       plt.title('Initial Position Clusters')
       plt.xlabel('Initial X Position')
       plt.ylabel('Initial Y Position')
       plt.colorbar(label='Cluster')
       plt.grid(True)
       plt.show()
       
       
       
       
       