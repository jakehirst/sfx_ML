import sys
sys.path.append('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/New_Models')

from Backward_feature_selection import *

all_labels = ['height', 'phi', 'theta', 
        'impact site x', 'impact site y', 'impact site z', 
        'impact site r', 'impact site phi', 'impact site theta']

with_or_without_transformations = 'without'
Paper2_path = f'/Volumes/Jake_ssd/Paper 2/{with_or_without_transformations}_transformations'
hyperparam_folder = Paper2_path + f'/bayesian_optimization_{with_or_without_transformations}_transformations'
hyperparam_folder = '/Volumes/Jake_ssd/bayesian_optimization_with_timestep_init'

'''the dataset we are useing's path'''
full_dataset_pathname = "/Volumes/Jake_ssd/Paper_1_results_no_feature_engineering/dataset/New_Crack_Len_FULL_OG_dataframe_2023_11_16.csv"
# full_dataset_pathname = "/Volumes/Jake_ssd/Paper_1_results_WITH_feature_engineering/dataset/feature_transformations_2023-11-16/height/HEIGHTALL_TRANSFORMED_FEATURES.csv"
'''where the results are stored'''
# results_folder = '/Volumes/Jake_ssd/Paper_1_results_no_feature_engineering/results'
results_folder = '/Volumes/Jake_ssd/Paper_1_results_with_timestep_init/results'
'''Where we will store the 5 fold datasets that we will make shortly'''
# data_folder = '/Volumes/Jake_ssd/Paper_1_results_no_feature_engineering/5fold_datasets'
# data_folder = '/Volumes/Jake_ssd/Paper_1_results_WITH_feature_engineering/5fold_datasets'
data_folder = Paper2_path + '/5fold_datasets'

if(not os.path.exists(data_folder)): os.makedirs(data_folder)

'''makes the 5 fold datasets'''
# make_5_fold_datasets(data_folder, full_dataset_pathname, remove_small_cracks=False, normalize=True)

dataset = pd.read_csv(full_dataset_pathname)
feature_set = dataset.drop(all_labels, axis=1)

'''get a list of all of the features we will use, and get rid of any that have "timestep_init" as welll as any that have "Unnamed: 0" in them. '''
all_features = feature_set.columns.to_list() #all of the features that will be considered
# all_features = [string for string in all_features if 'timestep_init' not in string] #
all_features = [string for string in all_features if 'Unnamed: 0' not in string]




'''start code'''


labels = ['height']


model_types = ['RF', 'GPR', 'linear', 'lasso', 'ridge',  'ANN', 'poly2']
# model_types = ['ANN', 'poly2']
model_types = ['poly2']

all_kept_features = {}
all_performances = {}
for label_to_predict in labels:
    all_kept_features[label_to_predict] = {}
    all_performances[label_to_predict] = {}
    for model_type in model_types:
        R2_across_kfolds = []
        for kfold in range(1,6):
                '''get datasets for this fold'''
                print(f'\n$$$$$$$$$$ NOW FOR PREDICTING {label_to_predict} WITH {model_type} fold {kfold}$$$$$$$$$$\n')
                train_features = pd.read_csv(data_folder + f'/{label_to_predict}/fold{kfold}/train_features.csv')
                test_features = pd.read_csv(data_folder + f'/{label_to_predict}/fold{kfold}/test_features.csv')
                train_labels = pd.read_csv(data_folder + f'/{label_to_predict}/fold{kfold}/train_labels.csv')
                test_labels = pd.read_csv(data_folder + f'/{label_to_predict}/fold{kfold}/test_labels.csv')
                train_features = train_features[all_features]
                test_features = test_features[all_features]
                
                '''train model'''
                model = train_model(model_type, train_features, train_labels, hyperparam_folder)
                
                '''get test predictions'''
                if(type(model) == ANNModel): 
                        with torch.no_grad():
                                model.eval()
                                test_preds = model(torch.FloatTensor(test_features.values).to(device)).numpy()
                                train_preds = model(torch.FloatTensor(train_features.values).to(device)).numpy()
                else:
                        test_preds = model.predict(test_features)
                        train_preds = model.predict(train_features)
                
                '''record performance'''
                test_r2 = r2_score(test_labels.to_numpy(), test_preds)
                test_MAE = mean_absolute_error(test_labels.to_numpy(), test_preds)
                train_r2 = r2_score(train_labels.to_numpy(), train_preds)
                train_MAE = mean_absolute_error(train_labels.to_numpy(), train_preds)
                
                R2_across_kfolds.append(test_r2)
                
                print(f'test_r2 = {test_r2}')
                print(f'train_r2 = {train_r2}')
                print(f'test_MAE = {test_MAE}')
                print(f'train_MAE = {train_MAE}')



                # kept_features, num_features_and_performances_TRAIN, num_features_and_performances_TEST = start_backward_feature_selection(data_folder, dataset, label_to_predict, model_type, all_labels, all_features, results_folder, num_features_to_keep=3)
                # plot_performances_vs_number_of_features(label_to_predict, model_type, results_folder, 'train')
                # plot_performances_vs_number_of_features(label_to_predict, model_type, results_folder, 'test')
        print('\n...next model type... ')
        print(f'AVERAGE TEST R2 FOR {model_type} = {np.mean(R2_across_kfolds)}')
            
print('here')