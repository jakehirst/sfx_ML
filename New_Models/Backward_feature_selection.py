from GPR import *
from linear_regression import *
from polynomial_regression import *
from lasso_regression import *
from ridge_regression import *
# from CNN import *
from Pytorch_ANN import *
from random_forest import *
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from Feature_engineering import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import concurrent.futures
from scipy import stats
import ast
from sklearn.preprocessing import StandardScaler


'''gets all of the poorly correlated features. they have a pearson correlation that is less than the threshold'''
def get_low_correlation_features(dataset, label_to_predict, all_labels, threshold):
    features = dataset.drop(columns=all_labels, errors='ignore')
    if label_to_predict in features:
        features = features.drop(columns=label_to_predict)
    
    low_correlation_features = []
    
    for feature in features.columns:
        correlation, _ = pearsonr(dataset[feature], dataset[label_to_predict])
        if abs(correlation) < threshold:
            low_correlation_features.append(feature)
            
    return low_correlation_features

'''get all of the features that are not significant... ie they have a p-value that is greater than 0.05'''
def get_non_significant_features(dataframe, label_to_predict, all_labels):
    # Remove all labels except the one to predict
    cols_to_remove = set(all_labels) - {label_to_predict}
    dataframe = dataframe.drop(columns=cols_to_remove, errors='ignore')
    
    non_significant_feats = []
    for feature in dataframe.columns:
        if feature != label_to_predict:
            _, p_value = stats.pearsonr(dataframe[feature], dataframe[label_to_predict])
            if p_value >= 0.05:
                non_significant_feats.append(feature)
    return non_significant_feats

''' creates the 5-fold cross validation datasets for each label to predict in labels to predict in the path given. 
    If remove_small_cracks=True, then it removes all of the simulations that have cracks less than 10mm in length.'''
def make_5_fold_datasets(saving_folder, full_dataset_pathname, normalize=True, remove_small_cracks=False, label_chunks=None):
    all_labels = ['height', 'phi', 'theta', 
            'impact site x', 'impact site y', 'impact site z', 
            'impact site r', 'impact site phi', 'impact site theta']

    labels_to_predict = ['impact site x', 'impact site y', 'height']

    for label_to_predict in labels_to_predict:
        if(not os.path.exists(f'{saving_folder}/{label_to_predict}')): os.makedirs(f'{saving_folder}/{label_to_predict}')
        
        # if(not os.path.exists(f'{saving_folder}/{label_to_predict}')): os.mkdir(f'{saving_folder}/{label_to_predict}')
        # full_dataset_features, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=1)
        data = pd.read_csv(full_dataset_pathname)
        
        if(remove_small_cracks):
            '''removing all of the examples whose crack length does not exceed max_len'''
            print(f'original number of examples = {len(data)}')
            min_len = 10.0
            data = data[data['crack len (unchanged)'] >= min_len]
            print(f'examples left = {len(data)}')
        
        
        
        full_dataset_features = data.drop(all_labels, axis=1)
        if(normalize == True):
            # Zero-center the data
            data_centered = full_dataset_features - data.mean()
            # Normalize to the range [-1, 1]
            full_dataset_features = data_centered / data_centered.abs().max()
                        
        
        full_dataset_labels = data[label_to_predict]
        
        rnge = range(1, len(full_dataset_features)+1)
        kf5 = KFold(n_splits=5, shuffle=True)
        fold_no = 1
        for train_index, test_index in kf5.split(rnge):
            if(not os.path.exists(f'{saving_folder}/{label_to_predict}/fold{fold_no}')): os.mkdir(f'{saving_folder}/{label_to_predict}/fold{fold_no}')
            
            train_df = full_dataset_features.iloc[train_index]
            y_train = full_dataset_labels.iloc[train_index]
            test_df = full_dataset_features.iloc[test_index]
            y_test = full_dataset_labels.iloc[test_index]
            
            # y_train = pd.DataFrame(y_train, columns=[label_to_predict])
            # y_test = pd.DataFrame(y_test, columns=[label_to_predict])

            train_df.to_csv(f'{saving_folder}/{label_to_predict}/fold{fold_no}/train_features.csv', index=False)
            y_train.to_csv(f'{saving_folder}/{label_to_predict}/fold{fold_no}/train_labels.csv', index=False)
            test_df.to_csv(f'{saving_folder}/{label_to_predict}/fold{fold_no}/test_features.csv', index=False)
            y_test.to_csv(f'{saving_folder}/{label_to_predict}/fold{fold_no}/test_labels.csv', index=False)

            fold_no += 1


def append_metrics(current_features_metrics, model, test_features, test_labels, train_features, train_labels):
    if(type(model) == ANNModel): 
        with torch.no_grad():
            model.eval()
            test_preds = model(torch.FloatTensor(test_features.values).to(device)).numpy()
            train_preds = model(torch.FloatTensor(train_features.values).to(device)).numpy()
    else:
        test_preds = model.predict(test_features)
        train_preds = model.predict(train_features)
        
    current_features_metrics['test_R2'].append(r2_score(test_labels.to_numpy(), test_preds))
    current_features_metrics['test_MSE'].append(mean_squared_error(test_labels.to_numpy(), test_preds))
    current_features_metrics['test_MAE'].append(mean_absolute_error(test_labels.to_numpy(), test_preds))
    current_features_metrics['train_R2'].append(r2_score(train_labels.to_numpy(), train_preds))
    current_features_metrics['train_MSE'].append(mean_squared_error(train_labels.to_numpy(), train_preds))
    current_features_metrics['train_MAE'].append(mean_absolute_error(train_labels.to_numpy(), train_preds))
    
    return current_features_metrics

def train_model(model_type, train_features, train_labels, hyperparameter_folder):
    if(model_type == 'ANN'):
        dropout, l1_lambda, l2_lambda, learning_rate = get_best_hyperparameters_ANN(label_to_predict=train_labels.columns[0], hyperparameter_folder=hyperparameter_folder)
        model = ANNModel(input_size=train_features.shape[1], output_size=1, dropout_rate=dropout).to(device)
        X_train_tensor = torch.FloatTensor(train_features.values).to(device)
        y_train_tensor = torch.FloatTensor(train_labels.values).to(device)
        # dropout = 0.01 #TODO seeing if using optimal hyperparams works...
        # l1_lambda = 0.0
        # l2_lambda = 0.0
        # learning_rate = 0.01
        model = train_ANN(model, X_train_tensor, y_train_tensor, loss_func='MAE', learning_rate=learning_rate, epochs=1000, l1_lambda=l1_lambda, l2_lambda=l2_lambda, patience=200, plot_losses=False) 

    elif(model_type == 'RF'):
        depth, features, samples_leaf, samples_split, estimators = get_best_hyperparameters_RF(label_to_predict=train_labels.columns[0], hyperparameter_folder=hyperparameter_folder)
        #  OrderedDict([('max_depth', 2), ('max_features', 1), ('min_samples_leaf', 20), ('min_samples_split', 2), ('n_estimators', 2334)])
        model =  RandomForestRegressor(max_depth=depth, max_features=features, 
                                       min_samples_leaf = samples_leaf, min_samples_split = samples_split, n_estimators=estimators, random_state=42)
        # model = RandomForestRegressor()
    elif(model_type == 'linear'):
        model = LinearRegression() 
    elif(model_type == 'lasso'):
        a = 0.1
        model = Lasso(alpha=a)
    elif(model_type == 'ridge'):
        a = 0.1
        model = Ridge(alpha=a)
    elif(model_type == 'poly2'):
        alpha, l1_ratio = get_best_hyperparameters_poly2(label_to_predict=train_labels.columns[0], hyperparameter_folder=hyperparameter_folder)
        model = make_pipeline(PolynomialFeatures(degree=2), ElasticNet(alpha = alpha, l1_ratio = l1_ratio, random_state=0))
    elif(model_type == 'poly3'):
        degree = 3
        model = make_pipeline(PolynomialFeatures(degree),LinearRegression())
    elif(model_type == 'poly4'):
        degree = 4
        model = make_pipeline(PolynomialFeatures(degree),LinearRegression())  
    elif(model_type == 'GPR'):
        c, length_scale, noise_level = get_best_hyperparameters_GPR(label_to_predict=train_labels.columns[0], hyperparameter_folder=hyperparameter_folder)
        kernel = ConstantKernel(constant_value=c) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        # kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * RBF() + WhiteKernel(noise_level=1) #TODO trying this one out... using best hyperparams underfits when bagging
        # kernel = ConstantKernel(1.0) * RBF() + WhiteKernel(noise_level=1) #TODO trying this one out... using best hyperparams underfits when bagging
        # kernel = ConstantKernel() + Matern() + WhiteKernel(noise_level=1) #TODO trying kernel from paper
        # model = GaussianProcessRegressor(kernel=kernel, random_state=0, alpha=50, n_restarts_optimizer=25) 
        model = GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=25)
    
    # '''doing Z-score normalization on features (mean of 0, std of 1)'''
    # # Initialize the StandardScaler
    # scaler = StandardScaler()
    # train_features = pd.DataFrame(scaler.fit_transform(train_features), columns=train_features.columns)

    if(model_type != 'ANN'):
        model.fit(train_features, train_labels)
    return model


'''get the performance of the model when randomizing one feature at a time.
returns a dictionary with the keys being the features that it randomized, and the values beign the performances of the model when that feature was randomized.'''
def get_performances_randomizing_features(model, train_features, train_labels, percentage_to_remove=50):
    if(type(model) == ANNModel): 
        with torch.no_grad():
            model.eval()
            OG_train_preds = model(torch.FloatTensor(train_features.values).to(device)).numpy()
            # train_preds = model(torch.FloatTensor(train_features.values).to(device)).numpy()
    else:
        OG_train_preds = model.predict(train_features)
    og_train_r2 = r2_score(OG_train_preds,train_labels)
    
    all_features = train_features.columns.tolist()
    features_and_r2_differences = {}
    for feature in all_features:
        randomized_train_features = train_features.copy()
        randomized_train_features[feature] = train_features[feature].sample(frac=1).reset_index(drop=True)
        if(type(model) == ANNModel):
            with torch.no_grad():
                model.eval()
                train_preds = model(torch.FloatTensor(randomized_train_features.values).to(device)).numpy()
        else:
            train_preds = model.predict(randomized_train_features)
        new_train_r2 = r2_score(train_preds,train_labels)
        
        r2_diff = new_train_r2 - og_train_r2
        features_and_r2_differences[feature] = r2_diff
    # features_and_r2_differences = dict(sorted(features_and_r2_differences.items(), key=lambda item: item[1], reverse=True))
    
    '''get the bottom contributing features'''
    # num_items = int(len(features_and_r2_differences) * (percentage_to_remove / 100))
    # features_to_remove = list(features_and_r2_differences.keys())[:num_items]
    
    return features_and_r2_differences

def plot_performances_vs_number_of_features(label_to_predict, model_type, results_folder, train_or_test):
    filepath = f'{results_folder}/{label_to_predict}/{model_type}/performances/{train_or_test}_performances.csv'
    df = pd.read_csv(filepath)
    # Extract the number of features (from the index column)
    num_features = df.iloc[:, 0].values

    # Scatter plot each fold's performance in the same color
    for col in df.columns[1:]:
        plt.scatter(num_features, df[col], color='blue', alpha=0.7)

    # Scatter plot the mean performance in a different color
    mean_performance = df.iloc[:, 1:].mean(axis=1)
    plt.scatter(num_features, mean_performance, color='red', label='Mean R2')

    # Set plot labels and title
    plt.ylim((0, 1))
    plt.xlabel('Number of features')
    plt.ylabel(r'R$^2$ Performance')  # using LaTeX syntax for superscript
    plt.title(f'{train_or_test} set \nR2 for varying number of features')
    plt.legend()
    

    # Display the plot
    fig_path = filepath.removesuffix(f'{train_or_test}_performances.csv')
    plt.savefig(fig_path + f'{train_or_test}_performance_plot.png')
    plt.close()
    return

'''uses the dataset to train the model_type model, then removes the bottom 10% important features. Then trains the model again, and repeats the process until 
    there are only num_features_to_keep features left. returns those features and the performances (R^2) on the test and training sets.'''
def start_backward_feature_selection(folds_data_folder, full_dataset, label_to_predict, model_type, all_labels, all_features, results_folder, num_features_to_keep, hyperparameter_folder):
    '''remove all of the features that do not have a p-value of less than 0.05 with the label. this is the first filter... '''
    non_significant_features = get_non_significant_features(full_dataset, label_to_predict, all_labels)
    kept_features = [item for item in all_features if item not in non_significant_features]
    '''remove all of the features that do not have a correlation above 0.2 with the label. this is the second filter... '''
    low_correlated_features = get_low_correlation_features(full_dataset[kept_features + all_labels], label_to_predict, all_labels, 0.2)
    kept_features = [item for item in kept_features if item not in low_correlated_features]
    
    all_feature_metrics = {}
    
    all_kept_features = []

    num_features_and_performances_TRAIN = {}
    num_features_and_performances_TEST = {}
    num_features_and_MSE_TRAIN  = {}
    num_features_and_MSE_TEST = {}
    num_features_and_MAE_TRAIN = {}
    num_features_and_MAE_TEST = {}
    
    performance_results_path = f'{results_folder}/{label_to_predict}/{model_type}/performances'
    if(os.path.exists(f'{performance_results_path}/features_kept.csv')):
        features_remaining_df = pd.read_csv(f'{performance_results_path}/features_kept.csv', index_col=0)
        kept_features = ast.literal_eval(features_remaining_df['features remaining'].iloc[-1])
        all_kept_features = [(num, ast.literal_eval(features)) for num, features in zip(features_remaining_df['Num features remaining'], features_remaining_df['features remaining'])]

        df = pd.read_csv(f'{performance_results_path}/train_performances.csv')
        num_features_and_performances_TRAIN = {df['Unnamed: 0'][i]: df.loc[i, 'fold0':'fold4'].tolist() for i in df.index}
        df = pd.read_csv(f'{performance_results_path}/test_performances.csv')
        num_features_and_performances_TEST = {df['Unnamed: 0'][i]: df.loc[i, 'fold0':'fold4'].tolist() for i in df.index}
        df = pd.read_csv(f'{performance_results_path}/train_MSE.csv')
        num_features_and_MSE_TRAIN = {df['Unnamed: 0'][i]: df.loc[i, 'fold0':'fold4'].tolist() for i in df.index}
        df = pd.read_csv(f'{performance_results_path}/test_MSE.csv')
        num_features_and_MSE_TEST = {df['Unnamed: 0'][i]: df.loc[i, 'fold0':'fold4'].tolist() for i in df.index}
        df = pd.read_csv(f'{performance_results_path}/train_MAE.csv')
        num_features_and_MAE_TRAIN = {df['Unnamed: 0'][i]: df.loc[i, 'fold0':'fold4'].tolist() for i in df.index}
        df = pd.read_csv(f'{performance_results_path}/test_MAE.csv')
        num_features_and_MAE_TEST = {df['Unnamed: 0'][i]: df.loc[i, 'fold0':'fold4'].tolist() for i in df.index}
        
        # print('make current_features_metrics, all_kept_features, and kept_features here with previous data.')

    '''keep taking features out until the proper number of features to keep is achieved.'''
    while(len(kept_features) > num_features_to_keep):  
        '''make sure to save the metrics for this set of features'''
        current_features_metrics = {'test_R2':  [],
                                    'test_MSE': [],
                                    'test_MAE': [],
                                    'train_R2':  [],
                                    'train_MSE': [],
                                    'train_MAE': []}
        all_kept_features.append((len(kept_features), kept_features))
            
        
        
        kfold_performances = {feature: [] for feature in kept_features}
        for kfold in range(1,6):
            print(f'Working on fold {kfold}')
            train_features = pd.read_csv(folds_data_folder + f'/{label_to_predict}/fold{kfold}/train_features.csv')
            test_features = pd.read_csv(folds_data_folder + f'/{label_to_predict}/fold{kfold}/test_features.csv')
            train_labels = pd.read_csv(folds_data_folder + f'/{label_to_predict}/fold{kfold}/train_labels.csv')
            test_labels = pd.read_csv(folds_data_folder + f'/{label_to_predict}/fold{kfold}/test_labels.csv')
            train_features = train_features[kept_features]
            test_features = test_features[kept_features]
            
            model = train_model(model_type, train_features, train_labels, hyperparameter_folder)
            current_features_metrics = append_metrics(current_features_metrics, model, test_features, test_labels, train_features, train_labels)
            randomizing_performances = get_performances_randomizing_features(model, train_features, train_labels, percentage_to_remove=50)
            for key, value in randomizing_performances.items():
                kfold_performances[key].append(value)
            tf.keras.backend.clear_session()
        
        '''keep track of the performance on the training set and then number of features'''
        num_features_and_performances_TRAIN[len(kept_features)] = current_features_metrics['train_R2']
        num_features_and_performances_TEST[len(kept_features)] = current_features_metrics['test_R2']
        num_features_and_MSE_TRAIN[len(kept_features)] = current_features_metrics['train_MSE']
        num_features_and_MSE_TEST[len(kept_features)] = current_features_metrics['test_MSE']
        num_features_and_MAE_TRAIN[len(kept_features)] = current_features_metrics['train_MAE']
        num_features_and_MAE_TEST[len(kept_features)] = current_features_metrics['test_MAE']

        # num_features_and_performances[len(kept_features)] = {'train': current_features_metrics['train_R2'], 'test': current_features_metrics['test_R2']}
        '''average the differences in performance for each randomized feature across the 5folds'''
        average_kfold_performances = {key: sum(value)/len(value) if value else None for key, value in kfold_performances.items()}

        '''remove 20% of the features that have the least effect on the predictions until theres only a little left.'''
        average_kfold_performances = dict(sorted(average_kfold_performances.items(), key=lambda item: item[1], reverse=True))
        if(len(average_kfold_performances) > 20):#COMMENT THIS IS WHERE THE NUMBER OF FEATURES REMOVED IS DEFINED
            how_many_to_remove = int(np.ceil(len(average_kfold_performances) * 0.10))
            # how_many_to_remove = 1
        else: 
            # how_many_to_remove = len(average_kfold_performances) - num_features_to_keep
            how_many_to_remove = 1

            
        keys_to_remove = list(average_kfold_performances.keys())[:how_many_to_remove]
        print(f'\nremoving these features: \n{keys_to_remove}')
        kept_features = [value for value in kept_features if value not in keys_to_remove]
        
        
        
        '''save the performances for each number of features'''
        train_R2_df = pd.DataFrame(num_features_and_performances_TRAIN).T
        test_R2_df = pd.DataFrame(num_features_and_performances_TEST).T
        train_MSE_df = pd.DataFrame(num_features_and_MSE_TRAIN).T
        test_MSE_df = pd.DataFrame(num_features_and_MSE_TEST).T
        train_MAE_df = pd.DataFrame(num_features_and_MAE_TRAIN).T
        test_MAE_df = pd.DataFrame(num_features_and_MAE_TEST).T

        # Rename columns
        new_columns = ['fold' + str(i) for i in range(train_R2_df.shape[1])]
        train_R2_df.columns = new_columns
        test_R2_df.columns = new_columns
        train_MSE_df.columns = new_columns
        test_MSE_df.columns = new_columns
        train_MAE_df.columns = new_columns
        test_MAE_df.columns = new_columns
        
        
        features_results_path = f'{results_folder}/{label_to_predict}/{model_type}'
        if(not os.path.exists(performance_results_path)): os.makedirs(performance_results_path)
        
        feature_df = pd.DataFrame(all_kept_features, columns=['Num features remaining', 'features remaining'])

        feature_df.to_csv(f'{performance_results_path}/features_kept.csv')
        train_R2_df.to_csv(f'{performance_results_path}/train_performances.csv')
        test_R2_df.to_csv(f'{performance_results_path}/test_performances.csv' )
        train_MSE_df.to_csv(f'{performance_results_path}/train_MSE.csv')
        test_MSE_df.to_csv(f'{performance_results_path}/test_MSE.csv' )
        train_MAE_df.to_csv(f'{performance_results_path}/train_MAE.csv')
        test_MAE_df.to_csv(f'{performance_results_path}/test_MAE.csv' )
        
        kept_features_df = pd.DataFrame(kept_features, columns=['features'])
        kept_features_df.to_csv(features_results_path + f'/top_{num_features_to_keep}_features.csv')

    plot_performances_vs_number_of_features(label_to_predict, model_type, results_folder, 'train')
    plot_performances_vs_number_of_features(label_to_predict, model_type, results_folder, 'test')
    
    return kept_features, num_features_and_performances_TRAIN, num_features_and_performances_TEST

'''
gets the best combination of features when predicting the label for this type of model

folder = folder path where all backward feature selection results are stored (example = "/Volumes/Jake_ssd/Paper_1_results_WITH_feature_engineering/results")
label = label you are predicting (example = 'height')
model_type = model you are using (example = 'ANN')
min_features = 
max_features = 
'''
def get_best_features(folder, label, model_type, min_features, max_features):
    performances = pd.read_csv(folder + f'/{label}/{model_type}/performances/test_performances.csv')
    #only include rows that have less than 100 features and more than 10 features
    performances = performances.drop(performances[(performances['Unnamed: 0'] < min_features) | (performances['Unnamed: 0'] > max_features)].index)

    #calculate an average performance across all folds for each feature combination
    performances['average'] = performances[['fold0', 'fold1', 'fold2', 'fold3', 'fold4']].mean(axis=1)
    row_with_largest_average = performances['average'].idxmax() #get the row index that has the best performance across all folds

    #now get the features that were used for the best performing set of features
    features_kept = pd.read_csv(folder + f'/{label}/{model_type}/performances/features_kept.csv')
    best_feature_combination = features_kept.iloc[row_with_largest_average]['features remaining']    
    return best_feature_combination
            
            





