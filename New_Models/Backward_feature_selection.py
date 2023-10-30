from GPR import *
from linear_regression import *
from polynomial_regression import *
from lasso_regression import *
from ridge_regression import *
from CNN import *
from random_forest import *
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from Feature_engineering import *
from Bagging_models import make_5_fold_datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import concurrent.futures
from scipy import stats

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
def make_5_fold_datasets(saving_folder, full_dataset_pathname, image_folder, normalize=True, remove_small_cracks=False):
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
    test_preds = model.predict(test_features)
    train_preds = model.predict(train_features)
    
    current_features_metrics['test_R2'].append(r2_score(test_labels.to_numpy(), test_preds))
    current_features_metrics['test_MSE'].append(mean_squared_error(test_labels.to_numpy(), test_preds))
    current_features_metrics['test_MAE'].append(mean_absolute_error(test_labels.to_numpy(), test_preds))
    current_features_metrics['train_R2'].append(r2_score(train_labels.to_numpy(), train_preds))
    current_features_metrics['train_MSE'].append(mean_squared_error(train_labels.to_numpy(), train_preds))
    current_features_metrics['train_MAE'].append(mean_absolute_error(train_labels.to_numpy(), train_preds))
    
    return current_features_metrics

def train_model(model_type, train_features, train_labels):
    if(model_type == 'ANN'):
        train_features, val_features, train_labels, val_labels = train_test_split(
            train_features, train_labels, test_size=0.2, random_state=42)
        model = make_1D_CNN_for_feature_selection(train_features, val_features, train_labels.to_numpy(), val_labels.to_numpy(), patience=100, max_epochs=1000, num_outputs=1, lossfunc='mean_squared_error')
    elif(model_type == 'RF'):
        # model =  RandomForestRegressor(max_depth=5, n_estimators=10000, random_state=42, max_features=5)
        model =  RandomForestRegressor(max_depth=5, n_estimators=1000, random_state=42, max_features=5)
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
        kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * RBF() + WhiteKernel(noise_level=1) # this one works well
        model = GaussianProcessRegressor(kernel=kernel, random_state=0, alpha=50, n_restarts_optimizer=25) 
    
    if(model != 'ANN'):
        model.fit(train_features.to_numpy(), train_labels.to_numpy()) 

    return model


'''get the performance of the model when randomizing one feature at a time.
returns a dictionary with the keys being the features that it randomized, and the values beign the performances of the model when that feature was randomized.'''
def get_performances_randomizing_features(model, train_features, train_labels, percentage_to_remove=50):
    OG_train_preds = model.predict(train_features)
    og_train_r2 = r2_score(OG_train_preds,train_labels)
    
    all_features = train_features.columns.tolist()
    features_and_r2_differences = {}
    for feature in all_features:
        randomized_train_features = train_features.copy()
        randomized_train_features[feature] = train_features[feature].sample(frac=1).reset_index(drop=True)
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
        plt.scatter(num_features, df[col], color='blue', alpha=0.7, label=col if "fold0" in col else "")

    # Scatter plot the mean performance in a different color
    mean_performance = df.iloc[:, 1:].mean(axis=1)
    plt.scatter(num_features, mean_performance, color='red', s=100, label='Mean Performance')

    # Set plot labels and title
    plt.xlabel('Number of Features')
    plt.ylabel(r'R$^2$ Performance')  # using LaTeX syntax for superscript
    plt.title('Performance by Number of Features')
    plt.legend(loc='best')

    # Display the plot
    fig_path = filepath.removesuffix(f'{train_or_test}_performances.csv')
    plt.savefig(fig_path + f'{train_or_test}_performance_plot.png')
    plt.close()
    return

'''uses the dataset to train the model_type model, then removes the bottom 10% important features. Then trains the model again, and repeats the process until 
    there are only num_features_to_keep features left. returns those features and the performances (R^2) on the test and training sets.'''
def start_backward_feature_selection(dataset, label_to_predict, model_type, all_labels, all_features, results_folder, num_features_to_keep):
    '''remove all of the features that do not have a p-value of less than 0.05 with the label. this is the first filter... '''
    non_significant_features = get_non_significant_features(dataset, label_to_predict, all_labels)
    kept_features = [item for item in all_features if item not in non_significant_features]
    '''remove all of the features that do not have a correlation above 0.2 with the label. this is the second filter... '''
    low_correlated_features = get_low_correlation_features(dataset[kept_features + all_labels], label_to_predict, all_labels, 0.2)
    kept_features = [item for item in kept_features if item not in low_correlated_features]
    # kept_features = kept_features[0:20] #TODO remove this...
    
    all_feature_metrics = {}

    num_features_and_performances_TRAIN = {}
    num_features_and_performances_TEST = {}

    '''keep taking features out until the proper number of features to keep is achieved.'''
    while(len(kept_features) > num_features_to_keep):  
        '''make sure to save the metrics for this set of features'''
        current_features_metrics = {'test_R2':  [],
                                    'test_MSE': [],
                                    'test_MAE': [],
                                    'train_R2':  [],
                                    'train_MSE': [],
                                    'train_MAE': []}
        
        kfold_performances = {feature: [] for feature in kept_features}
        for kfold in range(1,6):
            print(f'Working on fold {kfold}')
            train_features = pd.read_csv(data_folder + f'/{label_to_predict}/fold{kfold}/train_features.csv')
            test_features = pd.read_csv(data_folder + f'/{label_to_predict}/fold{kfold}/test_features.csv')
            train_labels = pd.read_csv(data_folder + f'/{label_to_predict}/fold{kfold}/train_labels.csv')
            test_labels = pd.read_csv(data_folder + f'/{label_to_predict}/fold{kfold}/test_labels.csv')
            train_features = train_features[kept_features]
            test_features = test_features[kept_features]
            
            model = train_model(model_type, train_features, train_labels)
            current_features_metrics = append_metrics(current_features_metrics, model, test_features, test_labels, train_features, train_labels)
            randomizing_performances = get_performances_randomizing_features(model, train_features, train_labels, percentage_to_remove=50)
            for key, value in randomizing_performances.items():
                kfold_performances[key].append(value)
            tf.keras.backend.clear_session()
        
        '''keep track of the performance on the training set and then number of features'''
        num_features_and_performances_TRAIN[len(kept_features)] = current_features_metrics['train_R2']
        num_features_and_performances_TEST[len(kept_features)] = current_features_metrics['test_R2']

        # num_features_and_performances[len(kept_features)] = {'train': current_features_metrics['train_R2'], 'test': current_features_metrics['test_R2']}
        '''average the differences in performance for each randomized feature across the 5folds'''
        average_kfold_performances = {key: sum(value)/len(value) if value else None for key, value in kfold_performances.items()}

        '''remove 20% of the features that have the least effect on the predictions until theres only a little left.'''
        average_kfold_performances = dict(sorted(average_kfold_performances.items(), key=lambda item: item[1], reverse=True))
        if(len(average_kfold_performances) > num_features_to_keep*1.5):
            how_many_to_remove = int(np.ceil(len(average_kfold_performances) * 0.10))
        else: 
            how_many_to_remove = len(average_kfold_performances) - num_features_to_keep
            
        keys_to_remove = list(average_kfold_performances.keys())[:how_many_to_remove]
        print(f'\nremoving these features: \n{keys_to_remove}')
        kept_features = [value for value in kept_features if value not in keys_to_remove]
        
        
        
    '''save the performances for each number of features'''
    train_df = pd.DataFrame(num_features_and_performances_TRAIN).T
    test_df = pd.DataFrame(num_features_and_performances_TEST).T
    # Rename columns
    new_columns = ['fold' + str(i) for i in range(train_df.shape[1])]
    train_df.columns = new_columns
    test_df.columns = new_columns
    performance_results_path = f'{results_folder}/{label_to_predict}/{model_type}/performances'
    features_results_path = f'{results_folder}/{label_to_predict}/{model_type}'
    if(not os.path.exists(performance_results_path)): os.makedirs(performance_results_path)
    
    train_df.to_csv(f'{performance_results_path}/train_performances.csv')
    test_df.to_csv(f'{performance_results_path}/test_performances.csv' )
    kept_features_df = pd.DataFrame(kept_features, columns=['features'])
    kept_features_df.to_csv(features_results_path + f'/top_{num_features_to_keep}_features.csv')

    plot_performances_vs_number_of_features(label_to_predict, model_type, results_folder, 'train')
    plot_performances_vs_number_of_features(label_to_predict, model_type, results_folder, 'test')
    
    return kept_features, num_features_and_performances_TRAIN, num_features_and_performances_TEST






data_folder = '/Volumes/Jake_ssd/Backward_feature_selection/5fold_datasets'
if(not os.path.exists(data_folder)): os.makedirs(data_folder)
full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_data/New_Crack_Len_FULL_OG_dataframe_2023_10_28.csv"
full_dataset_pathname = "/Volumes/Jake_ssd/OCTOBER_DATASET/feature_transformations_2023-10-28/height/HEIGHTALL_TRANSFORMED_FEATURES.csv"
image_folder = '/Users/jakehirst/Desktop/sfx/sfx_ML_data/images_sfx/new_dataset/Visible_cracks'

results_folder = '/Volumes/Jake_ssd/Backward_feature_selection/results'

'''only have to make the datasets once'''
# make_5_fold_datasets(data_folder, full_dataset_pathname, image_folder, remove_small_cracks=False)
all_labels = ['height', 'phi', 'theta', 
        'impact site x', 'impact site y', 'impact site z', 
        'impact site r', 'impact site phi', 'impact site theta']
'''
load the data
get a list of the features
go through each feature and train the model
only keep the best performing model

add a feature to the list of kept features
remove that feature from feature candidates
'''

label_to_predict = 'height'
label_to_predict = 'impact site x'
model_type = 'ANN'
model_type = 'RF'
dataset = pd.read_csv(full_dataset_pathname)
feature_set = dataset.drop(all_labels, axis=1)
# feature_candidates = feature_set.columns.to_list() #all of the features that will be considered
all_features = feature_set.columns.to_list() #all of the features that will be considered
all_features = [string for string in all_features if 'timestep_init' not in string]
# all_features = all_features[0:20]

labels = ['height', 'impact site x', 'impact site y']
labels = ['height']

model_types = ['RF', 'GPR', 'linear', 'lasso', 'ridge', 'poly2', 'poly3']
model_types = ['ANN']

all_kept_features = {}
all_performances = {}
for label_to_predict in labels:
    all_kept_features[label_to_predict] = {}
    all_performances[label_to_predict] = {}
    for model_type in model_types:
        print(f'\n$$$$$$$$$$ NOW FOR PREDICTING {label_to_predict} WITH {model_type} $$$$$$$$$$\n')

        kept_features, num_features_and_performances_TRAIN, num_features_and_performances_TEST = start_backward_feature_selection(dataset, label_to_predict, model_type, all_labels, all_features, results_folder, num_features_to_keep=10)

        print('\n...next model type... ')
            
print('here')
            
            
            





