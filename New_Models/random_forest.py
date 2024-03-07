from linear_regression import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from Cluster_coordinates import *


'''
splits the data into 5 different k-folds of test and training sets
then runs GPR on each of the training sets
then evaluates the models based on their respective test sets.
'''
def Kfold_RF_Regression(full_dataset, full_dataset_labels, important_features, saving_folder, label_to_predict, save_data=True, num_training_points=False): #TODO change title for different models
    # correlated_featureset, raw_images, full_dataset_labels = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.01)

    #full_dataset = remove_ABAQUS_features(full_dataset)
    models = []
    
    rnge = range(1, len(full_dataset)+1)
    kf5 = KFold(n_splits=5, shuffle=True)
    fold_no = 1
    for train_index, test_index in kf5.split(rnge):
        train_df = full_dataset.iloc[train_index]
        # train_images = raw_images[train_index]
        y_train = full_dataset_labels[train_index]
        test_df = full_dataset.iloc[test_index]
        # test_images = raw_images[test_index]
        y_test = full_dataset_labels[test_index]
        
        
        """ if we want to limit the number of training datapoints """
        if(not num_training_points == False):
            train_df.reset_index(drop=True, inplace=True)
            train_indicies = np.random.choice(np.arange(0, len(train_df)), size=num_training_points, replace=False)
            train_df = train_df.iloc[train_indicies]
            y_train = y_train[train_indicies]
        # Create the Lasso regression model
        # alpha = Regularization strength
        model =  RandomForestRegressor(max_depth=20, n_estimators=100, random_state=42)
        
        model.fit(train_df.to_numpy(), y_train)        
        
        y_pred_train  = model.predict(train_df.to_numpy())
        y_pred_test = model.predict(test_df.to_numpy())
        

        if(save_data):
            save_model(model, fold_no, saving_folder, model_type=f'RF') #TODO change model type for different models
            # collect_and_save_metrics(y_test, y_pred_test, train_df.__len__(), len(train_df.columns), full_dataset.columns.to_list(), fold_no, saving_folder)
            collect_and_save_metrics(y_train, y_pred_train, y_test, y_pred_test, list(train_df.columns), fold_no, saving_folder)
            #plot_test_predictions_heatmap(y_test, y_pred_test, y_pred_test_std, fold_no, saving_folder)
            parody_plot(y_test, y_pred_test, fold_no, saving_folder, label_to_predict, model_type=f'RF regression')
            
            
        models.append((model, y_test, test_df))
        fold_no += 1
        
        

def plot_hyperparameter_performance(opt, hyperparameter_name, saving_folder):
    '''
    Plots the performance of a model for each value of a specified hyperparameter.

    Parameters:
    opt (BayesSearchCV): The fitted BayesSearchCV object after running the optimization.
    hyperparameter_name (str): The name of the hyperparameter to plot.
    '''
    
    # Extract the scores from the optimization results
    results = opt.cv_results_
    scores = results['mean_test_score']
    
    # Extract the hyperparameter values tried during optimization
    hyperparameter_values = [
        params[hyperparameter_name] for params in results['params']
    ]
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.scatter(hyperparameter_values, scores, color='red', alpha=0.4)
    # plt.plot(hyperparameter_values, scores, color='lightblue', linestyle='--', marker='o')

    # Label the plot
    plt.title(f'Performance for different values of hyperparameter: {hyperparameter_name}')
    plt.xlabel(hyperparameter_name)
    plt.ylabel('Validation R^2')
    
    # Show the plot
    plt.grid(True)
    # plt.show()
    plt.savefig(f'{saving_folder}/{hyperparameter_name}.png')
    plt.close()


def do_bayesian_optimization_RF(feature_df, label_df, num_tries=100, saving_folder='/Users/jakehirst/Desktop'):
    '''
    example how to run this code:
    
    when predicting impact site y
    top_3_features = [ 'avg_prop_speed * crack len',
                        'avg_prop_speed * init y',
                        'avg_prop_speed * linearity']

    labels_to_predict = ['height', 'impact site x', 'impact site y']

    # Generate some synthetic data for demonstration purposes
    df = pd.read_csv("/Volumes/Jake_ssd/OCTOBER_DATASET/feature_transformations_2023-10-28/height/HEIGHTALL_TRANSFORMED_FEATURES.csv")
    label_df = df.copy()[labels_to_predict]
    df = df.drop(labels_to_predict, axis=1)
    if(df.columns.__contains__('timestep_init')):
        df = df.drop('timestep_init', axis=1)


    label = 'impact site y'
    do_bayesian_optimization(df, label_df[label], 100)
    '''
    
    # '''zero centering and normalizing features'''
    # To zero-center, subtract the mean of each column from the column
    # df_centered = feature_df - feature_df.mean()
    # # To normalize, divide each column by its standard deviation
    # df_normalized = df_centered / df_centered.std()
    # feature_df = df_normalized
    
    X = feature_df.to_numpy()
    y = label_df.to_numpy()

    '''not needed since the BayesSearchCV already splits it into training and validation sets.'''
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter space for the Random Forest
    param_space = {
        'n_estimators': (1000, 5000),  #Higher for generalization, lower for overfitting
        'max_depth': (4,25),  #Higher for overfitting, lower for generaliztion
        'min_samples_split': (5, 30),  #Higher for generalization, lower for overfitting
        'min_samples_leaf': (1, 25), #Higher for generalization, lower for overfitting
        # 'max_features': (1, X_train.shape[1]), #Higher for overfitting, lower for generalization
        'max_features': (3,25), #Higher for overfitting, lower for generalization

    }

    # Create a RandomForestRegressor instance
    # rf = RandomForestRegressor(random_state=0)
    rf = RandomForestRegressor()

    # Wrap the model with BayesSearchCV
    opt = BayesSearchCV(rf, 
                        param_space, 
                        n_iter=num_tries, 
                        # random_state=0, 
                        cv=5,
                        verbose=3,
                        scoring='r2') #COMMENT cv=5 indicates a 5 fold cross validation

    # Run the Bayesian optimization
    opt.fit(X, y)

    best_index = opt.best_index_
    # Retrieve the mean test score for the best parameters
    best_average_score = opt.cv_results_['mean_test_score'][best_index]
    # Best parameter set found
    # print(f"\n$$$$$$$$$$$$ Results for RF predicting {label_df.name} $$$$$$$$$$$$")
    print(f"$$$$$$$$$$$$ Best parameters found: {opt.best_params_} $$$$$$$$$$$$")
    print(f"$$$$$$$$$$$$ Best average test score across 5-fold cv: {best_average_score} $$$$$$$$$$$$\n")
    
    if(not os.path.exists(saving_folder)): os.makedirs(saving_folder)
    hyperparameter_names = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
    for name in hyperparameter_names:
        plot_hyperparameter_performance(opt, name, saving_folder)

    
    with open(f'{saving_folder}/best_hyperparams.txt', 'w') as file:
        file.write(str(opt.best_params_))
    return opt


'''gets the best hyperparameters to use for GPR when predicting label_to_predict'''
def get_best_hyperparameters_RF(label_to_predict, hyperparameter_folder):
    import ast
    best_hp_path = f'{hyperparameter_folder}/{label_to_predict}/RF/best_hyperparams.txt'
    try:
        with open(best_hp_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print("File not found best_hyperparams.txt.")
    except Exception as e:
        print(f"An error occurred opening best_hyperparams.txt: {e}")
    converted_dict = dict(ast.literal_eval(content.removeprefix('OrderedDict')))
    depth = converted_dict['max_depth']
    features = converted_dict['max_features']
    samples_leaf = converted_dict['min_samples_leaf']
    samples_split = converted_dict['min_samples_split']
    estimators = converted_dict['n_estimators']
    
    return depth, features, samples_leaf, samples_split, estimators
    
    
    
    
    
    
    
'''all of these top 10 features were collected by first doing feature transformations and interactions, then by doing the backward feature selection technique. '''
'''when predicting height'''
# top_10_features = ['abs_val_sum_kink * mean thickness',
#                     'abs_val_sum_kink / avg_prop_speed',
#                     'abs_val_sum_kink / thickness_at_init',
#                     'abs_val_sum_kink + init y',
#                     'crack len + init y',
#                     'crack len (unchanged)',
#                     'dist btw frts + init y',
#                     'abs_val_sum_kink - avg_prop_speed',
#                     'avg_prop_speed - abs_val_sum_kink',
#                     'abs_val_sum_kink - init z',
#                     'init z - abs_val_sum_kink']

top_10_features = ['abs_val_sum_kink^2', 
                   'dist btw frts * max_kink',
                   'abs_val_sum_kink / init z', 
                   'abs_val_sum_kink / thickness_at_init',
                   'abs_val_mean_kink + abs_val_sum_kink', 
                   'abs_val_sum_kink + init y',
                   'abs_val_sum_kink + linearity',
                   'abs_val_sum_kink + mean thickness',
                   'dist btw frts + max_kink',
                   'abs_val_mean_kink - abs_val_sum_kink',
                   'abs_val_sum_kink - avg_prop_speed',
                   'abs_val_sum_kink - init z', 
                   'linearity - abs_val_sum_kink',
                   'abs_val_sum_kink - mean thickness',
                   'abs_val_sum_kink - thickness_at_init',
                   'thickness_at_init - abs_val_sum_kink']

'''when predicting impact site x'''
# top_10_features = [ 'dist btw frts * init x',
#                     'init x * linearity',
#                     'init x * max thickness',
#                     'init x * mean thickness',
#                     'init x * thickness_at_init',
#                     'abs_val_mean_kink + init x',
#                     'avg_prop_speed + init x',
#                     'init x + max thickness',
#                     'init x + mean thickness',
#                     'init x + thickness_at_init']
# top_10_features = ['init x^3', 
#                    'init x (unchanged)', 
#                    'init x * init z', 
#                    'init x * linearity',
#                    'init x * max thickness',
#                    'init x * mean thickness',
#                    'init x * thickness_at_init', 
#                    'init x + linearity', 
#                    'init x + max thickness']

'''when predicting impact site y'''
# top_10_features = [ 'avg_prop_speed * crack len',
#                     'avg_prop_speed * init y',
#                     'avg_prop_speed * linearity',
#                     'init y * mean thickness',
#                     'init y * thickness_at_init',
#                     'linearity * thickness_at_init',
#                     'avg_prop_speed + init y',
#                     'init y + linearity',
#                     'init y + thickness_at_init',
#                     'init y - abs_val_mean_kink']

labels_to_predict = ['height', 'impact site x', 'impact site y']

# Generate some synthetic data for demonstration purposes
# df = pd.read_csv("/Volumes/Jake_ssd/OCTOBER_DATASET/feature_transformations_2023-10-28/height/HEIGHTALL_TRANSFORMED_FEATURES.csv")
# df = pd.read_csv("/Volumes/Jake_ssd/feature_datasets/feature_transformations_2023-11-05/height/HEIGHTALL_TRANSFORMED_FEATURES.csv")
# label_df = df.copy()[labels_to_predict]
# df = df.drop(labels_to_predict, axis=1)
# if(df.columns.__contains__('timestep_init')):
#     df = df.drop('timestep_init', axis=1)

# top_10_df = df.copy()[top_10_features]

# clusters = cluster_coordinates(df, 'init x (unchanged)', 'init y (unchanged)', num_clusters=5)
# top_10_df = add_clusters_to_df(top_10_df, clusters)

# label = 'height'
# saving_folder = f'/Volumes/Jake_ssd/bayesian_optimization/RF/{label}'
# if(not os.path.exists(saving_folder)): os.makedirs(saving_folder)
# optimal_stuff = do_bayesian_optimization(top_10_df, label_df[label], 10, saving_folder= saving_folder)

# df = pd.read_csv('/Users/jakehirst/Desktop/UMAP_EMBEDDING.csv', index_col=0)
# label_df = df.copy()['height']
# df.drop(['height'], axis=1, inplace=True)
# optimal_stuff = do_bayesian_optimization(df, label_df, 200, saving_folder= '/Users/jakehirst/Desktop/UMAP_regression_trial')
# print('here')