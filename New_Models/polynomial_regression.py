from linear_regression import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real
from sklearn.metrics import r2_score
import pandas as pd

'''
splits the data into 5 different k-folds of test and training sets
then runs GPR on each of the training sets
then evaluates the models based on their respective test sets.
'''
def Kfold_Polynomial_Regression(degree, full_dataset, full_dataset_labels, important_features, saving_folder, label_to_predict, save_data=True, num_training_points=False): #TODO change title for different models
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
            
        model = make_pipeline(PolynomialFeatures(degree),LinearRegression())
        model.fit(train_df.to_numpy(), y_train)
        
        y_pred_train  = model.predict(train_df.to_numpy())
        y_pred_test = model.predict(test_df.to_numpy())
        

        if(save_data):
            save_model(model, fold_no, saving_folder, model_type=f'poly_reg_degree') #TODO change model type for different models
            # collect_and_save_metrics(y_test, y_pred_test, train_df.__len__(), len(train_df.columns), full_dataset.columns.to_list(), fold_no, saving_folder)
            collect_and_save_metrics(y_train, y_pred_train, y_test, y_pred_test, list(train_df.columns), fold_no, saving_folder)
            #plot_test_predictions_heatmap(y_test, y_pred_test, y_pred_test_std, fold_no, saving_folder)
            parody_plot(y_test, y_pred_test, fold_no, saving_folder, label_to_predict, model_type=f'Polynomial Regression degree {degree}')
            
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




def do_bayesian_optimization_poly_reg(feature_df, label_df, num_tries=100, saving_folder='/Users/jakehirst/Desktop'):
    # Preprocessing as before
    df_centered = feature_df - feature_df.mean()
    df_normalized = df_centered / df_centered.std()
    feature_df = df_normalized
    
    X = feature_df.to_numpy()
    y = label_df.to_numpy()

    '''not needed since the BayesSearchCV already splits it into training and validation sets.'''
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter space for the ElasticNet within a polynomial regression
    param_space = {
        # 'polynomialfeatures__degree': Integer(2,2),  # Degree of the polynomial
        'elasticnet__alpha': Real(0.0001, 1.0, 'log-uniform'),  # Regularization strength
        # 'elasticnet__l1_ratio': Real(0.0, 1.0)  # Balance between L1 and L2 regularization
        'elasticnet__l1_ratio': Real(0.0, 0.95)  # Balance between L1 and L2 regularization
    }

    # Create a pipeline with PolynomialFeatures and ElasticNet
    polynomial_elasticnet = make_pipeline(PolynomialFeatures(degree=2), ElasticNet(random_state=0))

    # Wrap the model with BayesSearchCV
    opt = BayesSearchCV(polynomial_elasticnet, 
                        param_space, 
                        n_iter=num_tries, 
                        random_state=0, 
                        cv=5,
                        verbose=0)

    # Run the Bayesian optimization
    opt.fit(X, y)

    best_index = opt.best_index_
    # Retrieve the mean test score for the best parameters
    best_average_score = opt.cv_results_['mean_test_score'][best_index]
    # Best parameter set found
    print(f"\n$$$$$$$$$$$$ Results for Poly2 predicting {label_df.name} $$$$$$$$$$$$")
    print(f"$$$$$$$$$$$$ Best parameters found: {opt.best_params_} $$$$$$$$$$$$")
    print(f"$$$$$$$$$$$$ Best average test score across 5-fold cv: {best_average_score} $$$$$$$$$$$$\n")

    hyperparameter_names = ['elasticnet__alpha', 'elasticnet__l1_ratio']
    for name in hyperparameter_names:
        plot_hyperparameter_performance(opt, name, saving_folder)
        
    # Saving the best parameters
    with open(f'{saving_folder}/best_hyperparams.txt', 'w') as file:
        file.write(str(opt.best_params_))
    
    return opt


'''gets the best hyperparameters to use for poly2 when predicting label_to_predict'''
def get_best_hyperparameters_poly2(label_to_predict, hyperparameter_folder):
    import ast
    best_hp_path = f'{hyperparameter_folder}/{label_to_predict}/poly2/best_hyperparams.txt'
    try:
        with open(best_hp_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print("File not found best_hyperparams.txt.")
    except Exception as e:
        print(f"An error occurred opening best_hyperparams.txt: {e}")
    converted_dict = dict(ast.literal_eval(content.removeprefix('OrderedDict')))
    alpha = converted_dict['elasticnet__alpha']
    l1_ratio = converted_dict['elasticnet__l1_ratio']
    
    return alpha, l1_ratio