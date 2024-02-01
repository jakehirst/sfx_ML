from Pytorch_ANN import *
from GPR import *
from linear_regression import *
from ridge_regression import *
from polynomial_regression import *
from random_forest import *
from lasso_regression import *
import itertools


def get_feature_and_label_df(df, all_labels):
    '''removing any unwanted columns from the dataframe...'''
    df = df[df.columns[~df.columns.str.contains('timestep_init')]]
    if(df.columns.__contains__('Unnamed: 0')):
        df = df.drop('Unnamed: 0', axis=1)

    feature_df = df.drop(all_labels, axis=1)
    label_df = df[all_labels]

    '''preprocessing the data'''
    # First, zero-center the features by subtracting the mean
    feature_df_centered = feature_df - feature_df.mean()

    # Then, normalize the data to be between -10 and 10 by dividing by the half-range and multiplying by 10
    feature_df_range = (feature_df.max() - feature_df.min()) / 2
    feature_df_normalized = (feature_df_centered / feature_df_range) * 10
    feature_df = feature_df_normalized
    return feature_df, label_df


def train_model(model_type, label_to_predict, train_features, train_labels):
    if(model_type == 'ANN'):
        dropout, l1_lambda, l2_lambda, learning_rate = get_best_hyperparameters_ANN(label_to_predict=label_to_predict, hyperparameter_folder='/Volumes/Jake_ssd/bayesian_optimization')
        model = ANNModel(input_size=train_features.shape[1], output_size=1, dropout_rate=dropout).to(device)
        X_train_tensor = torch.FloatTensor(train_features.values).to(device)
        y_train_tensor = torch.FloatTensor(train_labels.values).to(device)
        model = train_ANN(model, X_train_tensor, y_train_tensor, loss_func='MAE', learning_rate=learning_rate, epochs=1000, l1_lambda=l1_lambda, l2_lambda=l2_lambda, patience=200, plot_losses=False) 

    elif(model_type == 'RF'):
        depth, features, samples_leaf, samples_split, estimators = get_best_hyperparameters_RF(label_to_predict=label_to_predict, hyperparameter_folder='/Volumes/Jake_ssd/bayesian_optimization')
        model =  RandomForestRegressor(max_depth=depth, max_features=features, 
                                       min_samples_leaf = samples_leaf, min_samples_split = samples_split, n_estimators=estimators, random_state=42)
    elif(model_type == 'linear'):
        model = LinearRegression() 
    elif(model_type == 'lasso'):
        a = 0.1
        model = Lasso(alpha=a)
    elif(model_type == 'ridge'):
        a = 0.1
        model = Ridge(alpha=a)
    elif(model_type == 'poly2'):
        alpha, l1_ratio = get_best_hyperparameters_poly2(label_to_predict=label_to_predict, hyperparameter_folder='/Volumes/Jake_ssd/bayesian_optimization')
        model = make_pipeline(PolynomialFeatures(degree=2), ElasticNet(alpha = alpha, l1_ratio = l1_ratio, random_state=0))
    elif(model_type == 'GPR'):
        c, length_scale, noise_level = get_best_hyperparameters_GPR(label_to_predict=label_to_predict, hyperparameter_folder='/Volumes/Jake_ssd/bayesian_optimization')
        kernel = ConstantKernel(constant_value=c) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        model = GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=25)

    if(model_type != 'ANN'):
        model.fit(train_features, train_labels)
        
    
    return model


def get_model_test_performance(model_type, trained_model, test_feats_df, test_labels_df):
    if(model_type == 'ANN'):
        predictions = pytorch_ANN_predict(trained_model, test_feats_df)
    else:
        predictions = trained_model.predict(test_feats_df)
    
    r2 = r2_score(test_labels_df.to_numpy(), predictions)
    mse = mean_squared_error(test_labels_df.to_numpy(), predictions)
    mae = mean_absolute_error(test_labels_df.to_numpy(), predictions)
    
    # parody_plot(test_labels_df.to_numpy(), predictions, 'none', 'none', test_labels_df.name, model_type)
    
    
    return r2, mae, mse 


def train_models_with_varying_number_datapoints(feature_df, label_df, labels_to_predict, model_types, all_features_to_keep, Num_training_datapoints_arr, fold_indicies):
    avg_performances_dict = {}
    for label in labels_to_predict:
        avg_performances_dict[label] = {}
        for model_type in model_types:
            avg_performances_dict[label][model_type] = {}
            num_datapoints_arr = []
            avg_R2_arr = []
            avg_MAE_arr = []
            avg_MSE_arr = []
            
            features_to_keep = all_features_to_keep[label][model_type]
            for num_datapoints in Num_training_datapoints_arr:
                
                r2_arr = []
                mae_arr = []
                mse_arr = []
                for fold in range(1,6):
                    trimmed_feature_df = feature_df[ast.literal_eval(features_to_keep)]
                    trimmed_label_df = label_df[label]
                    
                    train_feats_df = trimmed_feature_df.iloc[fold_indicies[fold]['train']]
                    train_labels_df = trimmed_label_df.iloc[fold_indicies[fold]['train']]
                    test_feats_df = trimmed_feature_df.iloc[fold_indicies[fold]['test']]
                    test_labels_df = trimmed_label_df.iloc[fold_indicies[fold]['test']]

                    '''get num_datapoints random samples without replacement'''
                    random_indices = random.sample(range(len(train_feats_df)), num_datapoints)
                    small_train_feats_df = train_feats_df.iloc[random_indices]
                    small_train_labels_df = train_labels_df.iloc[random_indices]
                    
                    trained_model = train_model(model_type, label, small_train_feats_df, small_train_labels_df)
                    
                    r2, mae, mse = get_model_test_performance(model_type, trained_model, test_feats_df, test_labels_df)
                    r2_arr.append(r2)
                    mae_arr.append(mae)
                    mse_arr.append(mse)
                
                '''append the average of the performances across all kfolds for this number of datapoints.'''
                avg_R2_arr.append(np.mean(r2_arr))
                avg_MAE_arr.append(np.mean(mae_arr))
                avg_MSE_arr.append(np.mean(mse_arr))
                num_datapoints_arr.append(num_datapoints)
            avg_performances_dict[label][model_type]['num_datapoints'] = num_datapoints_arr
            avg_performances_dict[label][model_type]['R2'] = avg_R2_arr
            avg_performances_dict[label][model_type]['Mean Absolute Error'] = avg_MAE_arr
            avg_performances_dict[label][model_type]['Mean Squared Error'] = avg_MSE_arr
    
    
    return avg_performances_dict


def plot_varying_number_of_datapoints(performances_dictionary, label, model_types, metric):
    # Define a list of markers and colors
    markers = ['o', 's', '^', 'D', '*', 'x', '+']  # Example markers
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Example colors
    # Create itertools cycle iterators for markers and colors
    marker_cycle = itertools.cycle(markers)
    color_cycle = itertools.cycle(colors)
    
    plt.figure(figsize=(12,10))
    for model_type in model_types:
        number_datapoints = performances_dictionary[label][model_type]['num_datapoints']
        metric_values = performances_dictionary[label][model_type][metric]
        plt.scatter(number_datapoints, metric_values, marker=next(marker_cycle), color=next(color_cycle), label=model_type)

    # Add legend to distinguish model types
    plt.legend()

    # Add labels and title if needed
    plt.xlabel("Number of Datapoints")
    plt.ylabel(metric)
    if(metric == 'R2'):
        plt.ylim((0, 1))
    plt.title(f"Scatter Plot of {metric} for varying number of datapoints\n Predicting {label}")
    plt.savefig(f"/Users/jakehirst/Desktop/writing class/paper_1_figures" + f'/Convergence_study_{label}_{model_type}_{metric}.png')
    # plt.show()
    
    return