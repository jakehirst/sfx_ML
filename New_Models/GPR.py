from sklearn.model_selection import train_test_split
from prepare_data import *
from CNN import *
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, ExpSineSquared, DotProduct, Matern, RationalQuadratic
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, ExpSineSquared, DotProduct, Matern, RationalQuadratic
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, precision_score, confusion_matrix, recall_score, f1_score
import os
import pickle
import matplotlib.animation as animation
from Metric_collection import *
from sklearn import preprocessing

import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pandas as pd
from skopt import gp_minimize
from skopt.utils import use_named_args
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pandas as pd
from skopt import gp_minimize
from skopt.utils import use_named_args


'''
splits the full_Dataset raw_images and labels into training (which can later be broken down into training and validation sets) and test dataset
'''
def split_test_and_training_datasets(full_dataset, raw_images, full_dataset_labels):
    #setting aside a test dataset
    np.random.seed(6) #this should reset the randomness to the same randomness so that the test_indicies are the same throughout the tests
    test_indicies = np.random.choice(np.arange(0, len(full_dataset)), size=30, replace=False) #30 for the test dataset
    test_df = full_dataset.iloc[test_indicies]
    test_images = raw_images[test_indicies]
    y_test = full_dataset_labels[test_indicies]
    full_df = full_dataset.drop(test_indicies, axis=0)
    full_images = np.delete(raw_images, test_indicies, axis=0)
    full_dataset_labels = np.delete(full_dataset_labels, test_indicies, axis=0)
    return full_df, test_df, full_images, test_images, full_dataset_labels, y_test

'''
makes a parody plot of the predictions from GPR including the standard deviations
'''
# def parody_plot_with_std(y_test, y_pred_test, y_pred_test_std, fold_no, saving_folder, label_to_predict):
#     plt.figure()
#     plt.errorbar(y_test, y_pred_test, yerr=y_pred_test_std, fmt='o')
#     plt.plot(y_test, y_test, c='r')
#     plt.title('Fold ' + str(fold_no) + ' Gaussian Process Regression, R2=%.2f' % r2_score(y_test, y_pred_test))
#     plt.xlabel('Actual')
#     plt.ylabel('Predicted')
#     plt.savefig(saving_folder +  f'/{label_to_predict}_fold_{fold_no}_parody_plot.png')
#     # plt.show()
#     plt.close()
# def parody_plot_with_std(y_test, y_pred_test, y_pred_test_std, fold_no, saving_folder, label_to_predict):
#     plt.figure()
#     plt.errorbar(y_test, y_pred_test, yerr=y_pred_test_std, fmt='o')
#     plt.plot(y_test, y_test, c='r')
#     plt.title('Fold ' + str(fold_no) + ' Gaussian Process Regression, R2=%.2f' % r2_score(y_test, y_pred_test))
#     plt.xlabel('Actual')
#     plt.ylabel('Predicted')
#     plt.savefig(saving_folder +  f'/{label_to_predict}_fold_{fold_no}_parody_plot.png')
#     # plt.show()
#     plt.close()


''' 
Converts the xs ys and zs from the ABAQUS basis into the basis that is centered at the center of mass of the skull CM, and 
the z axis goes through the the ossification site of the RPA bone. These new basis vectors are defined above as Material X, Y, and Z.
Returns the x, y and z values in the new basis.
'''
def convert_coordinates_to_new_basis(Material_X, Material_Y, Material_Z, CM, xs, ys, zs):
    Transform = np.linalg.inv( np.matrix(np.transpose([Material_X,Material_Y,Material_Z])) )
    x2 = []
    y2 = []
    z2 = []
    for i in range(len(xs)):
        #the old point is in reference to the center of mass of the skull, defined as CM
        old_point = np.array([xs[i], ys[i], zs[i]]) - CM 
        #using transformation matrix to go from abaqus basis to the material basis
        new_point = np.array( Transform * np.matrix(old_point.reshape((3,1))) ).reshape((1,3)) 
        x2.append(new_point[0][0])
        y2.append(new_point[0][1])
        z2.append(new_point[0][2])
        


    # print("Coordinates in the target basis:")
    # print(f"x2: {x2}, y2: {y2}, z2: {z2}")
        
    return np.array(x2), np.array(y2), np.array(z2)

'''3d plot of RPA nodes and the predicted impact site in x and y direction. '''
'''3d plot of RPA nodes and the predicted impact site in x and y direction. '''
def plot_test_predictions_heatmap(full_dataset, labels_to_predict, all_labels, all_important_features, models_fold_to_pull, saving_folder):
    
    #material basis vectors for RPA bone
    Material_X = np.array([-0.87491124, -0.44839274,  0.18295974])
    Material_Y = np.array([ 0.23213791, -0.71986519, -0.65414532])
    Material_Z = np.array([ 0.42502036, -0.5298472,   0.7339071 ])
    #Center of mass of the RPA bone in abaqus basis
    CM = np.array([106.55,72.79,56.64])
    # #Ossification center of the RPA bone in abaqus basis
    OC = np.array([130.395996,46.6063,98.649696])
    
    parietal_node_location_df = pd.read_csv('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/parital_node_locations.csv')
    RPA_x = parietal_node_location_df['RPA nodes x']; RPA_y = parietal_node_location_df['RPA nodes y']; RPA_z = parietal_node_location_df['RPA nodes z']
    
    #converting the RPA node locations into Jimmy's reference frame
    RPA_x, RPA_y, RPA_z = convert_coordinates_to_new_basis(Material_X, Material_Y, Material_Z, CM, RPA_x, RPA_y, RPA_z)
    
    #loading previously trained models
    model_x = load_GPR_model(f'/Users/jakehirst/Desktop/model_results/MODEL_COMPARISONS/GPR_{labels_to_predict[0]}/GPR_model_fold{models_fold_to_pull[labels_to_predict[0]]}.sav')
    model_y = load_GPR_model(f'/Users/jakehirst/Desktop/model_results/MODEL_COMPARISONS/GPR_{labels_to_predict[1]}/GPR_model_fold{models_fold_to_pull[labels_to_predict[1]]}.sav')
    # model_z = load_GPR_model(f'/Users/jakehirst/Desktop/model_results/GPR_{labels_to_predict[2]}/GPR_model_fold{models_fold_to_pull[labels_to_predict[2]]}.sav')
    #predicting with previously trained models
    x_predictions, x_stds = model_x.predict(full_dataset[all_important_features[labels_to_predict[0]]].to_numpy(), return_std=True)
    y_predictions, y_stds = model_y.predict(full_dataset[all_important_features[labels_to_predict[1]]].to_numpy(), return_std=True)
    # z_predictions, z_stds = model_z.predict(remove_ABAQUS_features(full_dataset[all_important_features[labels_to_predict[2]]]).to_numpy(), return_std=True)
    x_true = full_dataset[labels_to_predict[0]].to_numpy()
    y_true = full_dataset[labels_to_predict[1]].to_numpy()
    z_true = full_dataset['impact site z'].to_numpy()



    for i in range(len(full_dataset)):
        '''3d plot of RPA nodes and the predicted x, y and z values in '''
        # Create a 3D figure
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
        
        ax.scatter(RPA_z, RPA_x, RPA_y, c='grey', alpha=0.025)
        x_dist = np.random.normal(x_predictions[i], x_stds[i], 5000)
        y_dist = np.random.normal(y_predictions[i], y_stds[i], 5000)

        point_size = 50
        z_dist = np.random.normal(z_true[i], 3, 5000)
        ax.scatter(z_dist, x_dist, y_dist, c='cyan', alpha=0.01)
        ax.scatter(z_true[i], x_predictions[i], y_predictions[i], c='blue', label='Mean predicted impact location', s=point_size)
        ax.scatter(z_true[i], x_true[i], y_true[i], c='orange', label='True impact location', s=point_size)
        

        ax.set_xlabel('Z')
        ax.set_ylabel('X')
        ax.set_zlabel('Y')
        ax.set_xlim3d(-60, 60)
        ax.set_ylim3d(-80, 80)
        ax.set_zlim3d(-60, 60)
        ax.set_title('GPR prediction for impact site in right parietal bone', fontweight='bold')
        ax.legend()
        
        normal_vector = np.array([0,1,0])
        # Calculate the azimuthal and polar angles
        azimuth = np.arctan2(normal_vector[1], normal_vector[0])
        polar = np.arccos(normal_vector[2])
        # Convert angles to degrees
        azimuth = np.degrees(azimuth)
        polar = np.degrees(polar)
        # # Set the camera direction using the angles (customizing a bit)
        ax.view_init(elev=-5, azim=azimuth + 270)
        # # Show the plot
        plt.savefig(saving_folder + f'prediction_{i}.png')
        # plt.show()
        
        """Creating animation gif that rotates 360 degrees"""
        # if(i == 2 or i == 4 or i == 6):
        #     # Define the update function for the animation
        #     def update(frame):
        #         # ax.view_init(elev=polar - 5 + frame*2, azim=azimuth + 180+frame*2)  # Adjust the viewing angle
        #         ax.view_init(elev=-5, azim=azimuth + 180+frame*2)  # Adjust the viewing angle

        #         # line.set_data(x[:frame], y[:frame])  # Update the plot data
        #         # line.set_3d_properties(z[:frame])
        #         # return line

        #     num_frames = 180
        #     # Create the animation
        #     ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50)
        #     f = f'/Users/jakehirst/Desktop/sfx/Presentations_and_Papers/USNCCM/figures/rotation_figure{i}.gif'
        #     writergif = animation.PillowWriter(fps=10) 
        #     ani.save(f, writer=writergif)
        #     print("next")
            
        plt.close()

    

    
    return
 

def save_GPR_model(model, fold_no, saving_folder):
    # Save the model to a file
    filename = saving_folder + f'/GPR_model_fold{fold_no}.sav'
    pickle.dump(model, open(filename, 'wb'))

def load_GPR_model(filepath):
    loaded_model = pickle.load(open(filepath, 'rb'))
    return loaded_model
    
''' Evaluates how accurate the uncertainty quantification is based on how many predictions lie within the first, second, and third confidence intervals. '''
def evaluate_uncertainty(y_pred, y_pred_std, y_true, train_or_test):
    intervals_and_percentages = {}
    intervals = [f'{train_or_test} 1 std or 68%', f'{train_or_test} 2 std or 95%', f'{train_or_test} 3 std or 99%']
    for num_stds in range(1,4):
        interval = intervals[num_stds-1]
        num_correct = 0
        for i in range(len(y_pred)):
            pred = y_pred[i]; std = y_pred_std[i]; true_value = y_true[i]
            
            if(pred + std*num_stds >= true_value and pred - std*num_stds <= true_value):
                num_correct += 1

        percentage_correct_in_this_interval = num_correct / len(y_true)
        intervals_and_percentages[interval] = percentage_correct_in_this_interval    
    
    return intervals_and_percentages, list(intervals_and_percentages.values())

'''
splits the data into 5 different k-folds of test and training sets
then runs GPR on each of the training sets
then evaluates the models based on their respective test sets.
'''
def Kfold_Gaussian_Process_Regression(full_dataset, full_dataset_labels, important_features, saving_folder, label_to_predict, save_data=True, num_training_points=False):
    # correlated_featureset, raw_images, full_dataset_labels = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.01)
    #full_dataset = remove_ABAQUS_features(full_dataset)
    models = []
    performances = []
    
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
        
        """ defining the covariance (kernel) function """
        kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3))  + WhiteKernel(noise_level=2, noise_level_bounds=(1e-2, 1e3)) #TODO experiment with the kernel... but this one seems to work.
        kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * RBF() + WhiteKernel(noise_level=1)
        # kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * RBF()
        # kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * RBF()
        # kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * RBF() + ConstantKernel(1.0) * ExpSineSquared()+ WhiteKernel(noise_level=1)
        # kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * ExpSineSquared()+ WhiteKernel(noise_level=1)
        
        model = GaussianProcessRegressor(kernel=kernel, random_state=0, alpha=50, n_restarts_optimizer=50)
        # model = GaussianProcessRegressor(kernel=kernel, random_state=0, alpha=50)
        
        """ fitting and making predictions based on non-scaled data """
        # model.fit(train_df.to_numpy(), y_train)
        
        # print(model.get_params())
        # y_pred_train, y_pred_train_std = model.predict(train_df.to_numpy(), return_std=True)
        # y_pred_test, y_pred_test_std = model.predict(test_df.to_numpy(), return_std=True)
        
        # train_uncertainty, train_uncertainty_values = evaluate_uncertainty(y_pred_train, y_pred_train_std, y_train, 'Train')
        # print(f'\ntrain uncertainty = \n {train_uncertainty}')
        # test_uncertainty, test_uncertainty_values = evaluate_uncertainty(y_pred_test, y_pred_test_std, y_test, 'Test')
        # print(f'\ntest uncertainty = \n {test_uncertainty}')

        
        # if(save_data):
        #     save_GPR_model(model, fold_no, saving_folder)            
        #     collect_and_save_metrics(y_train, y_pred_train, y_test, y_pred_test, list(train_df.columns), fold_no, saving_folder)
            
        #     # adding the uncertainty metrics to the metric data
        #     metric_data = pd.read_csv(saving_folder + f'/model_metrics_fold_{fold_no}.csv')
        #     metric_data = metric_data.assign(**train_uncertainty)
        #     metric_data = metric_data.assign(**test_uncertainty)
        #     metric_data.to_csv(saving_folder + f'/model_metrics_fold_{fold_no}.csv')
            
        #     #plot_test_predictions_heatmap(y_test, y_pred_test, y_pred_test_std, fold_no, saving_folder)
        #     parody_plot_with_std(y_test, y_pred_test, y_pred_test_std, fold_no, saving_folder, label_to_predict)
        # models.append((model, y_test, test_df, y_train, train_df))

        """ fitting and making predictions based on non-scaled data """



        """ fitting and making predictions based on scaled data """
        """ Scaling my data """
        scaler = preprocessing.StandardScaler().fit(full_dataset.to_numpy())
        X_scaled_train = scaler.transform(train_df.to_numpy())
        X_scaled_test = scaler.transform(test_df.to_numpy())
        
        model.fit(X_scaled_train, y_train)
        
        y_pred_train, y_pred_train_std = model.predict(X_scaled_train, return_std=True)
        y_pred_test, y_pred_test_std = model.predict(X_scaled_test, return_std=True)
        
        train_uncertainty, train_uncertainty_values = evaluate_uncertainty(y_pred_train, y_pred_train_std, y_train, 'Train')
        # print(f'\ntrain uncertainty = \n {train_uncertainty}')
        # print(f'\ntrain uncertainty = \n {train_uncertainty}')
        test_uncertainty, test_uncertainty_values = evaluate_uncertainty(y_pred_test, y_pred_test_std, y_test, 'Test')
        # print(f'\ntest uncertainty = \n {test_uncertainty}')
        # print(f'\ntest uncertainty = \n {test_uncertainty}')

        
        if(save_data):
            save_GPR_model(model, fold_no, saving_folder)            
            collect_and_save_metrics(y_train, y_pred_train, y_test, y_pred_test, list(train_df.columns), fold_no, saving_folder)
            
            # adding the uncertainty metrics to the metric data
            metric_data = pd.read_csv(saving_folder + f'/model_metrics_fold_{fold_no}.csv')
            metric_data = metric_data.assign(**train_uncertainty)
            metric_data = metric_data.assign(**test_uncertainty)
            metric_data.to_csv(saving_folder + f'/model_metrics_fold_{fold_no}.csv')
            # plot_test_predictions_heatmap(y_test, y_pred_test, y_pred_test_std, fold_no, saving_folder)
            parody_plot_with_std(y_test, y_pred_test, y_pred_test_std, fold_no, saving_folder, label_to_predict)
        
        Scaled_train_df = pd.DataFrame(X_scaled_train, columns=train_df.columns)
        Scaled_test_df = pd.DataFrame(X_scaled_test, columns=test_df.columns)

        models.append((model, y_test, Scaled_test_df, y_train, Scaled_train_df))

        """ fitting and making predictions based on scaled data """



        performances.append((r2_score(y_test, y_pred_test), mean_squared_error(y_test, y_pred_test)))
        fold_no += 1
        
    r2s = np.array([t[0] for t in performances])
    mse_s = np.array([t[1] for t in performances])
    print(f'mean r^2 = {r2s.mean()}')
    print(f'mean mse = {mse_s.mean()}')
    return models, performances, r2s, mse_s



def plot_hyperparameter_performance(opt, hyperparameter_name, saving_folder):
    '''
    Plots the mean TEST performance of a model for each value of a specified hyperparameter.

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





def do_bayesian_optimization_GPR(feature_df, label_df, num_tries=100, saving_folder='/Users/jakehirst/Desktop'):
    # Preprocessing as before
    # df_centered = feature_df - feature_df.mean()
    # df_normalized = df_centered / df_centered.std()
    # feature_df = df_normalized
    
    X = feature_df.to_numpy()
    y = label_df.to_numpy()

    '''not needed since the BayesSearchCV already splits it into training and validation sets.'''
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter space for the Gaussian Process Regression
    param_space = {
        'kernel__k1__k2__length_scale': Real(1e-9, 1e5),  # Length scale of the RBF kernel
        'kernel__k1__k1__constant_value': Real(0.1, 10.0),  # Scale of the Constant kernel
        'kernel__k2__noise_level': Real(1e-10, 1e2, 'log-uniform')  # Noise level for WhiteKernel
    }

    # Create a GaussianProcessRegressor with an RBF kernel plus a WhiteKernel for noise
    kernel = ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=20)

    # Wrap the model with BayesSearchCV
    opt = BayesSearchCV(gpr, 
                        param_space, 
                        n_iter=num_tries, 
                        random_state=0, 
                        cv=5,
                        verbose=3,
                        scoring= 'neg_mean_absolute_error')
                        # scoring='r2')

    # Run the Bayesian optimization
    opt.fit(X, y)

    best_index = opt.best_index_
    # Retrieve the mean test score for the best parameters
    best_average_score = opt.cv_results_['mean_test_score'][best_index]
    # Best parameter set found
    print(f"\n$$$$$$$$$$$$ Results for GPR predicting {label_df.name} $$$$$$$$$$$$")
    print(f"$$$$$$$$$$$$ Best parameters found: {opt.best_params_} $$$$$$$$$$$$")
    print(f"$$$$$$$$$$$$ Best average test score across 5-fold cv: {best_average_score} $$$$$$$$$$$$\n")
    
    
    hyperparameter_names = ['kernel__k1__k2__length_scale', 'kernel__k1__k1__constant_value', 'kernel__k2__noise_level']
    for name in hyperparameter_names:
        plot_hyperparameter_performance(opt, name, saving_folder)
    
    # Save the best parameters
    with open(f'{saving_folder}/best_hyperparams.txt', 'w') as file:
        file.write(str(opt.best_params_))

    # Note: The function 'plot_hyperparameter_performance' would need to be adapted 
    # to handle the Gaussian Process Regressor hyperparameters.
    
    return opt


'''gets the best hyperparameters to use for GPR when predicting label_to_predict'''
def get_best_hyperparameters_GPR(label_to_predict, hyperparameter_folder):
    import ast
    best_hp_path = f'{hyperparameter_folder}/{label_to_predict}/GPR/best_hyperparams.txt'
    try:
        with open(best_hp_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print("File not found best_hyperparams.txt.")
    except Exception as e:
        print(f"An error occurred opening best_hyperparams.txt: {e}")
    converted_dict = dict(ast.literal_eval(content.removeprefix('OrderedDict')))
    c = converted_dict['kernel__k1__k1__constant_value']
    length_scale = converted_dict['kernel__k1__k2__length_scale']
    noise_level = converted_dict['kernel__k2__noise_level']
    
    return c, length_scale, noise_level

def plot_hyperparameter_performance(opt, hyperparameter_name, saving_folder):
    '''
    Plots the mean TEST performance of a model for each value of a specified hyperparameter.

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





def do_bayesian_optimization_GPR(feature_df, label_df, num_tries=100, saving_folder='/Users/jakehirst/Desktop'):
    # Preprocessing as before
    # df_centered = feature_df - feature_df.mean()
    # df_normalized = df_centered / df_centered.std()
    # feature_df = df_normalized
    
    X = feature_df.to_numpy()
    y = label_df.to_numpy()

    '''not needed since the BayesSearchCV already splits it into training and validation sets.'''
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter space for the Gaussian Process Regression
    param_space = {
        'kernel__k1__k2__length_scale': Real(1e-9, 1e5),  # Length scale of the RBF kernel
        'kernel__k1__k1__constant_value': Real(0.1, 10.0),  # Scale of the Constant kernel
        'kernel__k2__noise_level': Real(1e-10, 1e2, 'log-uniform')  # Noise level for WhiteKernel
    }

    # Create a GaussianProcessRegressor with an RBF kernel plus a WhiteKernel for noise
    kernel = ConstantKernel(constant_value=1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=20)

    # Wrap the model with BayesSearchCV
    opt = BayesSearchCV(gpr, 
                        param_space, 
                        n_iter=num_tries, 
                        random_state=0, 
                        cv=5,
                        verbose=3,
                        scoring= 'neg_mean_absolute_error')
                        # scoring='r2')

    # Run the Bayesian optimization
    opt.fit(X, y)

    best_index = opt.best_index_
    # Retrieve the mean test score for the best parameters
    best_average_score = opt.cv_results_['mean_test_score'][best_index]
    # Best parameter set found
    print(f"\n$$$$$$$$$$$$ Results for GPR predicting {label_df.name} $$$$$$$$$$$$")
    print(f"$$$$$$$$$$$$ Best parameters found: {opt.best_params_} $$$$$$$$$$$$")
    print(f"$$$$$$$$$$$$ Best average test score across 5-fold cv: {best_average_score} $$$$$$$$$$$$\n")
    
    
    hyperparameter_names = ['kernel__k1__k2__length_scale', 'kernel__k1__k1__constant_value', 'kernel__k2__noise_level']
    for name in hyperparameter_names:
        plot_hyperparameter_performance(opt, name, saving_folder)
    
    # Save the best parameters
    with open(f'{saving_folder}/best_hyperparams.txt', 'w') as file:
        file.write(str(opt.best_params_))

    # Note: The function 'plot_hyperparameter_performance' would need to be adapted 
    # to handle the Gaussian Process Regressor hyperparameters.
    
    return opt


'''gets the best hyperparameters to use for GPR when predicting label_to_predict'''
def get_best_hyperparameters_GPR(label_to_predict, hyperparameter_folder):
    import ast
    best_hp_path = f'{hyperparameter_folder}/{label_to_predict}/GPR/best_hyperparams.txt'
    try:
        with open(best_hp_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print("File not found best_hyperparams.txt.")
    except Exception as e:
        print(f"An error occurred opening best_hyperparams.txt: {e}")
    converted_dict = dict(ast.literal_eval(content.removeprefix('OrderedDict')))
    c = converted_dict['kernel__k1__k1__constant_value']
    length_scale = converted_dict['kernel__k1__k2__length_scale']
    noise_level = converted_dict['kernel__k2__noise_level']
    
    return c, length_scale, noise_level

