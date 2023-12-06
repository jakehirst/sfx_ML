from time import process_time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold
import statistics
from sklearn import metrics
import random
from bayes_opt import BayesianOptimization
import time



#COMMENT works on GPU, but is faster on CPU for some reason...
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )

device = "cpu"


# Define the ANN Model
class ANNModel(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        num_nodes = 64
        super(ANNModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, num_nodes),
            nn.BatchNorm1d(num_nodes),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(num_nodes, num_nodes),
            nn.BatchNorm1d(num_nodes),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(num_nodes, num_nodes),
            nn.BatchNorm1d(num_nodes),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            
            nn.Linear(num_nodes, output_size)
        )

    def forward(self, x):
        return self.layers(x)


''' 
    Randomly splits the training tensors into train and validation tensors.
    Trains the ANN while keeping track of validation and train losses
    Reports score and can plot the losses over time if needed.
    
    N = number of examples
    F = number of features
    
    Model = ANNModel object
    X_train_tensor = torch.Tensor type. Is a tensor of the training features [N,F] in shape
    y_train_tensor = torch.Tensor type. Is a tensor of the training labels [N] in shape
    loss_func = either 'MAE' or 'MSE' since this is for regression
    l1_lambda = coefficient for L1 regularization
    l2_lambda = coefficient for L2 regularization
    
    All other inputs are self explainatory.
'''
def train_ANN(model, X_train_tensor, y_train_tensor, loss_func='MAE', learning_rate=0.001, epochs=1000, l1_lambda=0.0, l2_lambda=0.0, patience=200, batch_size=100, plot_losses=False, verbose=False):
    if loss_func == "MAE":
        criterion = nn.L1Loss()
    elif loss_func == "MSE":
        criterion = nn.MSELoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    '''splitting tensors into training and validation sets'''
    # Upper limit to train/val split indicies
    upper_limit = y_train_tensor.__len__()
    # Generate the two lists
    train_indexes = random.sample(range(upper_limit), int(upper_limit * 0.8))
    val_indexes = list(set(range(upper_limit)) - set(train_indexes))
    x_train = X_train_tensor[train_indexes]
    y_train = y_train_tensor[train_indexes]
    x_val = X_train_tensor[val_indexes]
    y_val = y_train_tensor[val_indexes]
    
    '''putting training dataset into loader (preps the batches)'''
    dataset_train = TensorDataset(x_train, y_train)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    epochs_needed = []


    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0
    patience_counter = 0
    best_loss = float('inf')

    train_losses = []
    val_losses = []
    
    '''beginning training'''
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batches = 0
        for batch_x, batch_y in loader_train:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.flatten(), batch_y)

            # Calculate L1 penalty (L1 regularization)
            l1_penalty = torch.tensor(0.).to(batch_x.device)
            for param in model.parameters():
                l1_penalty += torch.norm(param, 1)
            loss += l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        train_losses.append(total_loss / batches) 

        model.eval()
        with torch.no_grad():
            outputs_validation = model(x_val)
            val_loss = criterion(outputs_validation.flatten(), y_val).item()
            val_losses.append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            epochs_needed.append(epoch)
            print('patience reached')
            break
        if(verbose):
            print(f'epoch = {epoch}, train loss = {train_losses[-1]}, val loss = {val_losses[-1]}')
    
        # Evaluate
    with torch.no_grad():
        model.eval()
        # Move predictions to CPU for evaluation
        pred = model(x_val).cpu().numpy()
        y_compare = y_val.cpu().numpy()
        val_score = metrics.r2_score(y_compare, pred)
        train_pred = model(x_train).cpu().numpy() 
        y_compare_train = y_train.cpu().numpy()
        train_score = metrics.r2_score(y_compare_train, train_pred)
        print(f'End train R2 score = {train_score} validation R2 score = {val_score}')
    
    '''Load and return the best model (model with best validation score)'''
    model.load_state_dict(best_model)
    
    
    '''plot the loss for the validation and train set over all the epochs'''
    if(plot_losses):
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo-', label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'ro-', label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    return model



def test_model(model, X_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    return predictions

'''pytorch only works with tensors, so this is needed to preprocess the data for pytorch.'''
def prepare_data_for_pytorch(train_feats, test_feats, train_labels, test_labels):
    # X_train, X_val, y_train, y_val = train_test_split(train_feats, train_labels, test_size=0.20, random_state=42,shuffle=True )

    # Convert data to PyTorch tensors
    # X_train_tensor = torch.FloatTensor(torch.tensor(train_feats.values, dtype=torch.float32), torch.tensor(train_labels.values, dtype=torch.float32))
    X_train_tensor = torch.FloatTensor(train_feats.values).to(device)
    y_train_tensor = torch.FloatTensor(train_labels.values).to(device)
    # val_tensor = torch.FloatTensor(torch.tensor(X_val.values, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32))
    # test_dataset = TensorDataset(torch.tensor(test_feats.to_numpy(), dtype=torch.float32), torch.tensor(test_labels.values, dtype=torch.float32))
    # X_test_tensor = torch.FloatTensor(torch.tensor(test_feats.values, dtype=torch.float32), torch.tensor(test_labels.values, dtype=torch.float32))
    X_test_tensor = torch.FloatTensor(test_feats.values).to(device)
    y_test_tensor = torch.FloatTensor(test_labels.values).to(device)
    # DataLoader
    # train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=False)
    # val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def parody_plot(test_true, test_predictions, dataset=None, label='label unkown'):
    R2_score = metrics.r2_score(test_true, test_predictions)
    plt.figure(figsize=(8, 6))
    plt.scatter(test_true, test_predictions, color='blue', label='Predictions')
    plt.plot([min(test_true), max(test_true)], [min(test_true), max(test_true)], color='red', label='True Value Line')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{dataset} Set Parody Plot predicting {label}. R2 = {R2_score}')
    plt.xlim(min(test_true), max(test_true))
    plt.ylim(min(test_true), max(test_true))
    plt.legend()
    plt.show()
    return

'''Code used for Bayesian Hyperparameter optimization.'''
def do_bayesian_optimization(x_tensor, y_tensor, num_iter=200):
    '''This is the objective function for the bayesian opt. It trains the model and keeps track of the R2 after training.'''
    def evaluate_network(learning_rate=1e-3, 
                        batch_size = 100, 
                        epochs=1000, 
                        l1_lambda=0.001, 
                        l2_lambda=0.001, 
                        patience=200, 
                        dropout=0.0, loss_func='MAE'):
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = [(train_index, test_index) for train_index, test_index in kf.split(x_tensor)]

        
        # boot = StratifiedShuffleSplit(n_splits=SPLITS, test_size=0.2)
        mean_benchmark = []
        epochs_needed = []


        for train, test in splits: 
            '''splitting data into training and test tensors'''
            X_train_tensor = x_tensor[train]
            y_train_tensor = y_tensor[train]
            X_test_tensor = x_tensor[test]
            y_test_tensor = y_tensor[test]
            
            model = ANNModel(input_size=x_tensor.shape[1], 
                    output_size = 1, 
                    dropout_rate = dropout).to(device)
            
            if loss_func == "MAE":
                criterion = nn.L1Loss()
            elif loss_func == "MSE":
                criterion = nn.MSELoss()
                
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

            # dataset_train = TensorDataset(x_train, y_train)
            # loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

            best_loss = float('inf')
            patience_counter = 0

            '''further splitting training tensors into training and validation tensors'''
            # Upper limit to train/val split indicies
            upper_limit = y_train_tensor.__len__()
            # Generate the two lists
            train_indexes = random.sample(range(upper_limit), int(upper_limit * 0.8))
            val_indexes = list(set(range(upper_limit)) - set(train_indexes))
            x_train = X_train_tensor[train_indexes]
            y_train = y_train_tensor[train_indexes]
            x_val = X_train_tensor[val_indexes]
            y_val = y_train_tensor[val_indexes]
            
            '''putting training dataset into loader (preps the batches)'''
            dataset_train = TensorDataset(x_train, y_train)
            loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
            
            epochs_needed = []


            best_val_loss = float('inf')
            best_model = None
            epochs_no_improve = 0
            patience_counter = 0
            best_loss = float('inf')

            train_losses = []
            val_losses = []
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                batches = 0
                for batch_x, batch_y in loader_train:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs.flatten(), batch_y)

                    # Calculate L1 penalty (L1 regularization)
                    l1_penalty = torch.tensor(0.).to(batch_x.device)
                    for param in model.parameters():
                        l1_penalty += torch.norm(param, 1)
                    loss += l1_lambda * l1_penalty

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batches += 1
                train_losses.append(total_loss / batches) 

                model.eval()
                with torch.no_grad():
                    outputs_validation = model(x_val)
                    val_loss = criterion(outputs_validation.flatten(), y_val).item()
                    val_losses.append(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    epochs_needed.append(epoch)
                    # print('patience reached')
                    break
                # print(f'epoch = {epoch}, train loss = {train_losses[-1]}, val loss = {val_losses[-1]}')
            
                # Evaluate
            with torch.no_grad():
                model.eval()
                # Move predictions to CPU for evaluation
                pred = model(x_val).cpu().numpy()
                y_compare = y_val.cpu().numpy()
                val_score = metrics.r2_score(y_compare, pred)
                train_pred = model(x_train).cpu().numpy() 
                y_compare_train = y_train.cpu().numpy()
                train_score = metrics.r2_score(y_compare_train, train_pred)
                # print(f'End train R2 score = {train_score} validation R2 score = {val_score}')
            
            '''Load and return the best model (model with best validation score)'''
            model.load_state_dict(best_model)
                # Evaluate
            with torch.no_grad():
                model.eval()
                # Move predictions to CPU for evaluation
                pred = model(X_test_tensor).cpu().numpy() 
                y_compare = y_test_tensor.cpu().numpy()
                score = metrics.r2_score(y_compare, pred)
                mean_benchmark.append(score)


        return statistics.mean(mean_benchmark)
    '''this code should take about 2 min per 10 iterations with a batch size of 100 and epochs = 1000'''
    # Supress NaN warnings
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    '''Bounded region of parameter space'''
    pbounds = {
            'learning_rate': (0.0001, 0.01),
            'dropout': (0.0, 0.6), 
            'l1_lambda': (0.0, 1.0),
            'l2_lambda': (0.0, 1.0)
            }

    '''define optimizer'''
    optimizer = BayesianOptimization(
        f=evaluate_network,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum
        # is observed, verbose = 0 is silent
        random_state=1,
        allow_duplicate_points=True
    )

    '''start optimization'''
    start_time = time.time()
    optimizer.maximize(init_points=10, n_iter=num_iter)
    time_took = time.time() - start_time

    print(f"Total runtime: {time_took}")
    print(optimizer.max)
    return optimizer

'''plots all parameters and their target function value after optimization.'''
def plot_parameter_trials(optimizer, saving_folder):
    with open(saving_folder + '/best_hyperparams.txt', 'w') as file:
        file.write(str(optimizer.max))
    # Extracting results
    results = optimizer.res  # This gives you a list of dictionaries

    # Preparing data for plotting
    param_names = list(results[0]['params'].keys())  # Get parameter names
    data = {name: [] for name in param_names}
    target = []

    for res in results:
        for name in param_names:
            data[name].append(res['params'][name])
        target.append(res['target'])

    # Plotting
    for name in param_names:
        plt.figure(figsize=(10, 5))
        plt.scatter(data[name], target, c='blue', marker='o')
        plt.title(f'Target Function vs {name}')
        plt.xlabel(name)
        plt.ylabel('Target Function')
        plt.grid(True)
        plt.savefig(saving_folder + f'/{name}.png')

def get_best_hyperparameters_ANN(label_to_predict, hyperparameter_folder='/Volumes/Jake_ssd/bayesian_optimization'):
    best_hp_path = hyperparameter_folder + f'/{label_to_predict}/ANN/best_hyperparams.txt'
    try:
        with open(best_hp_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        print("File not found best_hyperparams.txt.")
    except Exception as e:
        print(f"An error occurred opening best_hyperparams.txt: {e}")
    converted_dict = dict(ast.literal_eval(content.removeprefix('OrderedDict')))
    dropout = converted_dict['params']['dropout']
    l1_lambda = converted_dict['params']['l1_lambda']
    l2_lambda = converted_dict['params']['l2_lambda']
    learning_rate = converted_dict['params']['learning_rate']
    
    return dropout, l1_lambda, l2_lambda, learning_rate



'''below is code that shows an example of how to preprocess the data, and run the code to train an ANN in pytorch.'''
# '''Loading the data'''
# label = 'impact site y'
# # label = 'height'


# all_labels = ['height', 'phi', 'theta', 
#         'impact site x', 'impact site y', 'impact site z', 
#         'impact site r', 'impact site phi', 'impact site theta']

# # Generate some synthetic data for demonstration purposes
# full_dataset_pathname = "/Volumes/Jake_ssd/Paper_1_results_no_feature_engineering/dataset/New_Crack_Len_FULL_OG_dataframe_2023_11_16.csv"
# # full_dataset_pathname = "/Volumes/Jake_ssd/Paper_1_results_WITH_feature_engineering/dataset/feature_transformations_2023-11-16/height/HEIGHTALL_TRANSFORMED_FEATURES.csv"
# df = pd.read_csv(full_dataset_pathname)
# # df = df[:-1]
# label_df = df[label]


# if(df.columns.__contains__('timestep_init')):
#     df = df.drop('timestep_init', axis=1)
# if(df.columns.__contains__('Unnamed: 0')):
#     df = df.drop('Unnamed: 0', axis=1)

# feature_df = df.drop(all_labels, axis=1)

# train_feats_df, test_feats_df, train_label_df, test_label_df = train_test_split(
#     feature_df, 
#     label_df, 
#     test_size=0.2,
#     random_state=2  # For reproducibility
# )


# '''preprocessing the data'''
# # First, zero-center the features by subtracting the mean
# feature_df_centered = feature_df - feature_df.mean()

# # Then, normalize the data to be between -10 and 10 by dividing by the half-range and multiplying by 10
# feature_df_range = (feature_df.max() - feature_df.min()) / 2
# feature_df_normalized = (feature_df_centered / feature_df_range) * 10

# # Split the dataset into training and testing sets with an 80/20 split
# train_feats_df, test_feats_df, train_label_df, test_label_df = train_test_split(
#     feature_df, 
#     label_df, 
#     test_size=0.2,
#     random_state=2  # For reproducibility
# )



# ''' Usage example: ''' 
# # Define batch size and other parameters
# BATCH_SIZE = 100  # Modify this according to your requirement
# LEARNING_RATE = 0.0005
# EPOCHS = 2000
# L1_LAMBDA = 0.0
# L2_LAMBDA = 0.01
# PATIENCE = 200  # Number of epochs to wait for improvement before early stopping
# DROPOUT = 0.2

# X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = prepare_data_for_pytorch(train_feats_df, test_feats_df, train_label_df, test_label_df)


# # Model instantiation
# model = ANNModel(input_size=train_feats_df.shape[1], output_size=1, dropout_rate=DROPOUT).to(device)
# # input_features, output_size, L1_reg, L2_reg, dropout)
# # Train the model
# trained_model = train_ANN(model, X_train_tensor, y_train_tensor,
#                             loss_func="MAE", 
#                             learning_rate=LEARNING_RATE, 
#                             epochs=EPOCHS, 
#                             l1_lambda=L1_LAMBDA, 
#                             l2_lambda=L2_LAMBDA, 
#                             patience=PATIENCE,
#                             batch_size=BATCH_SIZE,
#                             plot_losses=True)




# # Evaluate on test set
# train_predictions = test_model(trained_model, X_train_tensor)
# train_predictions = train_predictions.flatten()
# parody_plot(y_train_tensor, train_predictions, dataset='Train', label=label)

# test_predictions = test_model(trained_model, X_test_tensor)
# test_predictions = test_predictions.flatten()
# parody_plot(y_test_tensor, test_predictions, dataset='Test', label=label)




