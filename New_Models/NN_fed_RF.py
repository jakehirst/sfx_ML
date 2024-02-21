import joblib  # For saving and loading model
from Pytorch_ANN import *
from random_forest import *
import matplotlib.pyplot as plt

class NN_fed_RF:
    def __init__(self):
        """
        Initialize the GenericMLModel with a specific ML model.
        :param model: An instance of a machine learning model.
        """
        self.ann = None
        self.rf = None
        self.is_trained = False

    def fit(self, train_features, train_labels, ann_hyperparam_folder, num_optimization_tries=10, hyperparam_folder='/Users/jakehirst/Desktop'):
        """
        Train the model on the provided training data.
        :param train_features: Training data features.
        :param train_labels: Training data labels.
        :param train_params: Additional parameters for the training process.
        """
        '''gets the best hyperparameters to use for GPR when predicting label_to_predict'''
        def get_best_hyperparameters_NN_fed_RF(label_to_predict, hyperparameter_folder):
            import ast
            best_hp_path = f'{hyperparameter_folder}/best_hyperparams.txt'
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
        #COMMENT first, train an ANN regularly...
        dropout, l1_lambda, l2_lambda, learning_rate = get_best_hyperparameters_ANN(label_to_predict=train_labels.columns[0], hyperparameter_folder=ann_hyperparam_folder)
        self.ann = ANNModel(input_size=train_features.shape[1], output_size=1, dropout_rate=dropout).to(device)
        X_train_tensor = torch.FloatTensor(train_features.values).to(device)
        y_train_tensor = torch.FloatTensor(train_labels.values).to(device)
        self.ann = train_ANN(self.ann, X_train_tensor, y_train_tensor, loss_func='MAE', learning_rate=learning_rate, epochs=1000, l1_lambda=l1_lambda, l2_lambda=l2_lambda, patience=200, plot_losses=False) 

        #COMMENT: now train the RF on the second to last layer of the ANN
        #now train GPR using the 2nd to last layer of ANN as inputs        
        #first, get the train features by doing a forward pass of the ANN and extracting the second to last layer outputs.
        features_from_NN = self.ann.extract_features(X_train_tensor).detach().numpy()
        
        '''optimizes the rf with the NN features using bayesian optimization '''
        feat_df = pd.DataFrame(features_from_NN)
        opt = do_bayesian_optimization_RF(feat_df, train_labels, num_tries=num_optimization_tries, saving_folder=hyperparam_folder)
        depth, features, samples_leaf, samples_split, estimators = get_best_hyperparameters_NN_fed_RF(label_to_predict=train_labels.columns[0], hyperparameter_folder=hyperparam_folder)

        #define and then train RF
        self.rf =  RandomForestRegressor(max_depth=depth, max_features=features, 
                                       min_samples_leaf =samples_leaf, min_samples_split = samples_split, n_estimators=estimators, random_state=42)
        self.rf.fit(features_from_NN, train_labels)
        
        self.is_trained = True
        


    def predict(self, X):
        """
        Make predictions using the trained model.
        :param X: Data to make predictions on. (must be numpy array not pandas dataframe)
        :return: Predictions made by the model.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        X_tensor = torch.FloatTensor(X).to(device)
        # y_train_tensor = torch.FloatTensor(train_labels.values).to(device)
        #do a forward pass of the NN, but only extract the outputs from the second to last layer
        features_from_NN = self.ann.extract_features(X_tensor).detach().numpy()
        
        tree_predictions = []
        # Iterate over all trees in the random forest
        for tree in self.rf.estimators_:
            # Predict using the current tree
            tree_pred = tree.predict(features_from_NN)
            # Append the predictions to the list
            tree_predictions.append(tree_pred)

        # Convert the list to a NumPy array for easier manipulation if needed
        tree_predictions = np.array(tree_predictions)
        
        # current_predictions = model.predict(test_features.to_numpy()) #COMMENT this is the same as the average of all the individual trees
        preds = np.mean(tree_predictions, axis=0)
        stds = np.std(tree_predictions, axis=0)
        
        return preds, stds

    # def save_model(self, file_path):
    #     """
    #     Save the trained model to a file.
    #     :param file_path: Path where the model should be saved.
    #     """
    #     if not self.is_trained:
    #         raise ValueError("Only trained models can be saved.")
    #     joblib.dump(self.model, file_path)

    # def load_model(self, file_path):
    #     """
    #     Load a model from a file.
    #     :param file_path: Path to the model file.
    #     """
    #     self.model = joblib.load(file_path)
    #     self.is_trained = True

    def tune_hyperparameters(self, X_train, y_train, param_grid, **tuning_params):
        """
        Tune the model's hyperparameters.
        :param X_train: Training data features.
        :param y_train: Training data labels.
        :param param_grid: Grid of hyperparameters to search.
        :param tuning_params: Additional parameters for the hyperparameter tuning process.
        :return: Best hyperparameters found.
        """
        # Example using GridSearchCV from scikit-learn
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(self.model, param_grid, **tuning_params)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_