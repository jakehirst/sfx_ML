import joblib  # For saving and loading model
from Pytorch_ANN import *
from GPR import *
import matplotlib.pyplot as plt

class NN_fed_GPR:
    def __init__(self):
        """
        Initialize the GenericMLModel with a specific ML model.
        :param model: An instance of a machine learning model.
        """
        self.ann = None
        self.gpr = None
        self.is_trained = False

    def fit(self, train_features, train_labels, ann_hyperparam_folder):
        """
        Train the model on the provided training data.
        :param train_features: Training data features.
        :param train_labels: Training data labels.
        :param train_params: Additional parameters for the training process.
        """
        #train an ANN regularly...
        dropout, l1_lambda, l2_lambda, learning_rate = get_best_hyperparameters_ANN(label_to_predict=train_labels.columns[0], hyperparameter_folder=ann_hyperparam_folder)
        self.ann = ANNModel(input_size=train_features.shape[1], output_size=1, dropout_rate=dropout).to(device)
        X_train_tensor = torch.FloatTensor(train_features.values).to(device)
        y_train_tensor = torch.FloatTensor(train_labels.values).to(device)
        self.ann = train_ANN(self.ann, X_train_tensor, y_train_tensor, loss_func='MAE', learning_rate=learning_rate, epochs=1000, l1_lambda=l1_lambda, l2_lambda=l2_lambda, patience=200, plot_losses=False) 

        #TODO: now train the GPR on the second to last layer of the ANN
        #now train GPR using the 2nd to last layer of ANN as inputs        
        #first, get the train features by doing a forward pass of the ANN and extracting the second to last layer outputs.
        features_from_NN = self.ann.extract_features(X_train_tensor).detach().numpy()
        
        #define and then train GPR
        #c, length_scale, noise_level = get_best_hyperparameters_GPR(label_to_predict=train_labels.columns[0], hyperparameter_folder=hyperparameter_folder)
        #kernel = ConstantKernel(constant_value=c) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        # kernel = ConstantKernel(constant_value=1.0) * RBF() + WhiteKernel(noise_level=1) #TODO try just not defining noise_level or constant_value
        kernel = ConstantKernel() * RBF() + WhiteKernel()
        self.gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, n_restarts_optimizer=50)
        self.gpr.fit(features_from_NN, train_labels)
        
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
        preds, stds = self.gpr.predict(features_from_NN, return_std=True)
        return preds, stds

    def save_model(self, file_path):
        """
        Save the trained model to a file.
        :param file_path: Path where the model should be saved.
        """
        if not self.is_trained:
            raise ValueError("Only trained models can be saved.")
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        """
        Load a model from a file.
        :param file_path: Path to the model file.
        """
        self.model = joblib.load(file_path)
        self.is_trained = True

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