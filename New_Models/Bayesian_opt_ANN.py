import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from kerastuner.tuners import BayesianOptimization


# Define the Keras model building function
def build_model(hp):
    model = Sequential()
    
    # Input layer
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    
    # Hidden layer 1
    model.add(Dense(128, activation='relu', kernel_regularizer=l1_l2(
        l1=hp.Float('l1_1', min_value=1e-5, max_value=1e-2, sampling='LOG'),
        l2=hp.Float('l2_1', min_value=1e-4, max_value=1e-1, sampling='LOG'))))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))

    # Hidden layer 2
    model.add(Dense(128, activation='relu', kernel_regularizer=l1_l2(
        l1=hp.Float('l1_2', min_value=1e-5, max_value=1e-2, sampling='LOG'),
        l2=hp.Float('l2_2', min_value=1e-4, max_value=1e-1, sampling='LOG'))))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.1)))

    # Hidden layer 3
    model.add(Dense(128, activation='relu', kernel_regularizer=l1_l2(
        l1=hp.Float('l1_3', min_value=1e-5, max_value=1e-2, sampling='LOG'),
        l2=hp.Float('l2_3', min_value=1e-4, max_value=1e-1, sampling='LOG'))))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout_4', min_value=0.0, max_value=0.5, step=0.1)))

    # Output layer for regression
    model.add(Dense(1, activation='linear'))
    
    # Compile the model
    model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
                  loss='mean_squared_error')
    return model

# Define a function for cross-validation within the Bayesian Optimization process
def cross_val_score(hp):
    validation_scores = []
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        model = build_model(hp)
        model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=10,  # You might want to lower this for faster iterations
            batch_size=32,
            verbose=0  # You can set to 1 to see the progress
        )
        scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        validation_scores.append(scores)

    # Return the average validation score for the current set of hyperparameters
    return np.mean(validation_scores)









'''get input features and labels'''
''' when predicting impact site x'''
top_10_features = ['sqrt(mean thickness)',
                   'init x * init z',
                    'init z * max thickness',
                    'init z * thickness_at_init',
                    'linearity * mean thickness',
                    'angle_btw ^ avg_ori',
                    'avg_ori + init x',
                    'init x + init y',
                    'init x + max_prop_speed',
                    'linearity + max thickness']

''' when predicting impact site y '''
# top_10_features =['sqrt(avg_prop_speed)',
#                 'init y * init z',
#                 'init y * mean thickness',
#                 'max_kink / abs_val_sum_kink',
#                 'avg_ori ^ crack len',
#                 'thickness_at_init ^ avg_ori',
#                 'abs_val_mean_kink ^ init z',
#                 'avg_prop_speed + init y',
#                 'init y + mean_kink',
#                 'init y + thickness_at_init']

'''when predicting height''' 
# top_10_features = ['abs_val_sum_kink * mean thickness',
#                     'abs_val_sum_kink / avg_prop_speed',
#                     'abs_val_sum_kink / thickness_at_init',
#                     'abs_val_sum_kink + init y',
#                     'crack len + init y',
#                     'crack len (unchanged)',
#                     'dist btw frts (unchanged)',
#                     'dist btw frts + init y',
#                     'abs_val_sum_kink - avg_prop_speed',
#                     'avg_prop_speed - abs_val_sum_kink',
#                     'abs_val_sum_kink - init z',
#                     'init z - abs_val_sum_kink']


labels_to_predict = ['height', 'impact site x', 'impact site y']

# Generate some synthetic data for demonstration purposes
df = pd.read_csv("/Volumes/Jake_ssd/OCTOBER_DATASET/feature_transformations_2023-10-28/height/HEIGHTALL_TRANSFORMED_FEATURES.csv")
label_df = df.copy()[labels_to_predict]
df = df.drop(labels_to_predict, axis=1)
if(df.columns.__contains__('timestep_init')):
    df = df.drop('timestep_init', axis=1)


label = 'impact site x'
optimal_stuff = do_bayesian_optimization(df, label_df[label], 100, features_to_keep=top_10_features)

''' get input features and labels'''





'''kfold stuff'''
# Assuming 'df_features' and 'df_labels' are your dataframes
X = df_features.values
y = df_labels.values.ravel()  # Assuming your labels are in one column

# Define the number of splits for K-Fold
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Instantiate the BayesianOptimization tuner
tuner = BayesianOptimization(
    build_model,
    objective=cross_val_score,
    max_trials=10,
    executions_per_trial=1,
    directory='bayesian_optimization',
    project_name='keras_regression_cv'
)

# Start the Bayesian Optimization
tuner.search_space_summary()
tuner.search()

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]
'''kfold stuff'''




# Specify the input shape (number of features in your dataset)
input_shape = (num_features,)

# Instantiate the BayesianOptimization tuner
tuner = BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='bayesian_optimization',
    project_name='keras_regression'
)

# Define the data here
# X_train, y_train, X_val, y_val = ...

# Start the Bayesian Optimization
tuner.search(X_train, y_train,
             epochs=50,
             validation_data=(X_val, y_val),
             verbose=1)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters()[0]

# Build the model with the best hyperparameters and train it
model = build_model(best_hps)
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))