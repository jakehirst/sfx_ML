from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
import numpy as np
import os


def bin_labels(train_labels, test_labels, label, num_bins):
    all_labels = {}
    if(label == 'impact site x'):
        min_value = -55 ; max_value = 45
    elif(label == 'impact site y'):
        min_value = -35 ; max_value = 45
    elif(label == 'height'):
        min_value = 1 ; max_value = 5
    else: print('wrong label... ')
        
    binned_test_labels = pd.cut(test_labels[label], bins=np.linspace(min_value, max_value, num_bins + 1), labels=False, include_lowest=True)
    binned_train_labels = pd.cut(train_labels[label], bins=np.linspace(min_value, max_value, num_bins + 1), labels=False, include_lowest=True)
            
    return binned_test_labels, binned_train_labels

'''Training a random forest on already binned labels using a random forest'''
def train_random_forest_classifier(X_train, y_train, n_estimators=100, max_depth=None, random_state=None, bootstrap=False):

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, bootstrap=bootstrap)
    model.fit(X_train, y_train.to_numpy().reshape((y_train.to_numpy().shape[0],)))

    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f'Train Accuracy = {train_accuracy}')

    return model, y_pred_train

'''Training a random forest on already binned labels using gradient boosting'''
def train_gradient_boosting_classifier(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
    
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train.to_numpy().reshape((y_train.to_numpy().shape[0],)))
    
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f'Train Accuracy = {train_accuracy}')
    
    return model, y_pred_train



def save_classifier(model, model_path):
    """Save the trained model to the specified file path using pickle."""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to {model_path}')

def load_classifier(model_path):
    """Load a trained model from the specified file path using pickle."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f'Model loaded from {model_path}')
    return model


def train_classifier(X_train, y_train, model_saving_folder, label_to_predict, features_to_keep, model_type, num_bins):
    X_train = X_train[features_to_keep]
    
    if(model_type == 'GB_classifier'):
        model, y_pred_train = train_gradient_boosting_classifier(X_train, y_train)
    
    elif(model_type == 'RF_classifier'):
        model, y_pred_train = train_random_forest_classifier(X_train, y_train)
        
    save_classifier(model, model_saving_folder + f'/{model_type}_{num_bins}_bins.pkl')        
    return
