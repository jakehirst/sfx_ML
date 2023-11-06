import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
# from sklearn.datasets import load_iris

# Load the Iris dataset
# data = load_iris()
# X = data.data
# y = data.target

# Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def make_log_regression_model(X_train, y_train, X_test, y_test, max_iter, L1=0, L2=0):
    # Initialize the Logistic Regression model
    if(L1 > 0 and L2 == 0):
        model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=max_iter, C=L1)
    if(L2 > 0 and L1 == 0):
        model = LogisticRegression(penalty='l2', solver='liblinear', max_iter=max_iter, C=L2)
    elif(L1 > 0 and L2 > 0):
        model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=max_iter, C=L2, l1_ratio=L2/L1)
    else:
        model = LogisticRegression(penalty='none', max_iter=max_iter)
        
        
    # Train the model
    model.fit(X_train, y_train.to_numpy().reshape((y_train.to_numpy().shape[0],)))

    # Make predictions on the test set
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    # Evaluate the model
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f'test accuracy = {test_accuracy}')
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f'train accuracy = {train_accuracy}')
    return model, y_pred_train, y_pred_test