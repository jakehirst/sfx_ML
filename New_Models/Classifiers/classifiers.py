from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def train_random_forest_classifier(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None, random_state=None, bootstrap=False):
    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, bootstrap=bootstrap)

    # Train the model
    model.fit(X_train, y_train.to_numpy().reshape((y_train.to_numpy().shape[0],)))

    # Make predictions on the test set
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Evaluate the model
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f'Test Accuracy = {test_accuracy}')
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f'Train Accuracy = {train_accuracy}')
    
    return model, y_pred_train, y_pred_test


def train_gradient_boosting_classifier(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
    # Initialize the Gradient Boosting Classifier
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train.to_numpy().reshape((y_train.to_numpy().shape[0],)))

    # Make predictions on the test set
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Evaluate the model
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f'Test Accuracy = {test_accuracy}')
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f'Train Accuracy = {train_accuracy}')
    
    return model, y_pred_train, y_pred_test