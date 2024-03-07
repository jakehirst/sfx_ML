from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def make_random_forest_classifier(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None, random_state=None, bootstrap=False):
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