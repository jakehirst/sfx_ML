from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def make_non_linear_svm_model(X_train, y_train, X_test, y_test, C=1.0, kernel='rbf', degree=3, random_state=None):
    '''
    C is the inverse of the regularization strength
    
    
    
    '''
    model = SVC(C=C, kernel=kernel, degree=degree, random_state=random_state)

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