from sklearn.metrics import r2_score, precision_score, confusion_matrix, recall_score, f1_score
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
import csv

''' 
returns the adjusted r^2 value of the given predictions
'''
def adjusted_r2(true_values, predictions, num_samples, num_features):
    r2 = r2_score(true_values, predictions)
    adjusted_r2 = 1 - ((1 - r2) * (num_samples - 1) / (num_samples - num_features - 1))
    return adjusted_r2

''' gets the sum of the squared errors '''
def get_SSE(y_true, y_pred):
    residual = y_true - y_pred
    sse = np.sum(residual ** 2)
    return sse

''' 
gets the Akaike Information Criterion (AICc) of the model, which is a metric that takes into account both the fit of the model, and the 
model's complexity.
'''
def get_AIC(y_train, y_pred_train, sse, num_parameters):
    n = len(y_train)
    p = num_parameters # (number of features used in model)
    return n * np.log(sse / n) + 2 * p

''' 
gets the Corrected Akaike Information Criterion (AICc) of the model, which is a metric that takes into account both the fit of the model, and the 
model's complexity.

AICc is essentially AIC with an extra penalty term for the number of parameters. Note that as n → ∞, the extra penalty term converges to 0, and thus AICc converges to AIC
'''
def get_AICc(y_train, y_pred_train, sse, num_parameters):
    n = len(y_train)
    p = num_parameters # (number of features used in model)
    return n * np.log(sse / n) + ((n + p) / (1 - ((p+2) / n)))


''' 
Gets the Bayesian Information Criterion (AICc) of the model, which is a metric that takes into account both the fit of the model, and the 
model's complexity.

AICc is essentially AIC with an extra penalty term for the number of parameters. Note that as n → ∞, the extra penalty term converges to 0, and thus AICc converges to AIC
'''
def get_BIC(y_train, y_pred_train, sse, num_parameters):
    n = len(y_train)
    p = num_parameters # (number of features used in model)
    return n * np.log(sse / n) + p * np.log(n)
    
    

'''
gets regression metrics comparing the predictions to the true values and saves them into the saving_folder
'''
def collect_and_save_metrics(y_train, y_pred_train, y_test, y_pred_test, features_used, fold_no, saving_folder):
    r2 = r2_score(y_test, y_pred_test)
    adj_r2 = adjusted_r2(y_test, y_pred_test, len(y_train), len(features_used))
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    train_sse = get_SSE(y_train, y_pred_train)
    aic = get_AIC(y_train, y_pred_train, train_sse, len(features_used))
    aicc = get_AICc(y_train, y_pred_train, train_sse, len(features_used))
    bic = get_BIC(y_train, y_pred_train, train_sse, len(features_used))
    
    with open(saving_folder + f'/model_metrics_fold_{fold_no}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        #write the header row
        writer.writerow(['AIC', 'AICc', 'BIC', 'Test R^2', 'Test adj_R^2', 'Test MAE', 'Test MSE', 'Test RMSE', 'features_used'])
        writer.writerow([aic, aicc, bic, r2, adj_r2, mae, mse, rmse, features_used])
    return