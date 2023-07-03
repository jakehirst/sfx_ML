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

'''
gets regression metrics comparing the predictions to the true values and saves them into the saving_folder
'''
def collect_and_save_metrics(y_test, y_pred_test, num_training_samples, num_features, important_features, fold_no, saving_folder):
    r2 = r2_score(y_test, y_pred_test)
    adj_r2 = adjusted_r2(y_test, y_pred_test, num_training_samples, num_features)
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    
    with open(saving_folder + f'/model_metrics_fold_{fold_no}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        #write the header row
        writer.writerow(['r^2', 'adj_r^2', 'MAE', 'MSE', 'RMSE', 'features_used'])
        writer.writerow([r2, adj_r2, mae, mse, rmse, important_features])
    return