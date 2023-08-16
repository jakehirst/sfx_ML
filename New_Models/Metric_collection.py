from sklearn.metrics import r2_score, precision_score, confusion_matrix, recall_score, f1_score
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
import csv
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_Uncertainty_across_kfolds(folder, label_to_predict):
    intervals = ['1 std or 68%', '2 std or 95%', '3 std or 99%']
    num_stds = 0
    for interval in intervals:
        num_stds += 1
        train_vals = []
        test_vals = []
        x = []
        for fold_no in range(1,6):
            metrics = pd.read_csv(folder + f'model_metrics_fold_{fold_no}.csv')
            train_vals.append(metrics[f'Train {interval}'][0])
            test_vals.append(metrics[f'Test {interval}'][0])
            x.append(fold_no)
        
        # Data for the first set of bars
        y1 = train_vals

        # Data for the second set of bars
        y2 = test_vals

        # Set the width of the bars
        bar_width = 0.35
        x = np.array(x)
        plt.figure(figsize=(12,12))
        plt.ylim(0,1)
        # Plot the bars
        plt.bar(x, y1, color='blue', width=bar_width, label='Train dataset')
        plt.bar(x + bar_width, y2, color='orange', width=bar_width, label='Test dataset')
        
        if(num_stds == 1): conf = 0.68
        elif(num_stds == 2): conf = 0.95
        elif(num_stds == 3): conf = 0.99
        # Draw a horizontal line for confidence interval
        plt.axhline(y=conf, color='red', linestyle='--', label=interval)

        # Add labels and title
        plt.xlabel('K-Fold cross validation number')
        plt.ylabel(f'Percentage of predictions whose true value falls within {num_stds} standard deviations')
        plt.title(f'Percentage of predictions falling within {interval} confidence interval when predicting {label_to_predict}')

        labels = ['fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5']
        # Set x-axis tick labels
        plt.xticks(x + bar_width/2, labels)

        # Add a legend
        plt.legend(loc='upper right')

        plt.savefig(f'/Users/jakehirst/Desktop/confidence_interval_images/conf_int_{interval}_predicting_{label_to_predict}.png')
        # Show the plot
        #plt.show()
        plt.close()
            
        
        
    return


# label_to_predict = 'height'
# label_to_predict = 'impact site x'
# label_to_predict = 'impact site y'
# saving_folder=f'/Users/jakehirst/Desktop/model_results/MODEL_COMPARISONS/GPR_{label_to_predict}/'
# plot_Uncertainty_across_kfolds(saving_folder, label_to_predict)