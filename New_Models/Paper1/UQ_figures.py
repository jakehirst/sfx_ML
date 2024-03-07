import sys
sys.path.append('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/New_Models')

from New_Models.Paper2.Bagging_models import *
from Backward_feature_selection import *

"""
Creates a parity plot comparing predictions to true labels, with uncertainties visualized as error bars.
Gridlines are removed and scatter points are more solid in color.

:param predictions: Array of predicted values.
:param uncertainties: Array of uncertainties for each prediction.
:param true_labels: Array of true values.
:param title: Title of the plot.
:param x_label: Label for the x-axis.
:param y_label: Label for the y-axis.
"""
def create_parity_plot(predictions, uncertainties, true_labels, title, x_label, y_label, saving_path):

    # Creating the scatter plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(true_labels, predictions, yerr=uncertainties, fmt='o', c='blue', ecolor='blue', alpha=0.5, capsize=0)

    # Plotting the parity line
    max_val = max(np.max(predictions), np.max(true_labels))
    min_val = min(np.min(predictions), np.min(true_labels))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', c='red')  # dashed line for perfect prediction

    # Setting titles and labels with custom font sizes
    # plt.title(title, fontsize=12)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    
    #making the tick marks equal the same thing on x and y axis
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Customizing tick label font sizes and how many ticks for each axis
    # ticks = np.linspace(min_val, max_val, 5)
    # plt.xticks(ticks, fontsize=20)
    # plt.yticks(ticks, fontsize=20)
    
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)

    # Removing grid for cleaner look
    plt.grid(False)

    # plt.show()
    plt.savefig(saving_path)
    plt.close()
    # Show the plot
    # plt.show()



all_labels = ['height', 'phi', 'theta', 
                            'impact site x', 'impact site y', 'impact site z', 
                            'impact site r', 'impact site phi', 'impact site theta']
labels_to_predict = ['impact site x', 'impact site y', 'height']

with_or_without_transformations = 'with'
# with_or_without_transformations = 'without'

Paper2_path = f'/Volumes/Jake_ssd/Paper 2/{with_or_without_transformations}_transformations'
# model_folder = '/Volumes/Jake_ssd/Paper 1/GPR_and_RF_UQ_results'
model_folder = Paper2_path + f'/UQ_bagging_models_{with_or_without_transformations}_transformations' #COMMENT just taking the models from here...
data_folder = Paper2_path + '/5fold_datasets'
results_folder = Paper2_path + '/Compare_Code_5_fold_ensemble_results'
hyperparam_folder = Paper2_path + f'/bayesian_optimization_{with_or_without_transformations}_transformations'

model_types = ['Single GPR', 'Single RF', ]


'''make GPR or RF model'''

#get the features you need
all_features_to_keep = {}
min_features = 1 #minimum number of features you want to select from BFS (backward feature selection)
max_features = 25 #maximum number of features you want to select from BFS
for label in labels_to_predict:
    all_features_to_keep[label] = {}
    for model_type in model_types:
        backward_feat_selection_results_folder = '/Volumes/Jake_ssd/Paper 1/Paper_1_results_WITH_feature_engineering/results'
        if(model_type) == 'Single RF':
            best_features = get_best_features(backward_feat_selection_results_folder, label, 'RF', min_features, max_features)
        elif(model_type) == 'Single GPR':
            best_features = get_best_features(backward_feat_selection_results_folder, label, 'GPR', min_features, max_features)
        all_features_to_keep[label][model_type] = best_features
        
        
# train models
for fold_no in range(1,6):
    for model_type in model_types:
        for label_to_predict in labels_to_predict:
            features_to_keep = ast.literal_eval(all_features_to_keep[label_to_predict][model_type])
            
            #load test data
            test_features = pd.read_csv(f'{data_folder}/{label_to_predict}/fold{fold_no}/test_features.csv').reset_index(drop=True)[features_to_keep]
            test_labels = pd.read_csv(f'{data_folder}/{label_to_predict}/fold{fold_no}/test_labels.csv').reset_index(drop=True).to_numpy().flatten()

            #load the model
            model_saving_folder = f'{model_folder}/{label_to_predict}/{model_type}/1_models/fold_{fold_no}'
            model = load_ensemble_model(model_saving_folder + '/model_no1.sav')
            
            
            if(model_type == 'Single RF'): 
                tree_predictions = []
                # Iterate over all trees in the random forest
                for tree in model.estimators_:
                    # Predict using the current tree
                    tree_pred = tree.predict(test_features.to_numpy())
                    tree_predictions.append(tree_pred)
                # Convert the list to a NumPy array for easier manipulation if needed
                tree_predictions = np.array(tree_predictions)
                # current_predictions = model.predict(test_features.to_numpy()) #COMMENT this is the same as the average of all the individual trees
                preds = np.mean(tree_predictions, axis=0)
                pred_stds = np.std(tree_predictions, axis=0)
            elif(model_type == 'Single GPR'):
                preds, pred_stds = model.predict(test_features.to_numpy(), return_std=True)
                # current_predictions = model.predict(test_features.to_numpy())
                # current_predictions = current_predictions.reshape(current_predictions.shape[0])
            
            
            
            #save the resulting parody plot and stuff
            results_folder = f'/Volumes/Jake_ssd/Paper 1/GPR_and_RF_UQ_results/{label_to_predict}/{model_type}'
            if(not os.path.exists(results_folder)): os.makedirs(results_folder)
            
            title = f'UQ parity plot for {model_type.removeprefix("Single ")}'
            x_label = f'True values'
            y_label = f'Predicted values'
            create_parity_plot(preds, pred_stds, test_labels, title, x_label, y_label, results_folder + f'/{model_type}_{label_to_predict}_parody_plot_fold{fold_no}.png')


            
            
            
            
            
            