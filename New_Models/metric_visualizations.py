import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import json


def plot_accuracy_histograms(parent_folder, folder_list, Title=None, x_label='number_of_bins/clusters', saving_filename=None):
    test_accuracies = []
    val_accuracies = []
    for experiment in folder_list:
        test_experiment_accuracies = []
        val_experiment_accuracies = []
        for i in range(1,6):
            df = pd.read_csv(parent_folder + experiment + f'/model_metrics_fold_{i}.csv')
            test_experiment_accuracies.append(df['Accuracy'].iloc[0])
            val_experiment_accuracies.append(df['Accuracy'].iloc[1])
        test_accuracies.append(np.array(test_experiment_accuracies))
        val_accuracies.append(np.array(val_experiment_accuracies))
            

        
    fig, ax = plt.subplots()
    # Plot the box and whisker plots
    ax.boxplot(test_accuracies)
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_xticklabels([str(int(label.get_text()) + 1) for label in ax.get_xticklabels()])
    ax.set_ylabel('Accuracy')
    ax.set_title("Test " + Title)
    
    if(saving_filename == None):
        plt.show()
        plt.close()
    else:
        plt.savefig(saving_filename)
        plt.close()
        
    fig, ax = plt.subplots()
    # Plot the box and whisker plots
    ax.boxplot(val_accuracies)
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_xticklabels([str(int(label.get_text()) + 1) for label in ax.get_xticklabels()])
    ax.set_ylabel('Accuracy')
    ax.set_title("Validation " + Title)
    
    if(saving_filename == None):
        plt.show()
        plt.close()
    else:
        plt.savefig(saving_filename)
        plt.close()
    return
        
def plot_precision_histograms(parent_folder, folder_list, Title=None, x_label='number_of_bins/clusters', saving_filename=None):
    test_precisions = []
    val_precisions = []
    for experiment in folder_list:
        test_experiment_precisions = []
        val_experiment_precisions = []
        for i in range(1,6):
            df = pd.read_csv(parent_folder + experiment + f'/model_metrics_fold_{i}.csv')
            test_experiment_precisions.append(df['Precision'].iloc[0])
            val_experiment_precisions.append(df['Precision'].iloc[1])
        test_precisions.append(np.array(test_experiment_precisions))
        val_precisions.append(np.array(val_experiment_precisions))
            

        
    fig, ax = plt.subplots()
    # Plot the box and whisker plots
    ax.boxplot(test_precisions)
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_xticklabels([str(int(label.get_text()) + 1) for label in ax.get_xticklabels()])
    ax.set_ylabel('Precision')
    ax.set_title("Test " + Title)
    
    if(saving_filename == None):
        plt.show()
        plt.close()
    else:
        plt.savefig(saving_filename)
        plt.close()
        
    fig, ax = plt.subplots()
    # Plot the box and whisker plots
    ax.boxplot(val_precisions)
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_xticklabels([str(int(label.get_text()) + 1) for label in ax.get_xticklabels()])
    ax.set_ylabel('Precision')
    ax.set_title("Validation " + Title)
    
    if(saving_filename == None):
        plt.show()
        plt.close()
    else:
        plt.savefig(saving_filename)
        plt.close()
    return

def plot_recall_histograms(parent_folder, folder_list, Title=None, x_label='number_of_bins/clusters', saving_filename=None):
    test_recalls = []
    val_recalls = []
    for experiment in folder_list:
        test_experiment_recalls = []
        val_experiment_recalls = []
        for i in range(1,6):
            df = pd.read_csv(parent_folder + experiment + f'/model_metrics_fold_{i}.csv')
            test_experiment_recalls.append(df['Recall'].iloc[0])
            val_experiment_recalls.append(df['Recall'].iloc[1])
        test_recalls.append(np.array(test_experiment_recalls))
        val_recalls.append(np.array(val_experiment_recalls))
            

        
    fig, ax = plt.subplots()
    # Plot the box and whisker plots
    ax.boxplot(test_recalls)
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_xticklabels([str(int(label.get_text()) + 1) for label in ax.get_xticklabels()])
    ax.set_ylabel('Recall')
    ax.set_title("Test " + Title)
    
    if(saving_filename == None):
        plt.show()
        plt.close()
    else:
        plt.savefig(saving_filename)
        plt.close()
        
    fig, ax = plt.subplots()
    # Plot the box and whisker plots
    ax.boxplot(val_recalls)
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_xticklabels([str(int(label.get_text()) + 1) for label in ax.get_xticklabels()])
    ax.set_ylabel('Recall')
    ax.set_title("Validation " + Title)
    
    if(saving_filename == None):
        plt.show()
        plt.close()
    else:
        plt.savefig(saving_filename)
        plt.close()
    return
        
def plot_F1score_histograms(parent_folder, folder_list, Title=None, x_label='number_of_bins/clusters', saving_filename=None):
    test_F1_scores = []
    val_F1_scores = []
    for experiment in folder_list:
        test_experiment_F1_scores = []
        val_experiment_F1_scores = []
        for i in range(1,6):
            df = pd.read_csv(parent_folder + experiment + f'/model_metrics_fold_{i}.csv')
            test_experiment_F1_scores.append(df['F1 Score'].iloc[0])
            val_experiment_F1_scores.append(df['F1 Score'].iloc[1])
        test_F1_scores.append(np.array(test_experiment_F1_scores))
        val_F1_scores.append(np.array(val_experiment_F1_scores))
            

        
    fig, ax = plt.subplots()
    # Plot the box and whisker plots
    ax.boxplot(test_F1_scores)
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_xticklabels([str(int(label.get_text()) + 1) for label in ax.get_xticklabels()])
    ax.set_ylabel('F1 Score')
    ax.set_title("Test " + Title)
    
    if(saving_filename == None):
        plt.show()
        plt.close()
    else:
        plt.savefig(saving_filename)
        plt.close()
        
    fig, ax = plt.subplots()
    # Plot the box and whisker plots
    ax.boxplot(val_F1_scores)
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_xticklabels([str(int(label.get_text()) + 1) for label in ax.get_xticklabels()])
    ax.set_ylabel('F1 Score')
    ax.set_title("Validation " + Title)
    
    if(saving_filename == None):
        plt.show()
        plt.close()
    else:
        plt.savefig(saving_filename)
        plt.close()
    return
        
def plot_regression_metrics(parent_folder, folder_list, Dataset=None, Title=None, x_label='fold_no',y_label='r^2',  saving_filename=None):
    if(Dataset == 'Test vs Validation'):
        plt.figure(figsize=(20,8))
    else:
        plt.figure(figsize=(10,8))


    for folder in folder_list:
        test_metrics = []
        val_metrics = []
        fold_nos = []
        for fold_no in range(1,6):
            csv = pd.read_csv(parent_folder + folder + f'/model_metrics_fold_{fold_no}.csv')
            test_metrics.append(csv[y_label][0])
            if(Dataset == 'Test vs Validation'):
                val_metrics.append(csv[y_label][1])
            fold_nos.append(fold_no)
        
        if(folder.__contains__('site x')): color = 'g';label = 'x'
        if(folder.__contains__('site y')): color = 'r';label = 'y'
        if(folder.__contains__('site z')): color = 'b';label = 'z'

        if(folder.__contains__('REMOVED_ABAQUS_REFERENCES')): linestyle ='--'; label = label + ' no ABAQUS feats'
        else: linestyle ='-'
        
        if(Dataset == 'Test vs Validation'):
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
            plt.plot(fold_nos, test_metrics, c=color, linestyle=linestyle, linewidth=2, label=label)
            plt.title("Test " + y_label + ' ' + Title)
            plt.xlabel(x_label)
            plt.ylim((0,1))
            plt.ylabel(y_label)
            plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
            plt.plot(fold_nos, val_metrics, c=color, linestyle=linestyle, linewidth=2, label=label)
            plt.ylim((0,1))
            plt.title("Validation " + y_label + ' ' + Title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        elif(Dataset == None):
            plt.plot(fold_nos, test_metrics, c=color, linestyle=linestyle, linewidth=2, label=label)
            # plt.ylim((0,1))
            plt.title("Test " + y_label + ' ' + Title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)



    plt.legend()
    plt.show()



            
            
        
        

# folder_list = ['impact_site_logi_reg_2_clusters',
#                'impact_site_logi_reg_3_clusters',
#                'impact_site_logi_reg_4_clusters',
#                'impact_site_logi_reg_5_clusters',
#                'impact_site_logi_reg_6_clusters',
#                'impact_site_logi_reg_7_clusters',
#                'impact_site_logi_reg_8_clusters',
#                'impact_site_logi_reg_9_clusters',
#                'impact_site_logi_reg_10_clusters']

# folder_list = ['height_logi_reg_2_bins',
#                'height_logi_reg_3_bins',
#                'height_logi_reg_4_bins',
#                'height_logi_reg_5_bins',
#                'height_logi_reg_6_bins',
#                'height_logi_reg_7_bins',
#                'height_logi_reg_8_bins',
#                'height_logi_reg_9_bins',
#                'height_logi_reg_10_bins']

# folder_list = ['Single_output_regression_Jimmy_impact site x',
#                 'Single_output_regression_Jimmy_impact site y',
#                 'Single_output_regression_Jimmy_impact site z',
#                 'Single_output_regression_REMOVED_ABAQUS_REFERENCES_Jimmy_impact site x',
#                 'Single_output_regression_REMOVED_ABAQUS_REFERENCES_Jimmy_impact site y',
#                 'Single_output_regression_REMOVED_ABAQUS_REFERENCES_Jimmy_impact site z'
#                 ]

folder_list = ['GPR_Jimmy_impact site x',
                'GPR_Jimmy_impact site y',
                'GPR_Jimmy_impact site z',
                ]

parent_folder = '/Users/jakehirst/Desktop/model_results/'

# plot_regression_metrics(parent_folder, folder_list, Dataset='Test vs Validation', Title='metrics across all k folds', x_label='fold_no',y_label='r^2' , saving_filename=None)
# plot_regression_metrics(parent_folder, folder_list, Dataset='Test vs Validation', Title='metrics across all k folds', x_label='fold_no',y_label='adj_r^2' , saving_filename=None)
# plot_regression_metrics(parent_folder, folder_list, Dataset='Test vs Validation', Title='metrics across all k folds', x_label='fold_no',y_label='MAE' , saving_filename=None)
# plot_regression_metrics(parent_folder, folder_list, Dataset='Test vs Validation', Title='metrics across all k folds', x_label='fold_no',y_label='MSE' , saving_filename=None)
# plot_regression_metrics(parent_folder, folder_list, Dataset='Test vs Validation', Title='metrics across all k folds', x_label='fold_no',y_label='RMSE' , saving_filename=None)


plot_regression_metrics(parent_folder, folder_list, Title='metrics across all k folds', x_label='fold_no',y_label='r^2' , saving_filename=None)
plot_regression_metrics(parent_folder, folder_list, Title='metrics across all k folds', x_label='fold_no',y_label='adj_r^2' , saving_filename=None)
plot_regression_metrics(parent_folder, folder_list, Title='metrics across all k folds', x_label='fold_no',y_label='MAE' , saving_filename=None)
plot_regression_metrics(parent_folder, folder_list, Title='metrics across all k folds', x_label='fold_no',y_label='MSE' , saving_filename=None)
plot_regression_metrics(parent_folder, folder_list, Title='metrics across all k folds', x_label='fold_no',y_label='RMSE' , saving_filename=None)


# plot_F1score_histograms(parent_folder, folder_list, Title='F1 scores for impact site logistic regression')
# plot_recall_histograms(parent_folder, folder_list, Title='recalls for impact site logistic regression')
# plot_precision_histograms(parent_folder, folder_list, Title='precisions for impact site logistic regression')
# plot_accuracy_histograms(parent_folder, folder_list, Title='accuracies for impact site logistic regression')