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
        

parent_folder = '/Users/jakehirst/Desktop/model_results/'
folder_list = ['impact_site_logi_reg_2_clusters',
               'impact_site_logi_reg_3_clusters',
               'impact_site_logi_reg_4_clusters',
               'impact_site_logi_reg_5_clusters',
               'impact_site_logi_reg_6_clusters',
               'impact_site_logi_reg_7_clusters',
               'impact_site_logi_reg_8_clusters',
               'impact_site_logi_reg_9_clusters',
               'impact_site_logi_reg_10_clusters']

folder_list = ['height_logi_reg_2_bins',
               'height_logi_reg_3_bins',
               'height_logi_reg_4_bins',
               'height_logi_reg_5_bins',
               'height_logi_reg_6_bins',
               'height_logi_reg_7_bins',
               'height_logi_reg_8_bins',
               'height_logi_reg_9_bins',
               'height_logi_reg_10_bins']


plot_F1score_histograms(parent_folder, folder_list, Title='F1 scores for impact site logistic regression')
plot_recall_histograms(parent_folder, folder_list, Title='recalls for impact site logistic regression')
plot_precision_histograms(parent_folder, folder_list, Title='precisions for impact site logistic regression')
plot_accuracy_histograms(parent_folder, folder_list, Title='accuracies for impact site logistic regression')