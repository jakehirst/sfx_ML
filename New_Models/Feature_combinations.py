from GPR import *
from linear_regression import *
from polynomial_regression import *
from lasso_regression import *
from ridge_regression import *
from CNN import *
from random_forest import *
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
from Feature_engineering import *

FORMAL_LABELS = {'init phi': '\u039E\u03A6',
                 'init z': '\u039Ez',
                 'angle_btw': '\u039B',
                 'sum_kink': '\u03A3\u0393',
                 'mean_kink': '\u0393\u2090',
                 'init r': '\u039Er',
                 'init theta': '\u039E\u03B8',
                 'avg_ori': '\u039F\u2090',
                 'abs_val_mean_kink': '|\u0393|\u2090',
                 'mean thickness': 't\u2090',
                 'init x': '\u039Ex',
                 'init y': '\u039Ey',
                 'max thickness': 't\u2098',
                 'dist btw frts': 'd',
                 'linearity': '\u03B6',
                 'max_kink': '\u0393\u2098',
                 'crack len': 'L\u209c',
                 'abs_val_sum_kink': '\u03A3|\u0393|'
                 }



def plot_mean_r2_and_mse(subsets_and_r2s, subsets_and_mse_s, figure_folder, label_to_predict):
    #get all of the means of the r^2 values for each subset of features
    subsets = list(subsets_and_r2s.keys())
    mean_r2s = []
    for i in range(len(subsets)):
        mean_r2s.append(subsets_and_r2s[subsets[i]].mean()) 
    
    #turn the data into a dataframe and sort it based on the mean r^2 values
    data = {'subsets': subsets, 'mean_r2s': mean_r2s}
    mean_r2_df = pd.DataFrame(data)
    mean_r2_df = mean_r2_df.sort_values(by='mean_r2s').reset_index(drop=True)

    #if there are more than 10 combos
    if(len(mean_r2_df) > 10):
        #only showing top 10
        mean_r2_df = mean_r2_df[len(mean_r2_df)-10:len(mean_r2_df)]
    #plot the mean r^2 values vs the features
    plt.figure(figsize=(18, 10))
    sns.barplot(x=mean_r2_df['subsets'].to_list(), y=mean_r2_df['mean_r2s'].to_list())
    # Rotate the x-axis labels vertically
    plt.xticks(rotation='vertical',fontsize=7, fontweight='bold')
    plt.subplots_adjust(bottom=0.37)
    plt.ylabel('mean r^2 values across 5 k-folds')
    plt.xlabel('feature subsets')
    plt.title('Comparing the r^2 values for each subset of features')
    # Display the plot
    # plt.show()
    plt.savefig(figure_folder + f'/Mean_R2_vs_subfeatures_plot_{label_to_predict}.png')
    plt.close()
        
        
    ''' now do the same thing for the MSE's'''
    
    #get all of the means of the r^2 values for each subset of features
    subsets = list(subsets_and_mse_s.keys())
    mean_MSEs = []
    for i in range(len(subsets)):
        mean_MSEs.append(subsets_and_mse_s[subsets[i]].mean()) 
    
    #turn the data into a dataframe and sort it based on the mean r^2 values
    data = {'subsets': subsets, 'mean_MSEs': mean_MSEs}
    mean_MSE_df = pd.DataFrame(data)
    mean_MSE_df = mean_MSE_df.sort_values(by='mean_MSEs').reset_index(drop=True)

    #if there are more than 10 combos
    if(len(mean_MSE_df) > 10):
        #only showing top 10
        mean_MSE_df = mean_MSE_df[0:10]

    #plot the mean MSE values vs the features
    plt.figure(figsize=(18, 10))
    sns.barplot(x=mean_MSE_df['subsets'].to_list(), y=mean_MSE_df['mean_MSEs'].to_list())
    # Rotate the x-axis labels vertically
    plt.xticks(rotation='vertical',fontsize=7, fontweight='bold')
    plt.subplots_adjust(bottom=0.37)
    plt.ylabel('mean MSE values across 5 k-folds')
    plt.xlabel('feature subsets')
    plt.title('Comparing the MSE values for each subset of features')
    # Display the plot
    # plt.show()
    plt.savefig(figure_folder + f'/Mean_MSE_vs_subfeatures_plot_{label_to_predict}.png')
    plt.close()
    
    return
    
#TODO NEED TO MAKE THIS THING GENERIC INSTEAD OF HARD CODED
''' Runs all of the possible feature combinations of a given model type, saving the metrics to be analyzed later. '''
def run_all_feature_combinations(Model_types, label_to_predict, feature_df, label_df, model_folder, height_range):
    all_labels = ['height', 'phi', 'theta', 
                'impact site x', 'impact site y', 'impact site z', 
                'impact site r', 'impact site phi', 'impact site theta']

    '''zero centering and normalizing data'''
    # Zero-center the data
    featureset_centered = feature_df - feature_df.mean()
    # Normalize to the range [-1, 1]
    featureset_normalized = featureset_centered / featureset_centered.abs().max()
    feature_df = featureset_normalized
    
    '''limiting data to a range of heights for predictions and for labels'''
    # Filter label_df based on 'height' column
    label_df = label_df[(label_df['height'] >= height_range[0]) & (label_df['height'] <= height_range[1])]
    # Get the indices of the filtered rows in label_df
    indices = label_df.index
    # Use loc to align feature_df with the filtered indices
    feature_df = feature_df.loc[indices]
    
    
    '''limiting data to a range of heights for predictions and for labels'''
    All_features = feature_df.columns
    
    if(not os.path.exists(model_folder)): os.makedirs(model_folder)
    saving_folder = model_folder + f'/{label_to_predict}' 
    if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)

    figure_folder = saving_folder
    if(not os.path.exists(figure_folder)):
        os.mkdir(figure_folder)


    # Get all possible subsets
    feature_subsets = []
    for i in range(1, len(All_features) + 1):
        feature_subsets.extend(list(combinations(All_features, i)))
    #TODO: get rid of this and uncomment above
    # feature_subsets = [All_features.to_list()]
    
    subsets_and_performances = {}
    subsets_and_r2s = {}
    subsets_and_mse_s = {}
    for model_type in Model_types:
        subset_no = 0

        model_folder = saving_folder + f'/{model_type}'
        if(not os.path.exists(model_folder)):     os.mkdir(model_folder)
        
        # for subset in feature_subsets: #TODO switch between these for the ANN if it freezes
        #     subset_no += 1
        for i in range(len(feature_subsets), 0 , -1): #TODO switch between these for the ANN if it freezes
            subset_no = i
            subset = feature_subsets[i-1]
            
            
            subset_saving_folder = model_folder + f'/{subset_no}'
            if(not os.path.exists(subset_saving_folder)):     os.mkdir(subset_saving_folder)
            
            print(f'\nfeatures = {list(subset)}')
            featureset = feature_df[list(subset)] #only including the columns included in the feature subset
            raw_images = []
            
            alpha = 0.1
            
            #Model_types = ['RF', 'ANN', 'GPR', 'Lasso', 'Linear', 'Poly2', 'Poly3', 'Poly4', 'Ridge']
            full_dataset_labels = label_df[label_to_predict].to_numpy()
            #TODO change the kfold call if the model type changes
            if(model_type == 'GPR'):
                Kfold_Gaussian_Process_Regression(featureset, full_dataset_labels, [], subset_saving_folder, label_to_predict, save_data=True)
            elif(model_type == 'Linear'):
                Kfold_Linear_Regression(featureset, full_dataset_labels, [], subset_saving_folder, label_to_predict, save_data=True)
            elif(model_type == 'Ridge'):
                Kfold_Ridge_Regression(alpha, featureset, full_dataset_labels, [], subset_saving_folder, label_to_predict, save_data=True)
            elif(model_type == 'Lasso'):
                Kfold_Lasso_Regression(alpha, featureset, full_dataset_labels, [], subset_saving_folder, label_to_predict, save_data=True)
            elif(model_type == 'Poly2'):
                Kfold_Polynomial_Regression(2, featureset, full_dataset_labels, [], subset_saving_folder, label_to_predict, save_data=True)
            elif(model_type == 'Poly3'):
                Kfold_Polynomial_Regression(3, featureset, full_dataset_labels, [], subset_saving_folder, label_to_predict, save_data=True)
            elif(model_type == 'RF'):
                Kfold_RF_Regression(featureset, full_dataset_labels, [], subset_saving_folder, label_to_predict, save_data=True)
            elif(model_type == 'ANN'):
                # Zero-center the data
                featureset_centered = featureset - featureset.mean()

                # Normalize to the range [-1, 1]
                featureset_normalized = featureset_centered / featureset_centered.abs().max()
                run_kfold_Regression_CNN(featureset_normalized, raw_images, full_dataset_labels, patience=50, max_epochs=2000, num_outputs=1, lossfunc='mean_squared_error', saving_folder=subset_saving_folder, use_images=False) 

            

    
''' returns the informal list of features in a new formal form '''
def replace_features_with_formal_names(feature_list):
    global FORMAL_LABELS
    return [FORMAL_LABELS.get(item, item) for item in feature_list]

''' plots the feature combinations for a given model and label_to_predict in a barplot and saves them to the figure_path '''
def barplot_model_and_metric_for_feature_combinations(Model_type, label_to_predict, metric, data, figure_path, ylims=None):
    
    x_values = [item[0] for item in data]
    y_values = [item[1] for item in data]
    
    
    modified_x_values = [string.replace('[', '').replace(']', '').replace(',', '\n') for string in x_values]

    plt.figure(figsize=(10,8))
    plt.bar(modified_x_values, y_values)
    plt.xlabel('Top 5 Feature combinations')
    plt.ylabel(f'Average {metric} across 5 k-fold cross validation')
    plt.ylim = ylims
    plt.title(f'Top 5 average {metric} values for predicting {label_to_predict} using {Model_type}')
    plt.xticks(fontsize=10, fontweight='bold', wrap=True)
    plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin to create room for x-axis labels

    # plt.show()
    plt.savefig(figure_path + f'/{metric}_predicting_{label_to_predict}_using_{Model_type}.png')
    plt.close()
    return

''' Finds the best 5 feature combinations based on R^2, adj_r^2 and MSE from the test datasets '''
def find_best_feature_combination(parent_folder, Model_type, label_to_predict):
    folder_path = parent_folder + f'/{label_to_predict}/{Model_type}'
    combinations_and_r2s = {}
    combinations_and_adj_r2s = {}
    combinations_and_mses = {}
    combinations_and_maes = {}
    folders = os.listdir(folder_path)
    filtered_folders = [string for string in folders if len(string) <= 3]
    for folder in filtered_folders:
        r2 = 0
        adj_r2 = 0
        mse = 0
        mae = 0
        for fold in range(1,6):
            data = pd.read_csv(folder_path + f'/{folder}/model_metrics_fold_{fold}.csv')
            r2 += data['Test R^2'][0]
            adj_r2 += data['Test adj_R^2'][0]
            mse += data['Test MSE'][0]
            mae += data['Test MAE'][0]
        features_used = data['features_used'][0]
        combinations_and_r2s[features_used] = r2/5
        combinations_and_adj_r2s[features_used] = adj_r2/5
        combinations_and_mses[features_used] = mse/5
        combinations_and_maes[features_used] = mae/5

    #sorting the dictionaries such that the best feature combinations are first
    top5_combinations_and_r2s = sorted(combinations_and_r2s.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_combinations_and_adj_r2s = sorted(combinations_and_adj_r2s.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_combinations_and_mses = sorted(combinations_and_mses.items(), key=lambda x: x[1])[:5]
    top5_combinations_and_maes = sorted(combinations_and_maes.items(), key=lambda x: x[1])[:5]

    figure_path = folder_path = parent_folder + f'/{Model_type}/{label_to_predict}'
    # barplot_model_and_metric_for_feature_combinations(Model_type, label_to_predict, "R^2", top5_combinations_and_r2s, figure_path, ylims=(0,1))
    # barplot_model_and_metric_for_feature_combinations(Model_type, label_to_predict, "adj R^2", top5_combinations_and_adj_r2s, figure_path,  ylims=(0,1))
    # barplot_model_and_metric_for_feature_combinations(Model_type, label_to_predict, "MSE", top5_combinations_and_mses, figure_path)
  
    return top5_combinations_and_r2s, top5_combinations_and_adj_r2s, top5_combinations_and_mses, top5_combinations_and_maes

''' 
For a given label to predict, 
and for r2, adj_r2, and mse metrics,
plots the metric for the best result of each model type 
and the features that each model used. 
'''
def barplots_for_best_on_best(label_to_predict, best_on_best, figure_path):
    metrics = ['MAE', 'R^2', 'adj_R^2', 'MSE']

    for metric in metrics:
        print('\n' + metric)
        BoB = best_on_best[metric]
        x_values = []
        y_values = []
        for key in BoB.keys():
            print(f'{key} : {BoB[key][1]} \nFeatures used = {BoB[key][0]}\n')
            feats = eval(BoB[key][0])
            feats = replace_features_with_formal_names(feats) #COMMENT tried to do this, takes up too much room
            feats = str(feats).replace('[', '').replace(']', '').replace(',', '\n')
            x_values.append(f'{key} \n\n {feats}')
            y_values.append(BoB[key][1])

        if(label_to_predict == 'height'): c='red'
        elif(label_to_predict == 'impact site x'): c='green'
        elif(label_to_predict == 'impact site y'): c='blue'

        plt.figure(figsize=(20,10))
        plt.bar(x_values, y_values, color=c)
        plt.xlabel('Model type and their features used')
        plt.ylabel(f'Average {metric} across 5 k-fold cross validation')
        plt.title(f'Comparing {metric} of the top performers of each model type when predicting {label_to_predict}')
        plt.xticks(fontsize=8, fontweight='bold', wrap=True)
        plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin to create room for x-axis labels

        # plt.show()
        plt.savefig(figure_path + f'/{metric}_best_on_best_predicting_{label_to_predict}.png')
        plt.close()
    
    return

''' Very brute force way to find best features for each model. Takes a very long time though.
- runs all feature combinations for all model types
- finds the best feature combination
- plots the best feature combination for each model type
'''
def find_best_on_best_for_all_metrics(Model_types, parent_folder, labels_to_predict):

    for label_to_predict in labels_to_predict:
        best_on_best = {'R^2':{}, 'adj_R^2': {}, 'MSE': {}, 'MAE': {}}
        for Model_type in Model_types:
            t5_r2, t5_adj_r2, t5_mses, t5_maes = find_best_feature_combination(parent_folder, Model_type, label_to_predict)
            best_on_best['R^2'][Model_type] = t5_r2[0]
            best_on_best['adj_R^2'][Model_type] = t5_adj_r2[0]
            best_on_best['MSE'][Model_type] = t5_mses[0]
            best_on_best['MAE'][Model_type] = t5_maes[0]


        
        barplots_for_best_on_best(label_to_predict, best_on_best, parent_folder+f'/{label_to_predict}')
    return
# run_all_feature_combinations()

parent_folder = '/Volumes/Jake_ssd/OCTOBER_DATASET/What_features_to_use'
Model_types = ['GPR', 'RF', 'Lasso', 'Linear', 'Poly2', 'Poly3', 'Ridge']
Model_types = ['RF', 'Lasso', 'Linear', 'Poly2', 'Poly3', 'Ridge', 'GPR']

# Model_types = ['ANN']
labels_to_predict = ['height', 'impact site x', 'impact site y']
# find_best_on_best_for_all_metrics(Model_types, parent_folder, labels_to_predict)



folder_with_data_transformations = '/Volumes/Jake_ssd/OCTOBER_DATASET/feature_transformations' #TODO: dont change this unless your feature transformations are somewhere else
# label_to_predict = 'impact site x'

label_to_predict = 'height'
all_labels = ['height', 'phi', 'theta', 
                'impact site x', 'impact site y', 'impact site z', 
                'impact site r', 'impact site phi', 'impact site theta']
feature_df, label_df = get_best_features_to_use(folder_with_data_transformations, label_to_predict, all_labels, maximum_redundancy=0.6, minimum_corr_to_label=0.2)
# feature_df, label_df = get_best_features_to_use(folder_with_data_transformations, label_to_predict, all_labels, maximum_redundancy=0.5, minimum_corr_to_label=0.1)

height_range = (1.5,4)
model_folder = f'/Volumes/Jake_ssd/OCTOBER_DATASET/Fall_height_chunks/{height_range}'
run_all_feature_combinations(Model_types, label_to_predict, feature_df, label_df, model_folder=model_folder, height_range=height_range)
find_best_on_best_for_all_metrics(Model_types, model_folder, labels_to_predict)
        

