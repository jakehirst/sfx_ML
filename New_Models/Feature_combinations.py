from GPR import *
from linear_regression import *
from polynomial_regression import *
from lasso_regression import *
from ridge_regression import *
from CNN import *
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

FORMAL_LABELS = {'init phi': 'Initiation site phi',
                 'init z': 'Initiation site z',
                 'angle_btw': 'Crack vector angle',
                 'sum_kink': 'Sum of kink angles',
                 'mean_kink': 'Average kink angle',
                 'init r': 'Initiation site r',
                 'init theta': 'Initiation site theta',
                 'avg_ori': 'Average crack orientation',
                 'abs_val_mean_kink': 'Average Abs(kink angle)',
                 'mean thickness': 'Average thickness of skull along crack',
                 'init x': 'Initiation site x',
                 'init y': 'Initiation site y',
                 'max thickness': 'Max thickness of skull along crack',
                 'dist btw frts': 'Distance between crack fronts',
                 'linearity': 'Linearity',
                 'max_kink': 'Max kink angle',
                 'crack len': 'Crack Length',
                 'abs_val_sum_kink': 'Sum of Abs(kink angles)'
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
def run_all_feature_combinations():

    full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/New_Crack_Len_FULL_OG_dataframe.csv"
    image_folder = '/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx/new_dataset/Visible_cracks'
    all_labels = ['height', 'phi', 'theta', 
                'impact site x', 'impact site y', 'impact site z', 
                'impact site r', 'impact site phi', 'impact site theta']

    LTP = ['impact site x', 'impact site y', 'height']
    feats = [['crack len', 'init phi', 'avg_ori', 'init x'], ['max_kink', 'angle_btw', 'init y'], ['abs_val_sum_kink', 'crack len', 'max_kink', 'linearity', 'max thickness']]

    LTP = ['impact site y', 'height']
    feats = [['max_kink', 'angle_btw', 'init y'], ['abs_val_sum_kink', 'crack len', 'max_kink', 'linearity', 'max thickness']]

    for i in range(len(LTP)):
        label_to_predict = LTP[i]
        All_features = feats[i]
        
        #TODO change the label to predict from impact site x to impact site y to height for each model type
        #when predicting impact site x
        # label_to_predict = 'impact site x' 
        # All_features = ['crack len', 'init phi', 'avg_ori', 'init x']
        #when predicting impact site y
        # label_to_predict = 'impact site y'
        # All_features = ['max_kink', 'angle_btw', 'init y']
        #when predicting impact site z
        # label_to_predict = 'impact site z'
        # All_features = ['init z', 'init theta']
        #when predicting height
        # label_to_predict = 'height'
        # All_features = ['abs_val_sum_kink', 'crack len', 'max_kink', 'linearity', 'max thickness']

        model_folder = '/Users/jakehirst/Desktop/Feature_subset_study/Poly4'#TODO change the saving folder if the model type changes
        if(not os.path.exists(model_folder)): os.mkdir(model_folder)
        saving_folder = model_folder + f'/{label_to_predict}' 
        if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
        correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.1)
        #correlated_featureset = remove_ABAQUS_features(correlated_featureset)



        # figure_folder = f'/Users/jakehirst/Desktop/Feature_subset_study/predicting_{label_to_predict}'
        figure_folder = saving_folder
        if(not os.path.exists(figure_folder)):
            os.mkdir(figure_folder)

        # Get all possible subsets
        feature_subsets = []
        for i in range(1, len(All_features) + 1):
            feature_subsets.extend(list(combinations(All_features, i)))

        subsets_and_performances = {}
        subsets_and_r2s = {}
        subsets_and_mse_s = {}
        subset_no = 0
        
        for subset in feature_subsets: #TODO switch between these for the ANN if it freezes
            subset_no += 1
        # for i in range(28,len(feature_subsets)): #TODO switch between these for the ANN if it freezes
        #     subset_no = i
        #     subset = feature_subsets[i-1]
            
            
            subset_saving_folder = saving_folder + f'/{subset_no}'
            if(not os.path.exists(subset_saving_folder)):     os.mkdir(subset_saving_folder)
            
            print(f'\nfeatures = {list(subset)}')
            featureset = correlated_featureset[list(subset)]
            raw_images = []
            
            alpha = 0.1
            
            #TODO change the kfold call if the model type changes
            # Kfold_Gaussian_Process_Regression(featureset, raw_images, full_dataset_labels, important_features, subset_saving_folder, label_to_predict, save_data=True)
            # Kfold_Linear_Regression(featureset, raw_images, full_dataset_labels, important_features, subset_saving_folder, label_to_predict, save_data=True)
            # Kfold_Ridge_Regression(alpha, featureset, raw_images, full_dataset_labels, important_features, subset_saving_folder, label_to_predict, save_data=True)
            # Kfold_Lasso_Regression(alpha, featureset, raw_images, full_dataset_labels, important_features, subset_saving_folder, label_to_predict, save_data=True)
            Kfold_Polynomial_Regression(2, featureset, raw_images, full_dataset_labels, important_features, subset_saving_folder, label_to_predict, save_data=True)
            # Kfold_Polynomial_Regression(3, featureset, raw_images, full_dataset_labels, important_features, subset_saving_folder, label_to_predict, save_data=True)
            # Kfold_Polynomial_Regression(4, featureset, raw_images, full_dataset_labels, important_features, subset_saving_folder, label_to_predict, save_data=True)
            # run_kfold_Regression_CNN(featureset, raw_images, full_dataset_labels, patience=100, max_epochs=2000, num_outputs=1, lossfunc='mean_squared_error', saving_folder=subset_saving_folder, use_images=False) #TODO fix how metrics are saved in the CNN

            
            # subsets_and_performances[str(list(subset))] = performances
            # subsets_and_r2s[str(list(subset))] = r2s
            # subsets_and_mse_s[str(list(subset))] = mse_s

    # plot_mean_r2_and_mse(subsets_and_r2s, subsets_and_mse_s, figure_folder, label_to_predict)
    
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
    folder_path = parent_folder + f'/{Model_type}/{label_to_predict}'
    combinations_and_r2s = {}
    combinations_and_adj_r2s = {}
    combinations_and_mses = {}
    folders = os.listdir(folder_path)
    filtered_folders = [string for string in folders if len(string) <= 3]
    for folder in filtered_folders:
        r2 = 0
        adj_r2 = 0
        mse = 0
        for fold in range(1,6):
            data = pd.read_csv(folder_path + f'/{folder}/model_metrics_fold_{fold}.csv')
            r2 += data['Test R^2'][0]
            adj_r2 += data['Test adj_R^2'][0]
            mse += data['Test MSE'][0]
        features_used = data['features_used'][0]
        combinations_and_r2s[features_used] = r2/5
        combinations_and_adj_r2s[features_used] = adj_r2/5
        combinations_and_mses[features_used] = mse/5
    #sorting the dictionaries such that the best feature combinations are first
    top5_combinations_and_r2s = sorted(combinations_and_r2s.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_combinations_and_adj_r2s = sorted(combinations_and_adj_r2s.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_combinations_and_mses = sorted(combinations_and_mses.items(), key=lambda x: x[1])[:5]
    
    figure_path = folder_path = parent_folder + f'/{Model_type}/{label_to_predict}'
    barplot_model_and_metric_for_feature_combinations(Model_type, label_to_predict, "R^2", top5_combinations_and_r2s, figure_path, ylims=(0,1))
    barplot_model_and_metric_for_feature_combinations(Model_type, label_to_predict, "adj R^2", top5_combinations_and_adj_r2s, figure_path,  ylims=(0,1))
    barplot_model_and_metric_for_feature_combinations(Model_type, label_to_predict, "MSE", top5_combinations_and_mses, figure_path)
  
    return top5_combinations_and_r2s, top5_combinations_and_adj_r2s, top5_combinations_and_mses

''' 
For a given label to predict, 
and for r2, adj_r2, and mse metrics,
plots the metric for the best result of each model type 
and the features that each model used. 
'''
def barplots_for_best_on_best(label_to_predict, best_on_best, figure_path):
    metrics = ['R^2', 'adj_R^2', 'MSE']

    for metric in metrics:
        print('\n' + metric)
        BoB = best_on_best[metric]
        x_values = []
        y_values = []
        for key in BoB.keys():
            print(f'{key} : {BoB[key][1]} \nFeatures used = {BoB[key][0]}\n')
            feats = eval(BoB[key][0])
            # feats = replace_features_with_formal_names(feats) #COMMENT tried to do this, takes up too much room
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

        plt.show()
        # plt.savefig(figure_path + f'/{metric}_best_on_best_predicting_{label_to_predict}.png')
        plt.close()
    
    return


parent_folder = '/Users/jakehirst/Desktop/Feature_subset_study'
Model_types = ['ANN', 'GPR', 'Lasso', 'Linear', 'Poly2', 'Poly3', 'Poly4', 'Ridge']
labels_to_predict = ['height', 'impact site x', 'impact site y']

for label_to_predict in labels_to_predict:
    best_on_best = {'R^2':{}, 'adj_R^2': {}, 'MSE': {}}
    for Model_type in Model_types:
        if(Model_type == 'Linear' or Model_type == 'Poly4'):
            print('here')
        t5_r2, t5_adj_r2, t5_mses = find_best_feature_combination(parent_folder, Model_type, label_to_predict)
        best_on_best['R^2'][Model_type] = t5_r2[0]
        best_on_best['adj_R^2'][Model_type] = t5_adj_r2[0]
        best_on_best['MSE'][Model_type] = t5_mses[0]

    
    barplots_for_best_on_best(label_to_predict, best_on_best, parent_folder)

        
        
print('done')
