from GPR import *
from CNN import *
from lasso_regression import *
from ridge_regression import *
from polynomial_regression import *
from linear_regression import *
from random_forest import *
import math as m


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

full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_data/New_Crack_Len_FULL_OG_dataframe_2023_10_14.csv"
image_folder = '/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx/new_dataset/Visible_cracks'
all_labels = ['height', 'phi', 'theta', 
            'impact site x', 'impact site y', 'impact site z', 
            'impact site r', 'impact site phi', 'impact site theta']

models_and_features_impact_site_x = {
                                    #  'ANN': ['crack len', 'init x'],
                                     'GPR': ['init x', 'max thickness', 'mean thickness', 'init r', 'init theta'],
                                     'RF': ['init x', 'max thickness', 'mean thickness', 'init theta'],
                                     '2nd deg poly': ['init x', 'max thickness', 'mean thickness', 'init r'],
                                     '3rd deg poly': ['mean thickness', 'init r', 'linearity'],
                                    #  '4th deg poly': ['init x'],
                                     'Ridge': ['init x', 'max thickness', 'init r'],
                                     'Lasso': ['init x', 'mean thickness', 'init r'],
                                     'Linear': ['init x', 'init r', 'init theta', 'linearity']
                                     }
models_and_features_impact_site_y = {
                                    #  'ANN': ['max_kink', 'init y'],
                                     'GPR': ['init y', 'avg_ori', 'init theta'],
                                     'RF': ['init y', 'avg_ori', 'init theta', 'mean thickness'],
                                     '2nd deg poly': ['init y', 'dist btw frts', 'init theta', 'mean thickness'],
                                     '3rd deg poly': ['init y', 'crack len', 'dist btw frts'],
                                    #  '4th deg poly': ['init y'],
                                     'Ridge': ['init y', 'crack len', 'mean thickness', ],
                                     'Lasso': ['init y', 'dist btw frts','mean thickness'],
                                     'Linear': ['init y', 'crack len', 'dist btw frts', 'avg_ori', 'mean thickness']
                                     }
models_and_features_height = {
                                # 'ANN': ['abs_val_sum_kink'],
                                'GPR': ['angle_btw'],
                                'RF': ['abs_val_sum_kink', 'crack len', 'max_kink'],
                                '2nd deg poly': ['crack len'],
                                '3rd deg poly': ['crack len'],
                                # '4th deg poly': ['abs_val_sum_kink', 'max thickness'],
                                'Ridge': ['abs_val_sum_kink', 'dist btw frts', 'angle_btw'],
                                'Lasso': ['abs_val_sum_kink', 'crack len', 'max_kink', 'dist btw frts'],
                                'Linear': ['abs_val_sum_kink', 'dist btw frts', 'crack len']
                                }

''' 
Runs experiment where each model in models_and_features_label will be trained with their respective label to predict with a number of datapoints that are in multiples of 20
until the full dataset is used. The metrics are saved just like any other training of an ML model. they will be stored in 
saving_folder --> label_to_predict --> model_type -->  num_training_points     
'''
def run_data_experiment(models_and_features_label, label_to_predict, saving_folder, full_dataset_pathname, image_folder, all_labels):
    correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=1)
    max_training_examples = m.floor(len(correlated_featureset) * 0.8 / 10) * 10
    
    n = 10 #the nubmer of trainin examples starts at n and increases by multiples of n
    multiples_of_n = [i for i in range(10, max_training_examples + 1, 10)] + [max_training_examples] #getting all multiples of 20 in a list along with the max amount of training datapoints

    # multiples_of_20 = [20,40,60]
    performances = {key: [] for key in models_and_features_label}
    for model_type in models_and_features_label.keys():
        for num_training_points in multiples_of_n:
            featureset = correlated_featureset[models_and_features_label[model_type]]
            print(model_type)
            if(not os.path.exists(saving_folder + f'/{label_to_predict}')):     os.mkdir(saving_folder + f'/{label_to_predict}')
            if(not os.path.exists(saving_folder + f'/{label_to_predict}' + f'/{model_type}')):     os.mkdir(saving_folder + f'/{label_to_predict}' + f'/{model_type}')
            if(not os.path.exists(saving_folder + f'/{label_to_predict}' + f'/{model_type}' + f'/{num_training_points}')):     os.mkdir(saving_folder + f'/{label_to_predict}' + f'/{model_type}'+ f'/{num_training_points}')
            real_saving_folder = saving_folder + f'/{label_to_predict}' + f'/{model_type}' + f'/{num_training_points}'
            
            alpha = 0.1
            if(model_type == 'GPR'):
                Kfold_Gaussian_Process_Regression(featureset, full_dataset_labels, important_features, real_saving_folder, label_to_predict, save_data=True, num_training_points=num_training_points)
            elif(model_type == 'RF'):
                Kfold_RF_Regression(featureset, full_dataset_labels, important_features, real_saving_folder, label_to_predict, save_data=True, num_training_points=num_training_points)
            elif(model_type == 'Linear'):
                Kfold_Linear_Regression(featureset, full_dataset_labels, important_features, real_saving_folder , label_to_predict, save_data=True, num_training_points=num_training_points)
            elif(model_type == 'Ridge'):
                Kfold_Ridge_Regression(alpha, featureset, full_dataset_labels, important_features, real_saving_folder, label_to_predict, save_data=True, num_training_points=num_training_points)
            elif(model_type == 'Lasso'):
                Kfold_Lasso_Regression(alpha, featureset, full_dataset_labels, important_features, real_saving_folder, label_to_predict, save_data=True, num_training_points=num_training_points)
            elif(model_type == '2nd deg poly'):
                Kfold_Polynomial_Regression(2, featureset, full_dataset_labels, important_features, real_saving_folder, label_to_predict, save_data=True, num_training_points=num_training_points)
            elif(model_type == '3rd deg poly'):
                Kfold_Polynomial_Regression(3, featureset, full_dataset_labels, important_features, real_saving_folder, label_to_predict, save_data=True, num_training_points=num_training_points)
            elif(model_type == '4th deg poly'):
                Kfold_Polynomial_Regression(4, featureset, full_dataset_labels, important_features, real_saving_folder, label_to_predict, save_data=True, num_training_points=num_training_points)
            elif(model_type == 'ANN'):
                run_kfold_Regression_CNN(featureset, [], full_dataset_labels, patience=100, max_epochs=2000, num_outputs=1, lossfunc='mean_squared_error', saving_folder=real_saving_folder, use_images=False, num_training_points=num_training_points) #TODO fix how metrics are saved in the CNN
    return 

# run_data_experiment(models_and_features_impact_site_x, 'impact site x', '/Users/jakehirst/Desktop/sfx/do_we_have_enough_data',full_dataset_pathname, image_folder, all_labels)
# run_data_experiment(models_and_features_impact_site_y, 'impact site y', '/Users/jakehirst/Desktop/sfx/do_we_have_enough_data',full_dataset_pathname, image_folder, all_labels)
# run_data_experiment(models_and_features_height, 'height', '/Users/jakehirst/Desktop/sfx/do_we_have_enough_data',full_dataset_pathname, image_folder, all_labels)

""" 
parses through the experiment that is done with the run_data_experiment function. gets the model metrics and stores them as the following:
models_and_data --> dict with model types as keys --> dict with num_train_pts as keys --> pandas dataframe with all metrics from all kfolds.
"""
def parse_data_experiment(directory_with_experiments, label_to_predict):
    dir = directory_with_experiments + f"/{label_to_predict}"
    if(not os.path.exists(dir)): os.mkdir(dir)
    model_types = [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]
    models_and_data = {}
    for model_type in model_types:
        models_and_data[model_type] = {}
        model_dir = dir + f'/{model_type}'
        all_num_train_pts = [name for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, name))]
        for num_train_pts in all_num_train_pts:
            models_and_data[model_type][num_train_pts] = pd.DataFrame()
            for fold_no in range(1,6):
                fold_metrics = pd.read_csv(model_dir + f'/{num_train_pts}/model_metrics_fold_{fold_no}.csv')
                models_and_data[model_type][num_train_pts] = pd.concat([models_and_data[model_type][num_train_pts], fold_metrics], ignore_index=True)
    return models_and_data


def scatterplot_models_and_data_over_all_num_training_pts(models_and_data, saving_folder, label_to_predict, metric_to_plot):
    # plt.figure()
    # for model_type in models_and_data.keys():
    #     for num_train_pts in models_and_data[model_type].keys():
    #         data = models_and_data[model_type][num_train_pts][metric_to_plot]
    
    # Assuming your dictionary is named data_dict
    data = []

    for model_type, model_dict in models_and_data.items():
        for num_train_pts, df in model_dict.items():
            test_r2_values = df[metric_to_plot].mean()
            data.extend([(model_type, int(num_train_pts), test_r2_values)])
    df = pd.DataFrame(data, columns=['Model Type', 'Number of Training Points', metric_to_plot])
    fig, ax = plt.subplots(figsize=(10,10))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'lime']
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'x']  # Define markers for each model type

    for i, model_type in enumerate(df['Model Type'].unique()):
        subset = df[df['Model Type'] == model_type]
        ax.scatter(subset['Number of Training Points'], subset[metric_to_plot], color=colors[i], marker=markers[i], label=model_type)
    ax.set_title(f'{metric_to_plot} for varying # of data when predicting {label_to_predict}', fontweight='bold')
    ax.set_xlabel('Number of Training Points')
    ax.set_ylabel(f"Average {metric_to_plot} over 5-folds")
    
    if(metric_to_plot.__contains__('R^2')): ax.set_ylim(0,1);  ax.legend(loc='upper left')
    elif(metric_to_plot.__contains__('MSE') or metric_to_plot.__contains__('MAE')):
        # Calculate y-limits excluding outliers
        y_min, y_max = np.percentile(df[metric_to_plot], [0, 90])  # Excluding the top 10% values
        ax.set_ylim(y_min, y_max)
        ax.legend(loc='upper right')

    if(saving_folder == None):
        plt.show()
    else:
        plt.savefig(saving_folder + f'/{metric_to_plot}_vs_datapoints_predicting_{label_to_predict}.png')
        plt.close()
    return


saving_folder = '/Volumes/Jake_ssd/OCTOBER_DATASET/do_we_have_enough_data'
run_data_experiment(models_and_features_impact_site_x, 'impact site x', saving_folder, full_dataset_pathname, image_folder, all_labels)
height_models_and_data_impact_site_x = parse_data_experiment(saving_folder, 'impact site x')
scatterplot_models_and_data_over_all_num_training_pts(height_models_and_data_impact_site_x, saving_folder + '/impact site x', 'impact site x', 'Test R^2')
scatterplot_models_and_data_over_all_num_training_pts(height_models_and_data_impact_site_x, saving_folder + '/impact site x', 'impact site x', 'Test adj_R^2')
scatterplot_models_and_data_over_all_num_training_pts(height_models_and_data_impact_site_x, saving_folder + '/impact site x', 'impact site x', 'Test MAE')
scatterplot_models_and_data_over_all_num_training_pts(height_models_and_data_impact_site_x, saving_folder + '/impact site x', 'impact site x', 'Test MSE')

run_data_experiment(models_and_features_impact_site_y, 'impact site y', saving_folder, full_dataset_pathname, image_folder, all_labels)
height_models_and_data_impact_site_y = parse_data_experiment(saving_folder, 'impact site y')
scatterplot_models_and_data_over_all_num_training_pts(height_models_and_data_impact_site_y, saving_folder + '/impact site y', 'impact site y', 'Test R^2')
scatterplot_models_and_data_over_all_num_training_pts(height_models_and_data_impact_site_y, saving_folder + '/impact site y', 'impact site y', 'Test adj_R^2')
scatterplot_models_and_data_over_all_num_training_pts(height_models_and_data_impact_site_y, saving_folder + '/impact site y', 'impact site y', 'Test MAE')
scatterplot_models_and_data_over_all_num_training_pts(height_models_and_data_impact_site_y, saving_folder + '/impact site y', 'impact site y', 'Test MSE')

run_data_experiment(models_and_features_height, 'height', saving_folder, full_dataset_pathname, image_folder, all_labels)
height_models_and_data_height = parse_data_experiment(saving_folder, 'height')
scatterplot_models_and_data_over_all_num_training_pts(height_models_and_data_height, saving_folder + '/height', 'height', 'Test R^2')
scatterplot_models_and_data_over_all_num_training_pts(height_models_and_data_height, saving_folder + '/height', 'height', 'Test adj_R^2')
scatterplot_models_and_data_over_all_num_training_pts(height_models_and_data_height, saving_folder + '/height', 'height', 'Test MAE')
scatterplot_models_and_data_over_all_num_training_pts(height_models_and_data_height, saving_folder + '/height', 'height', 'Test MSE')