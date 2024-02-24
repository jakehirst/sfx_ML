import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

'''
makes 4 plots in a grid via seaborn showing test_R2, miscal_area, sharpness, 
and dispersion all in order from best performinc (left) to worst performing (right) in each plot.
'''
def make_seaborn_plots(df, overall_title, saving_folder):
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    df['Model_type'] = df['Model_type'].str.replace(' ', '\n').str.replace('_', '\n')
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Adjust the size as needed

    # Sort the DataFrame based on 'test R2' (higher is better)
    sorted_df_test_R2 = df.sort_values('test R2', ascending=False)
    # Bar plot for 'test R2'
    bplot1 = sns.barplot(x='Model_type', y='test R2', data=sorted_df_test_R2, ax=axs[0, 0], palette='Blues_d')
    axs[0, 0].set_title('Test R2 Comparison')
    bplot1.set_xticklabels(bplot1.get_xticklabels(), fontweight='bold')

    # Sort the DataFrame based on 'miscal_area' (lower is better)
    sorted_df_miscal_area = df.sort_values('miscal_area', ascending=True)
    # Bar plot for 'miscal_area'
    bplot2 = sns.barplot(x='Model_type', y='miscal_area', data=sorted_df_miscal_area, ax=axs[0, 1], palette='Reds_d')
    axs[0, 1].set_title('Miscal Area Comparison')
    bplot2.set_xticklabels(bplot2.get_xticklabels(), fontweight='bold')

    # Sort the DataFrame based on 'sharpness' (lower is better)
    sorted_df_sharpness = df.sort_values('sharpness', ascending=True)
    # Bar plot for 'sharpness'
    bplot3 = sns.barplot(x='Model_type', y='sharpness', data=sorted_df_sharpness, ax=axs[1, 0], palette='Purples_d')
    axs[1, 0].set_title('Sharpness Comparison')
    bplot3.set_xticklabels(bplot3.get_xticklabels(), fontweight='bold')

    # Sort the DataFrame based on 'dispersion' (higher is better)
    sorted_df_dispersion = df.sort_values('dispersion', ascending=False)
    # Bar plot for 'dispersion'
    bplot4 = sns.barplot(x='Model_type', y='dispersion', data=sorted_df_dispersion, ax=axs[1, 1], palette='Greens_d')
    axs[1, 1].set_title('Dispersion Comparison')
    bplot4.set_xticklabels(bplot4.get_xticklabels(), fontweight='bold')

    # Add an overarching title
    plt.suptitle(overall_title, fontsize=16)

    # Fine-tune and show plot, adjust the padding to leave space for the suptitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if(saving_folder == None):
        plt.show()
    else:
        plt.savefig(saving_folder + f'/{saving_folder.split("/")[-1]}_performance_grid.png')
    return 




def main():
    '''define the base folder where all the results are...'''
    with_or_without_transformations = 'with'
    with_or_without_transformations = 'without'

    if(with_or_without_transformations == 'with'):
        base_folder = '/Volumes/Jake_ssd/Paper 2/with_transformations/Compare_Code_5_fold_ensemble_results/'
    else:
        base_folder = '/Volumes/Jake_ssd/Paper 2/without_transformations/Compare_Code_5_fold_ensemble_results/'

    model_types = ['ANN', 'RF', 'GPR','ridge', 'Single GPR', 'Single RF', 'NN_fed_GPR','NN_fed_RF', 'RF_fed_GPR']
    labels_to_predict = ['impact site x', 'impact site y', 'height']

    num_models_for_ensemble = 20
        
    '''Start making plots'''
    for label_to_predict in labels_to_predict:
        this_label_result_df = pd.DataFrame(columns=['Model_type', 'train R2', 'test R2', 'miscal_area', 'cal_error', 'sharpness', 'dispersion'])
        print(f'LABEL = {label_to_predict}')
        for model_type in model_types:
            print(f'MODEL TYPE = {model_type}')
            
            if(model_type in ['ANN', 'RF', 'GPR','ridge']):
                result_df = pd.read_csv(f'{base_folder}{label_to_predict}/{model_type}/{num_models_for_ensemble}_models/{label_to_predict}_{model_type}_{num_models_for_ensemble}results.csv')     
            else:
                result_df = pd.read_csv(f'{base_folder}{label_to_predict}/{model_type}/1_models/{label_to_predict}_{model_type}_1results.csv')     
            
            averages = result_df.iloc[5]
            #replaceing the fold_no column with the model_type
            averages = averages.rename({'fold_no': 'Model_type'})
            averages['Model_type'] = model_type  # Replace `new_value` with the value you want to assign`
            
            this_label_result_df = pd.concat([this_label_result_df, averages.to_frame().transpose()], axis=0, ignore_index=True)

        '''great seaborn plots for organized performances of each model_type'''
        title = f'Performance of models predicting {label_to_predict.capitalize()} \n ordered from best (left) to worst (right)'
        saving_folder = base_folder + f'{label_to_predict}'
        make_seaborn_plots(this_label_result_df, overall_title=title, saving_folder=saving_folder)
        
    
    
if __name__ == "__main__":
    main()