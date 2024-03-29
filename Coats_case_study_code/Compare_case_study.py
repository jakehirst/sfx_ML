import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


case_study_df = pd.read_excel('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Coats_case_study_code/Coats_case_studies.xlsx')
# kdiff_df = pd.read_csv('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Coats_case_study_code/Coats_kdiff_dataframe.csv')
# kdiff_df = pd.read_csv('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/New_Crack_Len_FULL_OG_dataframe.csv') #new crack lengths
# kdiff_df = pd.read_excel('/Users/jakehirst/Desktop/sfx/sfx_ML_data/New_Crack_Len_FULL_OG_dataframe_2023_09_27.xlsx', index_col=0)
new_kdiff_df = pd.read_csv("/Users/jakehirst/Desktop/sfx/sfx_ML_data/New_Crack_Len_FULL_OG_dataframe_2023_10_28.csv")
old_data_df = pd.read_csv("/Users/jakehirst/Desktop/sfx/sfx_ML_data/New_Crack_Len_FULL_OG_dataframe_2023_07_14.csv")
# kdiff_df = kdiff_df.drop('impact_sites', axis=1)


""" Limiting the study to cases where fracture is on parietal """
case_study_df = case_study_df[case_study_df['Location - Stats'].str.contains('parietal') == True]

""" Limiting the study to cases with only one fracture """
case_study_df = case_study_df[case_study_df['Num of Fx'] == 1]

""" Limiting the study to cases with 2 or less crack fronts """
case_study_df = case_study_df[case_study_df['Num of Crack Fronts'] <= 2]

""" Converting the BC height from meters to feet """
case_study_df['BC height'] = case_study_df['BC height'] * 3.28084


case_study_df = case_study_df[~case_study_df['Age (mo)'].isna()]#getting rid of all unknown ages
""" Limiting the study to cases where age is less than 3 months """
case_study_df = case_study_df[case_study_df['Age (mo)'] <= 3.0]

""" Liminting the case studies to only hard surfaces """
# case_study_df = case_study_df[case_study_df['Surface COR'] == 'high']

name = 'All_surfaces'
case_study_df.to_excel(f'/Volumes/Jake_ssd/Case_study_data/{name}.xlsx')
print("here")

''' Filtering the kdiff df so that there are no falls less than 1.5 feet '''
# kdiff_df = kdiff_df[kdiff_df['height'] > 1.5]

'''Filtering the kdiff df so that there are no cracks less than 15 mm'''
# kdiff_df = kdiff_df[kdiff_df['crack len'] >= 20]

# case_study_df = case_study_df[case_study_df['BC height'] <= 4]



"""gets the number of occurances in df where the column "column_name" is equal to "value_to_count" """
def get_number_of_occurances(df, column_name, value_to_count):
    return df[column_name].value_counts()[value_to_count]


def side_by_side_plot(case_study_df, kdiff_df, kdiff_col, coats_col, title, ylim):
    k_diff_heights = kdiff_df['height'].tolist()
    k_diff_y = kdiff_df[kdiff_col].tolist()
    
    case_study_data = case_study_df[['BC height', coats_col]]
    case_study_data = case_study_data[case_study_data[coats_col] != "Siemens - can't analyze"]
    case_study_data = case_study_data[case_study_data[coats_col] != "no scale factor - cannot analyze"]

    case_study_heights = case_study_data['BC height'].tolist()
    case_study_y = case_study_data[coats_col].tolist()
    # Create a figure and axis object
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    # Create scatter plots on the first axis
    ax[0].scatter(case_study_heights, case_study_y)
    ax[0].set_title(title + " CASE STUDY")
    ax[0].set_xlabel('case_study_heights')
    ax[0].set_ylabel(coats_col)
    ax[0].set_ylim(ylim)

    # Create scatter plots on the second axis
    ax[1].scatter(k_diff_heights, k_diff_y)
    ax[1].set_title(title + " k-diff")
    ax[1].set_xlabel('k_diff_heights')
    ax[1].set_ylabel(kdiff_col)
    ax[1].set_ylim(ylim)

    # Show the plot
    plt.show()
    
# new_kdiff_df = new_kdiff_df.drop(['init x', 'init y', 'init z', 'dist btw frts',
#         'max thickness', 'mean thickness', 'max_kink', 'linearity',
#        'abs_val_mean_kink', 'mean_kink', 'sum_kink', 'abs_val_sum_kink',
#        'avg_ori', 'angle_btw', 'phi', 'theta'], axis=1)

# full_case_study_df = case_study_df.copy()
# case_study_df = case_study_df.drop(['Image', 'Num of Fx', 'Num of Crack Fronts', 'Branching ', 
#                                     'Unnamed: 36','Unnamed: 37','Unnamed: 38','Unnamed: 39',
#                                     'Cross-Suture', 'Final Straight Line Length (mm)',
#                                     'Gender', 'Ethnicity', 'Age Bin', 'Age (mo)', 'Description',
#                                     'BC category', 'subcatgory', 'sub-subcategory', 'BC Notes',
#                                     'Surface', 'Surface COR', 'Surface Code', 'Location - Stats',
#                                     'Location - Descriptive', 'Fx Notes', 'SS', 'CAP', 'Category',
#                                     'Witness', 'Witness Notes', 'STS Location',
#                                     'ICH/ICI Notes (small unless O/W)', 'General Notes', 'STAT PLAN'], axis=1)

def get_mins_and_maxes(df):
    mean = df.mean()
    std = df.std()
    max = df.max()
    min = df.min()
    median = df.median()
    mode = df.mode()
    range = max - min
    iqr = df.quantile(0.75) - df.quantile(0.25)
    variance = df.var()
    cv = std / mean
    skewness = df.skew()
    kurtosis = df.kurtosis()
    return mean, std, max, min, median, mode, range, iqr, variance, cv, skewness, kurtosis

# case_study_df = case_study_df[case_study_df[case_study_df.columns.to_list()] != "Siemens - can't analyze"]
# case_study_df = case_study_df[case_study_df[case_study_df.columns.to_list()] != "no scale factor - cannot analyze"]


# mean_1, std_1, max_1, min_1, median_1, mode_1, range_1, iqr_1, variance_1, cv_1, skewness_1, kurtosis_1 = get_mins_and_maxes(case_study_df)

# mean_2, std_2, max_2, min_2, median_2, mode_2, range_2, iqr_2, variance_2, cv_2, skewness_2, kurtosis_2 = get_mins_and_maxes(kdiff_df)


def histogram(df1, column_name_df1, mean_1, std_1, what_are_we_comparing, x_limits=None):
    # Plotting histogram with mean and standard deviation
    # plt.hist(data=df1, x=column_name_df1, density=True, bins=20)
    plt.hist(data=df1, x=column_name_df1, weights=np.ones(len(df1)) / len(df1), bins=20)

    # plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])
    plt.axvline(mean_1[column_name_df1], color='r', linestyle='--', label='Mean')
    plt.axvline(mean_1[column_name_df1] + std_1[column_name_df1], color='y', linestyle='--', label='Standard deviation')
    plt.axvline(mean_1[column_name_df1] - std_1[column_name_df1], color='y', linestyle='--')
    if(df1.columns[0] == 'Suture to Suture'):
        title = f"{what_are_we_comparing} from case studies"
    else:
        title = f'{what_are_we_comparing} from simulation data'
    plt.title(title)
    plt.xlim(x_limits)
    plt.ylabel('Percentage of cases')
    plt.xlabel(what_are_we_comparing)
    plt.ylim(0, 0.4)
    plt.legend()
    plt.show()
    image_name = f"{title}.png".replace(" ", "_")
    # plt.savefig(f"/Users/jakehirst/Desktop/bjorn_figures/{image_name}")
    #plt.savefig(f"/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Coats_case_study_code/{image_name}")
    # plt.close()
        
def percentage_histogram(df, column_name, x_limits, y_limits, x_label, title):
    # if(df.columns[0] == 'Suture to Suture'):
    #     title = f"{x_label} from Case Studies"
    # else:
    #     title = f'{x_label} from Simulations'
    # Calculate the mean and standard deviation
    mean = df[column_name].mean()
    std = df[column_name].std()

    plt.figure(figsize=(10,6))
    # Plot the histogram
    sns.histplot(data=df, x=column_name, kde=False, bins=20)

    # Set the y-axis as a percentage of the total number of examples
    total_examples = len(df)
    plt.gca().set_yticklabels(['{:.1f}%'.format(x*100/total_examples) for x in plt.gca().get_yticks()])

    # Add a vertical line for the mean
    plt.axvline(mean, color='r', linestyle='--', label='Mean')
    # plt.text(mean+1, plt.gca().get_ylim()[1]*0.9, 'Mean: {:.2f}'.format(mean), color='r')

    # Add a shaded region for +/- one standard deviation
    plt.axvspan(mean-std, mean+std, color='g', alpha=0.3, label='Std Dev')
    #plt.text(mean-std+1, plt.gca().get_ylim()[1]*0.8, 'Standard Deviation', color='g')

    # Set the labels and title
    size = 12
    plt.xlabel('Crack length', fontweight='bold', fontsize=size)
    plt.ylabel('Percent total examples', fontweight='bold', fontsize=size)
    plt.xlim(x_limits)
    # plt.ylim((0,35))
    plt.title(title, fontweight='bold', fontsize=size+3)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()
    image_name = f"{title}.png".replace(" ", "_")
    # plt.savefig(f"/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Coats_case_study_code/PERCENTAGE_{image_name}")
    # plt.savefig(f"/Users/jakehirst/Desktop/bjorn_figures/{image_name}")
    # plt.close()

def boxplot(df, column_name, max, min):# Plotting boxplot with maximum and minimum values
    sns.boxplot(data=df, x=column_name)
    plt.text(0.9, max[column_name], f"Max: {max[column_name]}")
    plt.text(0.9, min[column_name], f"Min: {min[column_name]}")
    plt.show()
    
def scatter_plot(df, column_name, median, mode, what_are_we_comparing):
    # Plotting scatterplot with median and mode values
    sns.scatterplot(data=df, x='coats_orientation', y='coats_linearity')
    plt.axhline(median[column_name], color='r', linestyle='--', label='Median')
    plt.axhline(mode[column_name][0], color='g', linestyle='--', label='Mode')
    if(df.columns[0] == 'Suture to Suture'):
        plt.title(f"Scatterplot with mean and medians of Case study {what_are_we_comparing}")
    else:
        plt.title(f'Scatterplot with mean and medians of k-diff {what_are_we_comparing}')
    plt.legend()
    # plt.show()
    plt.close()
    
def plot_heights_for_case_studies_and_kdiff(case_study_df, kdiff_df):
    
    
    return
    

 
def plot_crack_len_vs_height(kdiff_df, kdiff_crack_len_name, kdiff_height_name, coats_df, coats_crack_len_name, coats_height_name):
    kdiff_crack_lens = kdiff_df[kdiff_crack_len_name]
    kdiff_heights = kdiff_df[kdiff_height_name]
    
    # Convert coats_height_name column to numeric values, coerce non-float values to NaN
    coats_df[coats_height_name] = pd.to_numeric(coats_df[coats_height_name], errors='coerce')
    # Remove rows where coats_height_name cannot be interpreted as a float
    coats_df = coats_df.dropna(subset=[coats_height_name])
    
    # Convert coats_crack_len_name column to numeric values, coerce non-float values to NaN
    coats_df[coats_crack_len_name] = pd.to_numeric(coats_df[coats_crack_len_name], errors='coerce')
    # Remove rows where coats_crack_len_name cannot be interpreted as a float
    coats_df = coats_df.dropna(subset=[coats_crack_len_name])
    


    coats_crack_lens = coats_df[coats_crack_len_name]
    coats_df['BC height'] = coats_df[coats_height_name]  
    coats_heights = coats_df[coats_height_name] #heights are now in feet

    coats_correlation = coats_df[coats_height_name].corr(coats_df[coats_crack_len_name])
    print(f'Pearson correlation with BC height and true crack length from case studies: {coats_correlation}')
    
    coats_ICHICI_correlation = coats_df[coats_height_name].corr(coats_df['ICH/ICI'].astype(float))
    print(f'Pearson correlation with BC height and ICH/ICI from case studies: {coats_ICHICI_correlation}')

    simulation_dataset_correlation = kdiff_df[kdiff_height_name].corr(kdiff_df[kdiff_crack_len_name])
    print(f'Pearson correlation with simulation height and true crack length from simulations: {simulation_dataset_correlation}')


    

    # Create a scatter plot with different colors for each dataset
    plt.scatter(kdiff_heights, kdiff_crack_lens, color='red', label='Data from simulation')
    plt.scatter(coats_heights, coats_crack_lens, color='blue', label='Data from case studies')

    # Add labels and title
    plt.xlabel('height (ft)')
    plt.ylabel('crack length (mm)')
    plt.title('Fall heights vs crack length from case studies and simulations')

    # Add a legend
    plt.legend()
    
    # plt.savefig('/Users/jakehirst/Desktop/sfx/Presentations_and_Papers/USNCCM/figures/younger_than_5_mo_fall_heights_vs_crack_len_BOTH.png')
    # plt.savefig(f"/Users/jakehirst/Desktop/bjorn_figures/crack_len_vs_height.png")
    # Display the scatter plot
    plt.show()
    # plt.close()


    return
    

'''preprocessing'''
new_kdiff_df = new_kdiff_df.drop(['init x', 'init y', 'init z', 'dist btw frts',
        'max thickness', 'mean thickness', 'max_kink', 'linearity',
       'abs_val_mean_kink', 'mean_kink', 'sum_kink', 'abs_val_sum_kink',
       'avg_ori', 'angle_btw', 'phi', 'theta'], axis=1)

old_data_df = old_data_df.drop(['init x', 'init y', 'init z', 'dist btw frts',
        'max thickness', 'mean thickness', 'max_kink', 'linearity',
       'abs_val_mean_kink', 'mean_kink', 'sum_kink', 'abs_val_sum_kink',
       'avg_ori', 'angle_btw', 'phi', 'theta'], axis=1)

full_case_study_df = case_study_df.copy()
case_study_df = case_study_df.drop(['Image', 'Num of Fx', 'Num of Crack Fronts', 'Branching ', 
                                    'Unnamed: 36','Unnamed: 37','Unnamed: 38','Unnamed: 39',
                                    'Cross-Suture', 'Final Straight Line Length (mm)',
                                    'Gender', 'Ethnicity', 'Age Bin', 'Age (mo)', 'Description',
                                    'BC category', 'subcatgory', 'sub-subcategory', 'BC Notes',
                                    'Surface', 'Surface COR', 'Surface Code', 'Location - Stats',
                                    'Location - Descriptive', 'Fx Notes', 'SS', 'CAP', 'Category',
                                    'Witness', 'Witness Notes', 'STS Location',
                                    'ICH/ICI Notes (small unless O/W)', 'General Notes', 'STAT PLAN'], axis=1)


case_study_df = case_study_df[case_study_df[case_study_df.columns.to_list()] != "Siemens - can't analyze"]
case_study_df = case_study_df[case_study_df[case_study_df.columns.to_list()] != "no scale factor - cannot analyze"]





'''preprocessing'''


# Boxplot of crack len
# boxplot(case_study_df, 'Final True Line Length (mm)', max_1, min_1)
# scatter_plot(case_study_df, 'Final True Line Length (mm)', median_1, mode_1, 'crack length')
# scatter_plot(kdiff_df, 'crack len', median_2, mode_2, 'crack length', x_limits=xlimits)

'''plotting crack len vs fall height for both case studies (blue) and simulation dataset (red)'''
# plot_crack_len_vs_height(old_data_df, 'crack len', 'height', case_study_df, 'Final True Line Length (mm)', 'BC height')
# plot_crack_len_vs_height(new_kdiff_df, 'crack len', 'height', case_study_df, 'Final True Line Length (mm)', 'BC height')

'''Histograms comparing crack len with percentages of dataset on y axis'''
xlimits = (0, 140)
ylimits = (0, 25)
percentage_histogram(case_study_df, 'Final True Line Length (mm)', xlimits, ylimits, 'crack length', title = f"Case studies")
percentage_histogram(old_data_df, 'crack len', xlimits, ylimits, 'crack length', title = f"Simulations with old R-curve")
percentage_histogram(new_kdiff_df, 'crack len', xlimits, ylimits, 'crack length', title = f"Simulations with modified R-curve")


'''Histograms comparing fall heights'''
# xlimits = (0, 5)
# mean_1, std_1, max_1, min_1, median_1, mode_1, range_1, iqr_1, variance_1, cv_1, skewness_1, kurtosis_1 = get_mins_and_maxes(case_study_df)
# histogram(case_study_df, 'BC height', mean_1, std_1, 'fall height', x_limits=xlimits)

# mean_2, std_2, max_2, min_2, median_2, mode_2, range_2, iqr_2, variance_2, cv_2, skewness_2, kurtosis_2 = get_mins_and_maxes(old_data_df)
# histogram(old_data_df, 'height', mean_2, std_2, 'fall height', x_limits=xlimits)

# mean_2, std_2, max_2, min_2, median_2, mode_2, range_2, iqr_2, variance_2, cv_2, skewness_2, kurtosis_2 = get_mins_and_maxes(new_kdiff_df)
# histogram(new_kdiff_df, 'height', mean_2, std_2, 'fall height', x_limits=xlimits)




# Histograms comparing Linearity
# xlimits = (0.75, 2.1)
# histogram(case_study_df, 'Linearity', mean_1, std_1, 'Linearity', x_limits=xlimits)
# histogram(kdiff_df, 'coats_linearity', mean_2, std_2, 'Linearity', x_limits=xlimits)


# Histograms comparing Orientation
# xlimits = (0, 100)
# histogram(case_study_df, 'Final Orient', mean_1, std_1, 'Orientation', x_limits=xlimits)
# histogram(kdiff_df, 'coats_orientation', mean_2, std_2, 'Orientation', x_limits=xlimits)

# # Histograms comparing Height

# xlimits = (0,5)
# histogram(case_study_df, 'BC height', mean_1, std_1, 'Height', x_limits=xlimits)
# histogram(kdiff_df, 'height', mean_2, std_2, 'Height', x_limits=xlimits)



# # side_by_side_plot(case_study_df, kdiff_df, kdiff_col='coats_orientation', coats_col="Final Orient", title="Orient vs height", ylim=(0,100))
# # side_by_side_plot(case_study_df, kdiff_df, kdiff_col='coats_linearity', coats_col="Linearity", title="linearity vs height", ylim=(0.9, 1.5))
# # side_by_side_plot(case_study_df, kdiff_df, kdiff_col='suture to suture', coats_col="Suture to Suture", title="S2S vs height", ylim=None)
# # side_by_side_plot(case_study_df, kdiff_df, kdiff_col='crack len', coats_col="Final True Line Length (mm)", title="Length vs height", ylim=(0,120))


# num_suture_to_suture_kdiff = get_number_of_occurances(kdiff_df, "suture to suture", 1)
# num_suture_to_suture_Coats = get_number_of_occurances(case_study_df, "Suture to Suture", 1)

# print(f"Total k-diff S2S = {num_suture_to_suture_kdiff}. percentage = {num_suture_to_suture_kdiff / len(kdiff_df)}")
# print(f"Total Case_study S2S = {num_suture_to_suture_Coats}. percentage = {num_suture_to_suture_Coats / len(case_study_df)}")
# full_case_study_df.to_csv(f"/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Coats_case_study_code/Filtered_cases_for_Yousef.csv")

# """getting cases where the final crack len is greater than 100mm for yousef """
# long_crack_len_cases = full_case_study_df[full_case_study_df['Final True Line Length (mm)'] > 100]
# print('all image ID\'s used in this analysis: \n' + str(full_case_study_df['Image'].to_list()))
# print('long crack len cases = \n' + str(long_crack_len_cases['Image'].to_list()))
# print("done")
