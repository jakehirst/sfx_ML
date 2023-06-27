import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# case_study_df = pd.read_excel('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Coats_case_study_code/Coats_case_studies.xlsx')
# kdiff_df = pd.read_csv('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Coats_case_study_code/Coats_kdiff_dataframe.csv')
kdiff_df = pd.read_csv('C:\\Users\\u1056\\sfx\\sfx_ML\\sfx_ML\\Feature_gathering\\New_Crack_Len_FULL_OG_dataframe.csv')
# """ Limiting the study to cases with only one fracture """
# case_study_df = case_study_df[case_study_df['Num of Fx'] == 1]

# """ Limiting the study to cases with 2 or less crack fronts """
# case_study_df = case_study_df[case_study_df['Num of Crack Fronts'] <= 2]

# """ Limiting the study to cases where fracture is on parietal """
# case_study_df = case_study_df[case_study_df['Location - Stats'] == "parietal"]

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
    
kdiff_df = kdiff_df.drop(['Unnamed: 0', 'front 0 x', 'front 0 y', 'front 0 z', 'front 1 x',
       'front 1 y', 'front 1 z', 'init x', 'init y', 'init z', 'dist btw frts',
        'max thickness', 'mean thickness', 'max_kink', 'linearity',
       'abs_val_mean_kink', 'mean_kink', 'sum_kink', 'abs_val_sum_kink',
       'avg_ori', 'angle_btw', 'phi', 'theta'], axis=1)

# case_study_df = case_study_df.drop(['Image', 'Num of Fx', 'Num of Crack Fronts', 'Branching ', 
#                                     'Unnamed: 36','Unnamed: 37','Unnamed: 38','Unnamed: 39',
#                                     'Cross-Suture', 'Final Straight Line Length (mm)',
#                                     'Gender', 'Ethnicity', 'Age Bin', 'Age (mo)', 'Description',
#                                     'BC category', 'subcatgory', 'sub-subcategory', 'BC Notes',
#                                     'Surface', 'Surface COR', 'Surface Code', 'Location - Stats',
#                                     'Location - Descriptive', 'Fx Notes', 'SS', 'CAP', 'Category',
#                                     'Witness', 'Witness Notes', 'STS Location', 'ICH/ICI',
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

mean_2, std_2, max_2, min_2, median_2, mode_2, range_2, iqr_2, variance_2, cv_2, skewness_2, kurtosis_2 = get_mins_and_maxes(kdiff_df)


def histogram(df1, column_name_df1, mean_1, std_1, what_are_we_comparing, x_limits=None):

    # Plotting histogram with mean and standard deviation
    sns.histplot(data=df1, x=column_name_df1, bins=20)
    plt.axvline(mean_1[column_name_df1], color='r', linestyle='--', label='Mean')
    plt.axvline(mean_1[column_name_df1] + std_1[column_name_df1], color='y', linestyle='--', label='Standard Deviation')
    plt.axvline(mean_1[column_name_df1] - std_1[column_name_df1], color='y', linestyle='--')
    if(df1.columns[0] == 'Suture to Suture'):
        title = f"Histogram of Case study {what_are_we_comparing}"
    else:
        title = f'Histogram of k-diff {what_are_we_comparing}'
    plt.title(title)
    plt.xlim(x_limits)
    plt.legend()
    plt.show()
    image_name = f"{title}.png".replace(" ", "_")
    # plt.savefig(f"/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Coats_case_study_code/{image_name}")
    plt.close()
        
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
    
# Boxplot of crack len
# boxplot(case_study_df, 'Final True Line Length (mm)', max_1, min_1)
# scatter_plot(case_study_df, 'Final True Line Length (mm)', median_1, mode_1, 'crack length')
# scatter_plot(kdiff_df, 'crack len', median_2, mode_2, 'crack length', x_limits=xlimits)


# Histograms comparing crack len
xlimits = (0, 250)
# histogram(case_study_df, 'Final True Line Length (mm)', mean_1, std_1, 'crack length', x_limits=xlimits)
histogram(kdiff_df, 'crack len', mean_2, std_2, 'crack length', x_limits=xlimits)

# # Histograms comparing Linearity
# xlimits = (0.75, 2.1)
# histogram(case_study_df, 'Linearity', mean_1, std_1, 'Linearity', x_limits=xlimits)
# histogram(kdiff_df, 'coats_linearity', mean_2, std_2, 'Linearity', x_limits=xlimits)

# # Histograms comparing Orientation
# xlimits = (0, 100)
# histogram(case_study_df, 'Final Orient', mean_1, std_1, 'Orientation', x_limits=xlimits)
# histogram(kdiff_df, 'coats_orientation', mean_2, std_2, 'Orientation', x_limits=xlimits)

# # Histograms comparing Height
# xlimits = (0,4)
# histogram(case_study_df, 'BC height', mean_1, std_1, 'Height', x_limits=xlimits)
# histogram(kdiff_df, 'height', mean_2, std_2, 'Height', x_limits=xlimits)


# side_by_side_plot(case_study_df, kdiff_df, kdiff_col='coats_orientation', coats_col="Final Orient", title="Orient vs height", ylim=(0,100))
# side_by_side_plot(case_study_df, kdiff_df, kdiff_col='coats_linearity', coats_col="Linearity", title="linearity vs height", ylim=(0.9, 1.5))
# side_by_side_plot(case_study_df, kdiff_df, kdiff_col='suture to suture', coats_col="Suture to Suture", title="S2S vs height", ylim=None)
# side_by_side_plot(case_study_df, kdiff_df, kdiff_col='crack len', coats_col="Final True Line Length (mm)", title="Length vs height", ylim=(0,120))


num_suture_to_suture_kdiff = get_number_of_occurances(kdiff_df, "suture to suture", 1)
num_suture_to_suture_Coats = get_number_of_occurances(case_study_df, "Suture to Suture", 1)

print(f"Total k-diff S2S = {num_suture_to_suture_kdiff}. percentage = {num_suture_to_suture_kdiff / len(kdiff_df)}")
print(f"Total Case_study S2S = {num_suture_to_suture_Coats}. percentage = {num_suture_to_suture_Coats / len(case_study_df)}")


print("done")
