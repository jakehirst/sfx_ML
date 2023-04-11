import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

case_study_df = pd.read_excel('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Coats_case_study_code/Coats_case_studies.xlsx')
kdiff_df = pd.read_csv('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Coats_case_study_code/Coats_kdiff_dataframe.csv')

""" Limiting the study to cases with only one fracture """
case_study_df = case_study_df[case_study_df['Num of Fx'] == 1]

""" Limiting the study to cases with 2 or less crack fronts """
case_study_df = case_study_df[case_study_df['Num of Crack Fronts'] <= 2]

""" Limiting the study to cases where fracture is on parietal """
case_study_df = case_study_df[case_study_df['Location - Stats'] == "parietal"]

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

side_by_side_plot(case_study_df, kdiff_df, kdiff_col='coats_orientation', coats_col="Final Orient", title="Orient vs height", ylim=(0,100))
side_by_side_plot(case_study_df, kdiff_df, kdiff_col='coats_linearity', coats_col="Linearity", title="linearity vs height", ylim=(0.9, 1.5))
side_by_side_plot(case_study_df, kdiff_df, kdiff_col='suture to suture', coats_col="Suture to Suture", title="S2S vs height", ylim=None)
side_by_side_plot(case_study_df, kdiff_df, kdiff_col='crack len', coats_col="Final True Line Length (mm)", title="Length vs height", ylim=(0,120))


num_suture_to_suture_kdiff = get_number_of_occurances(kdiff_df, "suture to suture", 1)
num_suture_to_suture_Coats = get_number_of_occurances(case_study_df, "Suture to Suture", 1)

print(f"Total k-diff S2S = {num_suture_to_suture_kdiff}. percentage = {num_suture_to_suture_kdiff / len(kdiff_df)}")
print(f"Total Case_study S2S = {num_suture_to_suture_Coats}. percentage = {num_suture_to_suture_Coats / len(case_study_df)}")


print("done")
