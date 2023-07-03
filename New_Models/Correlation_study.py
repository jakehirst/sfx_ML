from prepare_data import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


""" Returns the correlation matrix, p matrix, and the features that have a p value less than the minimum_p_value """
def Pearson_correlation(df, label_to_predict, maximum_p_value):
    corr_matrix, p_matrix = df.corr(method=lambda x, y: pearsonr(x, y)[0]), df.corr(method=lambda x, y: pearsonr(x, y)[1])
    important_features = p_matrix[p_matrix[label_to_predict] < maximum_p_value].index
    return corr_matrix, p_matrix, list(important_features)

''' creates a heatmap of the correlations given the correlation matrix '''
def visualize_correlations(corr_mtx, label_to_predict):
    # visualise the data with seaborn
    mask = np.triu(np.ones_like(corr_mtx, dtype=np.bool))
    sns.set_style(style = 'white')
    f, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(10, 250, as_cmap=True)
    plt.subplots_adjust(bottom=0.25)
    plt.title("Correlation matrix for feature selection")
    sns.heatmap(corr_mtx, mask=mask, cmap=cmap, 
            square=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    print("done")
    
def barplot_correlations(corr_matrix, label_to_predict):
    corr_matrix = corr_matrix.drop(label_to_predict, axis=0)
    plt.figure(figsize = (10,8))
    plt.bar(corr_matrix[label_to_predict].axes[0], abs(corr_matrix[label_to_predict]))
    plt.xticks(rotation='vertical', ha='center')
    plt.xlabel("Features")
    plt.ylabel("Pearson correlation values (magnitude)")
    plt.ylim((0,1))
    plt.title(f"Pearson correlation values relating to {label_to_predict}")
    plt.subplots_adjust(bottom=0.25)
    plt.show()
    plt.close()
    


full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/New_Crack_Len_FULL_OG_dataframe.csv"
label_to_predict = 'impact site z'
all_labels = ['height', 'phi', 'theta', 
              'impact site x', 'impact site y', 'impact site z', 
              'impact site r', 'impact site phi', 'impact site theta', 'Unnamed: 0']
all_labels.remove(label_to_predict)

df = pd.read_csv(full_dataset_pathname)
df = df.drop(columns=all_labels)
# df = remove_ABAQUS_features(df)
corr_matrix, p_matrix, important_features = Pearson_correlation(df, label_to_predict, maximum_p_value=0.05)
sorted_corr_matrix = corr_matrix.iloc[corr_matrix[label_to_predict].abs().argsort()]

top_5_features = sorted_corr_matrix.index.to_list()
top_5_features.remove(label_to_predict)
top_5_features = top_5_features[len(top_5_features)-5:]
print('\n' + str(top_5_features) + '\n')
barplot_correlations(sorted_corr_matrix, label_to_predict)

top_10 = sorted_corr_matrix[label_to_predict][len(sorted_corr_matrix)-11 : len(sorted_corr_matrix)-1]
print(f"top 10 = \n{list(top_10._stat_axis)}")

# Set x-axis label
# plt.xlabel(corr_matrix.axes[0])
visualize_correlations(corr_matrix, label_to_predict)