from prepare_data import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



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

''' returns the informal list of features in a new formal form '''
def replace_features_with_formal_names(feature_list):
    global FORMAL_LABELS
    return [FORMAL_LABELS.get(item, item) for item in feature_list]


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

def horiz_barplot_1D_correlation_visualization(sorted_corr_mtx, label_to_predict, fig_path):
    fontsize = 15
    #remove all the front location features
    sorted_corr_mtx = sorted_corr_mtx[~sorted_corr_mtx.index.str.contains('front')]

    values = sorted_corr_mtx[label_to_predict]
    features = list(values._stat_axis)
    features.remove(label_to_predict)
    values = values.drop(label_to_predict)
    
    features = replace_features_with_formal_names(features)

    # Determine bar colors based on values
    colors = ['blue' if value >= 0 else 'red' for value in values]

    # Create a larger figure with adjusted subplots spacing
    fig, ax = plt.subplots(figsize=(7, 10))
    # fig.subplots_adjust(left=0.4)  # Adjust the spacing on the left side
    # Create custom legend patches
    legend_patches = [
        mpatches.Patch(color='blue', label='Positive Correlations'),
        mpatches.Patch(color='red', label='Negative Correlations')
    ]

    # Add legend
    ax.legend(handles=legend_patches, loc='lower right', fontsize=fontsize)
    # # Plot the horizontal bar chart
    # plt.barh(features, np.abs(values), color=colors)

    # Create the bars
    bars = ax.barh(features, np.abs(values), color=colors)
    ax.set(yticklabels=[]) #removing all of the labels on the y column
    # Set the alpha (transparency) value for all the bars
    alpha = 0.5
    for bar in bars:
        bar.set_alpha(alpha)
    # Add labels inside the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(.05, bar.get_y() + bar.get_height() / 2, features[i], ha='left', va='center', fontweight='bold', fontsize=fontsize)

    # Add labels and title
    plt.xlim(0,1)
    plt.xlabel('Pearson Correlation Values',fontsize=fontsize)
    plt.ylabel('Features',fontsize=fontsize)
    plt.title(f'{label_to_predict.capitalize()}', fontsize=fontsize, fontweight='bold')
     
    # Show the plot
    # plt.show()
    plt.savefig(fig_path + f'/barplot_correlations_for_{label_to_predict}.png')
    plt.close()
    
    return
    
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
    
def principal_component(df, features):
    
    x = StandardScaler().fit_transform(df.loc[:,features].values)
    PCA_s = PCA(n_components=1) #n_components is the number of PCA features you want
    principalComponents = PCA_s.fit_transform(x)
    return principalComponents

full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/New_Crack_Len_FULL_OG_dataframe.csv"
label_to_predict = 'height'
label_to_predict = 'impact site x'
label_to_predict = 'impact site y'
# label_to_predict = 'impact site phi'
# label_to_predict = 'impact site theta'

all_labels = ['height', 'phi', 'theta', 
              'impact site x', 'impact site y', 'impact site z', 
              'impact site r', 'impact site phi', 'impact site theta', 'Unnamed: 0']
all_labels.remove(label_to_predict)

df = pd.read_csv(full_dataset_pathname)
df = df.drop(columns=all_labels)
# df = remove_ABAQUS_features(df)
corr_matrix, p_matrix, important_features = Pearson_correlation(df, label_to_predict, maximum_p_value=0.05)
sorted_corr_matrix = corr_matrix.iloc[corr_matrix[label_to_predict].abs().argsort()]


horiz_barplot_1D_correlation_visualization(sorted_corr_matrix, label_to_predict, fig_path='/Users/jakehirst/Desktop/sfx/Presentations_and_Papers/USNCCM/figures')

# top_5_features = sorted_corr_matrix.index.to_list()
# top_5_features.remove(label_to_predict)
# top_5_features = top_5_features[len(top_5_features)-5:]
# print('\n' + str(top_5_features) + '\n')
# barplot_correlations(sorted_corr_matrix, label_to_predict)

# top_10 = sorted_corr_matrix[label_to_predict][len(sorted_corr_matrix)-11 : len(sorted_corr_matrix)-1]
# print(f"top 10 = \n{list(top_10._stat_axis)}")

# # Set x-axis label
# # plt.xlabel(corr_matrix.axes[0])
# visualize_correlations(corr_matrix, label_to_predict)