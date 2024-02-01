import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

FORMAL_LABELS = {'angle_btw': '\u039B',
                 'avg_ori': '\u039F\u2090',
                 'linearity': '\u03B6',
                 'max_kink': '\u0393\u2098',
                 'sum_kink': '\u03A3\u0393',
                 'abs_val_sum_kink': '\u03A3|\u0393|',
                 'mean_kink': '\u0393\u2090',
                 'abs_val_mean_kink': '|\u0393|\u2090',
                 'median_kink': 'M(\u0393)',
                 'var_kink': 'var(\u0393)',
                 'std_kink': 'std(\u0393)',
                 'max thickness': 't\u2098',
                 'mean thickness': 't\u2090',
                 'median_thickness': 'M(t)',
                 'var_thickness': 'var(t)',
                 'std_thickness': 'std(t)',
                 'dist btw frts': 'd',
                 'thickness_at_init': 't\u2092',
                 'crack len': 'L\u209c',
                 'max_prop_speed': 'p\u2098',
                 'avg_prop_speed': 'p\u2090',
                 'init x': '\u039Ex',
                 'init y': '\u039Ey',
                 'init z': '\u039Ez',
                 'init phi': '\u039E\u03A6',
                 'init theta': '\u039E\u03B8',
                 'init r': '\u039Er',
                 'impact site x': 'Impact site x',
                 'impact site y': 'Impact site y',
                 'height': 'Height'
                 }

file_path_new = '/Users/jakehirst/Desktop/sfx/sfx_ML_data/New_Crack_Len_FULL_OG_dataframe_2023_11_16.csv'
# file_path_new = '/Users/jakehirst/Desktop/sfx/sfx_ML_data/New_Crack_Len_FULL_OG_dataframe_2023_11_06.csv'

df = pd.read_csv(file_path_new, index_col=0)
df.drop('timestep_init', axis=1, inplace=True)
df.drop('impact site theta', axis=1, inplace=True)
df.drop('impact site phi', axis=1, inplace=True)
df.drop('impact site r', axis=1, inplace=True)
df.drop('phi', axis=1, inplace=True)
df.drop('theta', axis=1, inplace=True)
df.drop('impact site z', axis=1, inplace=True)


df.rename(columns=FORMAL_LABELS, inplace=True)




corrs = df.corr()

# corrs.to_csv('/Users/jakehirst/Desktop/correlations_for_brian.csv')

# # Create a heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(corrs, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Heatmap")
# plt.show()

# Create a heatmap without the correlation values in each cell
plt.figure(figsize=(10, 11))
plt.title('Pearson Correlation of Base Features \n and Fall Parameters', fontsize=16, fontweight='bold')
ax = sns.heatmap(corrs, annot=False, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1,
            cbar_kws={'label': 'Correlation value'})
# Bold the axis labels
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('Pearson Correlation Value', fontsize='14', fontweight='bold')
plt.xticks(fontweight='bold',fontsize=12,)
plt.yticks(fontweight='bold',fontsize=12,)

#moving the position of the plot
pos = ax.get_position()
ax.set_position([pos.x0 + 0.05, pos.y0 + 0.05, pos.width, pos.height])

#moving the position of the colorbar
cbar_ax = cbar.ax
cbar_pos = cbar_ax.get_position()
cbar_ax.set_position([cbar_pos.x0 + 0.05, cbar_pos.y0 + 0.05, cbar_pos.width, cbar_pos.height])



# Display the heatmap
plt.savefig('/Volumes/Jake_ssd/Paper 1/figures/correlations.png')
# plt.show()



