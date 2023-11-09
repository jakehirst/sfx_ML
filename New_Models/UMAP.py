import pandas as pd
import numpy as np
import umap 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.cluster import KMeans, AffinityPropagation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import os



''' adds the clusters to a dataframe. the clusters are added as one-hot vectors.'''
def add_clusters_to_df(df, clusters):
    new_df = df.copy()
    # Convert clusters to one-hot encoded DataFrame
    cluster_df = pd.get_dummies(clusters, prefix='cluster')
    
    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    new_df = pd.concat([new_df, cluster_df], axis=1)
    
    return new_df

def cluster_coordinates(data, x_coord, y_coord, num_clusters=5):
    """
    Finds the optimal number of clusters for KMeans clustering on two columns of a DataFrame
    and plots the elbow method graph.

    :param data: pandas DataFrame containing the dataset
    :param column1: string, name of the first column to use for clustering
    :param column2: string, name of the second column to use for clustering
    :param max_clusters: int, the maximum number of clusters to test for
    """
    coordinates = np.column_stack((data[x_coord], data[y_coord]))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(coordinates)
    
    return cluster_labels

def plot_UMAP_3D(binned_heights_array, embedding, path_to_save):
    max_int = int(binned_heights_array.max()) + 1
    # Create a discrete color scale at the middle point between integers
    colorscale = [
        # Color changes between the range of each integer, i.e., [1, 2), [2, 3)
        [(i - 0.5) / max_int, 'color_for_integer_{}'.format(i)] for i in range(max_int)
    ]
    # Assuming 'embedding' and 'binned_heights_array' are your data
    trace = go.Scatter3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=binned_heights_array,   # set color to an array/list of desired values
            colorscale='Spectral',        # choose a colorscale
            opacity=1,
            cmin=-0.5,                        # starting color min at -0.5 so 0 centers on the first integer
            cmax=max_int - 0.5,               # setting color max to align with the largest integer
            colorbar=dict(
                tickvals=np.arange(max_int),  # set tick values at every integer
                ticktext=[str(i) for i in range(max_int)],  # set tick text to string representation
                title='Fall Height bins'        # title for the color bar
            )
        )
    )
    data = [trace]
    layout = go.Layout(
        title='UMAP projection of the Dataset',
        scene=dict(
            xaxis=dict(title='UMAP 0'),
            yaxis=dict(title='UMAP 1'),
            zaxis=dict(title='UMAP 2')
        )
    )
    fig = go.Figure(data=data, layout=layout)
    fig.write_html(path_to_save)


def get_UMAP_embedding(label, feature_df, label_df, args):
    (num_neighbors, min_distance, metric_type, spread_num) = args
    reducer = umap.UMAP(metric=metric_type,
                        min_dist=min_distance,
                        n_neighbors=num_neighbors,
                        spread=spread_num,
                        random_state=42,
                        )
    
    reducer = umap.UMAP(a=None, angular_rp_forest=False, b=None,
        force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
        local_connectivity=1.0, low_memory=False, metric=metric_type,
        metric_kwds=None, min_dist=min_distance, n_components=3, n_epochs=None,
        n_neighbors=num_neighbors, negative_sample_rate=5, output_metric='euclidean',
        output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
        set_op_mix_ratio=1.0, spread=spread_num, target_metric='categorical',
        target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
        transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)
    reducer.fit(feature_df.to_numpy())

    embedding = reducer.transform(feature_df.to_numpy())
    # Verify that the result of calling transform is
    # idenitical to accessing the embedding_ attribute
    assert(np.all(embedding == reducer.embedding_))
    # embedding.shape
    return embedding, 


#COMMENT example of umap on mnist dataset
# digits = load_digits()
# print(digits.DESCR)

# reducer = umap.UMAP(a=None, angular_rp_forest=False, b=None,
#      force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
#      local_connectivity=1.0, low_memory=False, metric='euclidean',
#      metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
#      n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
#      output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
#      set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
#      target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
#      transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)
# reducer.fit(digits.data)

# embedding = reducer.transform(digits.data)
# # Verify that the result of calling transform is
# # idenitical to accessing the embedding_ attribute
# assert(np.all(embedding == reducer.embedding_))
# embedding.shape


# plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
# plt.gca().set_aspect('equal', 'datalim')
# plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
# plt.title('UMAP projection of the Digits dataset', fontsize=24)
# plt.show()
#COMMENT example of umap on mnist dataset



''' $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ TRYING ON BINNED FALL HEIGHTS $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ '''
all_labels = ['height', 'phi', 'theta', 
                'impact site x', 'impact site y', 'impact site z', 
                'impact site r', 'impact site phi', 'impact site theta']

# top ten features for height from RF model
top_10_features = ['abs_val_sum_kink * mean thickness',
                    'abs_val_sum_kink / avg_prop_speed',
                    'abs_val_sum_kink / thickness_at_init',
                    'abs_val_sum_kink + init y',
                    'crack len + init y',
                    'crack len (unchanged)',
                    'dist btw frts + init y',
                    'abs_val_sum_kink - avg_prop_speed',
                    'avg_prop_speed - abs_val_sum_kink',
                    'abs_val_sum_kink - init z',
                    'init z - abs_val_sum_kink']


dataset_path = '/Volumes/Jake_ssd/feature_datasets/feature_transformations_2023-11-05/height/HEIGHTALL_TRANSFORMED_FEATURES.csv'
# dataset_path = '/Users/jakehirst/Desktop/sfx/sfx_ML_data/New_Crack_Len_FULL_OG_dataframe_2023_11_06.csv'
df = pd.read_csv(dataset_path)
# OG_df.drop('Unnamed: 0', axis=1, inplace=True)
# OG_df.drop(['timestep_init', 'Unnamed: 0'], axis=1, inplace=True)
label_df = df[all_labels]
OG_df = df.drop(all_labels, axis=1)

'''removing all columns that have timestep_init or Unnamed in them'''
OG_df = OG_df.filter(regex='^(?!.*timestep_init|.*Unnamed).*$')



'''only including the top 10 most correlated features'''
# OG_df = OG_df[top_10_features]


''' adding clustered initiation sites to the dataframe '''
clusters = cluster_coordinates(df, 'init x (unchanged)', 'init y (unchanged)', num_clusters=5)
# clusters = cluster_coordinates(df, 'init x', 'init y', num_clusters=5)
OG_df = add_clusters_to_df(OG_df, clusters)

'''remove all columsn that have values not suitable for float32s'''
# Assuming `df` is your DataFrame
# Define the maximum and minimum values for float32
max_float32 = np.finfo(np.float32).max
min_float32 = np.finfo(np.float32).min

columns_to_drop = []

for column in OG_df.columns:
    if OG_df[column].dtype == np.float64:  # Check only float64 columns
        if (OG_df[column].max() > max_float32) or (OG_df[column].min() < min_float32):
            columns_to_drop.append(column)

# Drop columns that are out of float32 range
OG_df.drop(columns=columns_to_drop, axis=1, inplace=True)






'''normalizing all values'''
# Initialize the StandardScaler
scaler = StandardScaler()
# Fit the scaler to the features and transform (normalize and zero-center)
features_scaled = scaler.fit_transform(OG_df)
# The scaler returns a NumPy array, so let's convert it back to a DataFrame
df_normalized = pd.DataFrame(features_scaled, columns=OG_df.columns)
OG_df = df_normalized



'''start umap by making bins'''
label = 'height'
bin_edges = [1, 2, 3, 4, 5]
bin_edges = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

# bin_edges = [1, 2.5, 5]
heights = label_df[label]
num_bins = len(bin_edges) - 1

binned_values = np.digitize(heights, bin_edges) - 1
binned_heights_array = np.array(binned_values)


'''start actual UMAP'''


figure_path = '/Users/jakehirst/Desktop/UMAP figures/feature_transformation_dataset_normalized_sparse_bins'#TODO CHANGE THIS TO MATCH THE DATASET YOU ARE USING
if(not os.path.exists(figure_path)): os.makedirs(figure_path)


embedding = umap.UMAP(n_components=3, random_state=42).fit_transform(OG_df.to_numpy(), y=binned_heights_array)
plot_UMAP_3D(binned_heights_array, embedding, 
                f'{figure_path}/automatic_supervised_dimension_reduction.html')




# for num_neighbors in n_neighbors_arr:
#     for min_distance in min_dist_arr:
#         for metric_type in metric_arr:
#             for spread_num in spread_arr:
#                 if(min_distance > spread_num): continue #min_dist cannot be mroe than spread
#                 args = (num_neighbors, min_distance, metric_type, spread_num)
#                 embedding = get_UMAP_embedding(label, OG_df, label_df, args)
                
#                 plot_UMAP_3D(binned_heights_array, embedding, 
#                              f'{figure_path}/n_neighhbors_{num_neighbors}_min_dist_{min_distance}_metric_{metric_type}_spread_{spread_num}.html')


