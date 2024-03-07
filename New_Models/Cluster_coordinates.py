import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.mixture import GaussianMixture
from scipy.stats import pearsonr
from matplotlib.ticker import MaxNLocator



def cluster_coordinates(x_coord, y_coord, num_clusters, cluster_type='kmeans'):
    # Combine init_x and init_y into a 2D array
    coordinates = np.column_stack((x_coord, y_coord))

    if(cluster_type == 'kmeans'):
        # Perform K-means clustering with 5 clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(coordinates)
    elif cluster_type == 'kmeans++':
        # Perform K-means clustering with K-Means++ initialization
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
        cluster_labels = kmeans.fit_predict(coordinates)
    elif cluster_type == 'affinity_propagation':
        # Perform Affinity Propagation clustering
        affinity_propagation = AffinityPropagation()
        cluster_labels = affinity_propagation.fit_predict(coordinates)
    elif cluster_type == 'gmm':
        # Perform Gaussian Mixture Model clustering with 'num_clusters' components
        gmm = GaussianMixture(n_components=num_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(coordinates)
    else:
        # Handle other cases or raise an error
        raise ValueError("Invalid cluster_type.")
    
    return cluster_labels 

def side_by_side_3d_plot(plot_x, plot_y, plot_z, cluster_labels, plot_colored, x_col, y_col, z_col, xlim, ylim, zlim, cluster_number='ALL'):
    
    try:
        corr, p_value = pearsonr(plot_z, plot_colored)
    except:
        print('correlation unavailable')
        corr = 0
        p_value = 100
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20,12), subplot_kw={'projection': '3d'})
    # fig.set_facecolor('black')
    for ax in axes:
        ax.set_facecolor('grey')
    

    # First subplot (left): K-means clustering
    scatter1 = axes[0].scatter(plot_x, plot_y, plot_z, c=cluster_labels, cmap='viridis', marker='o')
    axes[0].set_xlabel(x_col)
    axes[0].set_ylabel(y_col)
    axes[0].set_zlabel(z_col)
    axes[0].set_title('K-means Clusters')
    # Set limits for the left subplot
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[0].set_zlim(zlim)
    
    cbar1 = plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    cbar1.locator = MaxNLocator(integer=True)  # Only show integer ticks
    cbar1.update_ticks()

    # Second subplot (right): The commented code
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    colormap = plt.cm.ScalarMappable(cmap=plt.get_cmap('RdPu'))
    colormap.set_array(plot_colored)
    scatter2 = axes[1].scatter(plot_x, plot_y, plot_z, c=plot_colored, cmap='RdPu', marker='o')
    axes[1].set_xlabel(x_col)
    axes[1].set_ylabel(y_col)
    axes[1].set_zlabel(z_col)
    axes[1].set_title('Sample Data')
    # Set limits for the left subplot
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    axes[1].set_zlim(zlim)

    plt.colorbar(colormap, ax=axes[1], label='crack len')

    plt.title(f'Showing cluster: {cluster_number} \n height correlation = {corr} pval = {p_value}')
    plt.tight_layout()
    plt.show()
    return


'''
example of how to call this:

dataset_path = '/Volumes/Jake_ssd/feature_datasets/feature_transformations_2023-11-05/height/HEIGHTALL_TRANSFORMED_FEATURES.csv'
data = pd.read_csv(dataset_path)
cluster_and_make_side_by_side_3d_plots(data, 'impact site x', 'impact site y', 'height', 'dist btw frts (unchanged)', 5, cluster_type='kmeans')
'''
def cluster_and_make_side_by_side_3d_plots(data, x_col, y_col, z_col, colored, num_clusters, cluster_type):
    plot_x = data[x_col]
    plot_y = data[y_col]
    plot_z = data[z_col]
    plot_colored = data[colored]

    
    # Create cluster labels
    cluster_labels = cluster_coordinates(plot_x, plot_y, num_clusters, cluster_type=cluster_type)
    xlim = (-60, 60)
    ylim = (-40, 40)
    # ylim = (min(plot_y), max(plot_y))
    zlim = (0, 5)
    '''plot all clusters on the same plot'''
    side_by_side_3d_plot(plot_x, plot_y, plot_z, cluster_labels, plot_colored, x_col, y_col, z_col, xlim, ylim, zlim)

    '''now plot each cluster individually and look for patterns'''
    for cluster in range(cluster_labels.max() + 1):
        cluster_indexes = np.where(cluster_labels == cluster)[0]
        
        print('here')
        side_by_side_3d_plot(plot_x.iloc[cluster_indexes], 
                             plot_y.iloc[cluster_indexes], 
                             plot_z.iloc[cluster_indexes], 
                             cluster_labels[cluster_indexes],
                             plot_colored.iloc[cluster_indexes], 
                             x_col, y_col, z_col,
                             xlim, ylim, zlim,
                             cluster_number=cluster)
    
    return


def find_optimal_clusters(data, column1, column2, max_clusters=20):
    """
    Finds the optimal number of clusters for KMeans clustering on two columns of a DataFrame
    and plots the elbow method graph.

    :param data: pandas DataFrame containing the dataset
    :param column1: string, name of the first column to use for clustering
    :param column2: string, name of the second column to use for clustering
    :param max_clusters: int, the maximum number of clusters to test for
    """

    # Extract the data for clustering
    points = data[[column1, column2]].values

    # Calculate the sum of squared distances for different numbers of clusters
    sse = []
    for k in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(points)
        sse.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters+1), sse, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow Method For Optimal Clusters')
    plt.xticks(range(1, max_clusters+1))  # Set x-axis ticks to be integers

    plt.show()


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

''' adds the clusters to a dataframe. the clusters are added as one-hot vectors.'''
def add_clusters_to_df(df, clusters):
    new_df = df.copy()
    # Convert clusters to one-hot encoded DataFrame
    cluster_df = pd.get_dummies(clusters, prefix='cluster')
    
    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    new_df = pd.concat([new_df, cluster_df], axis=1)
    
    return new_df



