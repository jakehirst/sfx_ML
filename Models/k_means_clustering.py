import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math as m

""" 
This fucntion library will help bin phi and theta values into clusters that have hopefully equal number of examples in them. 
The clusters minimize the variance between the clusters to find the best clustering strategy. 

PSEUDOCODE

- repeat until clusters no longer change for a given k value
    - pick k number of random points. These are your k clusters.
    - for each datapoint in the dataset
        - find the distance between the datapoint and each cluster
        - assign the point to the nearest cluster
    - calculate the center of each cluster by getting the mean of the locations of each cluster and then 
      recluster all of the points based on these means
    - calculate the variance of each cluster and add them up for a total variance score for this clustering strategy.
    - if the variance is lower than the minimum variance so far, then save the new variance and the k points for clustering. 

"""

def main_clustering_call(df, k, num_tries):
    #COMMENT must change the phi and theta values to radians otherwise built in python trig functions will not work.
    df['theta'] = df['theta'].multiply(m.pi / 180)
    df['phi'] = df['phi'].multiply(m.pi / 180)
    
    clusters = []
    
    print(f"\nFinding best clusters for k = {k}...\n")
    for i in range(num_tries):
        clusters.append(find_good_clusters(df, k))
        #plot_clusters(df, clusters)
        
    clusters = remove_bad_clusters(clusters)
    clusters = np.asarray(clusters)
    min_SSE_idx = np.argmin(clusters[:,1])
    
    best_clusters = clusters[min_SSE_idx]
    plot_clusters(df, best_clusters)
    return

""" 
    This shows a plot at which the k-means clustering becomes not so useful. You can tell the max number of clusters that will be useful
    by the "elbow" in the plot
"""
def find_clustering_elbow(df, num_tries):
    #COMMENT must change the phi and theta values to radians otherwise built in python trig functions will not work.
    df['theta'] = df['theta'].multiply(m.pi / 180)
    df['phi'] = df['phi'].multiply(m.pi / 180)
    ks = []
   
    
    SSEs = []
    for k in range(2, 16):
        print(f"\nFinding best clusters for k = {k}...\n")
        clusters = []
        for i in range(num_tries):
            clusters.append(find_good_clusters(df, k))
            #plot_clusters(df, clusters)
        clusters = remove_bad_clusters(clusters)
        clusters = np.asarray(clusters)
        min_SSE_idx = np.argmin(clusters[:,1])
        
        SSEs.append(clusters[min_SSE_idx][1])
        ks.append(k)
    
    plt.plot(ks, SSEs, "r*")
    plt.title("Elbow curve for K-means clustering")
    plt.xlabel("number of clusters")
    plt.ylabel("Sum of Squared Error")
    plt.show()
    
    
    return

""" Removes any cluster sets that have 0 examples in a cluster """
def remove_bad_clusters(clusters):
    for cluster in clusters:
        for i in range(len(cluster[2])):
            if(len(cluster[0][i]) == 0):
                clusters.remove(cluster)
                break
    return clusters


def find_good_clusters(df, k):
    
    cluster_centroids = find_random_clusters(df, k)
    #print(cluster_centroids)
    old_cluster_centroids = []
    old_clusters = cluster_centroids

    new_df, cluster_arrays = cluster_points(df, cluster_centroids)

    while(not old_cluster_centroids.__contains__(cluster_centroids)):
        old_cluster_centroids.append(cluster_centroids)
        old_clusters = cluster_centroids
        

        cluster_centroids = find_mean_clusters(new_df, cluster_arrays, old_clusters)

        new_df, cluster_dfs = cluster_points(new_df, cluster_centroids)
    
    try:
        total_SSE = find_SSE(cluster_dfs)
    except:
        total_SSE = 1000
    return np.array([cluster_dfs, total_SSE, cluster_centroids])

""" plots all of the points the same color on a polar coordinate system as well as the centroids of the clusters """
def plot_clusters(df, clusters):
    k = len(clusters[2])
    r_arr = []
    theta_arr = []
    
    for row in df.iterrows():
        row = row[1]
        """ to make this 2d, we plot phi as r and theta as theta"""
        r = row["phi"] 
        theta = row["theta"]
        r_arr.append(r)
        theta_arr.append(theta)
    
    cluster_theta = []
    cluster_phi = []
    for cluster in clusters[2]:
        cluster_theta.append(cluster[1])
        cluster_phi.append(cluster[0])

    # Plot the data in polar coordinates
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.set_title(f"k = {k} SSE = {clusters[1]}")
    ax.set_thetamin(0)
    ax.set_thetamax(360)
    ax.set_rmax(60)
    ax.scatter(theta_arr, r_arr, c='blue', marker="x")
    ax.scatter(cluster_theta, cluster_phi, c='red')

    # Add gridlines
    ax.grid(True)

    # Show the plot
    plt.show()
    plt.close()
    

    for i in range(k):
        r_arr = []
        theta_arr = []
        for row in clusters[0][i].iterrows():
            row = row[1]
            """ to make this 2d, we plot phi as r and theta as theta"""
            r = row["phi"] 
            theta = row["theta"]
            r_arr.append(r)
            theta_arr.append(theta)
    
        cluster_theta = [clusters[2][i][1]]
        cluster_phi = [clusters[2][i][0]]


        # Plot the data in polar coordinates
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.set_title(f"cluster {i} total in this cluster = {len(clusters[0][i])}")
        ax.set_thetamin(0)
        ax.set_thetamax(360)
        ax.set_rmax(60)
        ax.scatter(theta_arr, r_arr, c='blue', marker="x")
        ax.scatter(cluster_theta, cluster_phi, c='red')

        # Add gridlines
        ax.grid(True)

        # Show the plot
        plt.show()
        plt.close()
    return

""" gets k random points from the dataset and uses those as the clusters. """
def find_random_clusters(df, k):
    clusters = []
    samples = random.sample(range(0,len(df)), k)         
    for sample in samples:
        row = df.iloc[sample]
        clusters.append((row["phi"], row["theta"]))
    return clusters

""" clustering the points in the df by minimum distance to the clusters """
def cluster_points(df, clusters):
    df["Cluster"] = None
    df["Distance_to_Cluster"] = None
    cluster_arrays = {}
    for cluster_num in range(len(clusters)):
        cluster_arrays[cluster_num] = []

    for i in range(len(df)):
        row = df.iloc[i]
        min_distance = None
        for cluster_num in range(len(clusters)):
            cluster = clusters[cluster_num]
            row_phi = row["phi"]
            row_theta = row["theta"]
            distance = find_polar_distance(cluster, row_phi, row_theta)
            #print(distance)
            if(min_distance == None or min_distance > distance): 
                min_distance = distance
                df.at[i,"Cluster"] = cluster_num
                df.at[i,"Distance_to_Cluster"] = min_distance
        cluster_arrays[df.at[i, "Cluster"]].append(df.iloc[i])
            
    for cluster_num in range(len(clusters)): cluster_arrays[cluster_num] = pd.DataFrame(cluster_arrays[cluster_num])

    return df, cluster_arrays

""" assigns a cluster centroid to a new cluster value - the mean of all of the examples in each cluster
    if there are no examples in a cluster, then the old cluster value is kept.
"""
def find_mean_clusters(df, cluster_arrays, old_clusters):
    mean_clusters = []
    for key in cluster_arrays.keys():
        if(len(cluster_arrays[key]) == 0):
            mean_clusters.append(old_clusters[key])
            continue
        mean_phi = cluster_arrays[key]["phi"].mean()
        mean_theta = cluster_arrays[key]["theta"].mean()
        #mean_clusters.append((int(mean_phi), int(mean_theta)))
        mean_clusters.append((mean_phi, mean_theta))

    return mean_clusters


def find_mean_clusters_using_cartesian(df, cluster_arrays):
    mean_clusters = []
    for key in cluster_arrays.keys():
        phis = np.asarray(cluster_arrays[key]["phi"])
        thetas = np.asarray(cluster_arrays[key]["theta"])
        xs = phis * np.cos(thetas)
        ys = phis * np.sin(thetas)
        
        mean_x = np.mean(xs)
        mean_y = np.mean(ys)
        
        r_avg = np.sqrt(mean_x**2 + mean_y**2)
        theta_avg = np.arctan2(mean_y, mean_x)
        
        mean_phi = cluster_arrays[key]["phi"].mean()
        mean_theta = cluster_arrays[key]["theta"].mean()
        #mean_clusters.append((int(mean_phi), int(mean_theta)))
        mean_clusters.append((mean_phi, mean_theta))

    return mean_clusters


"""finds the sum of squared error for a clustering technique """
def find_SSE(cluster_dfs):
    SSE = 0
    for cluster_num in cluster_dfs.keys():
        SSE += sum(cluster_dfs[cluster_num]["Distance_to_Cluster"]**2)
    return SSE
            

""" finds the distance between the two points in polar coordinates where phi = r and theta = theta """
def find_polar_distance(cluster, row_phi, row_theta):
    phi1 = cluster[0] ; theta1 = cluster[1]
    phi2 = row_phi ; theta2 = row_theta
    return ((phi1**2) + (phi2**2) - (2*phi1*phi2*m.cos(theta1 - theta2)))**(1/2)
