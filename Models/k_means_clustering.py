import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math as m
from keras.utils import to_categorical
import os

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

def main_clustering_call(df, k, num_tries, folder):
    #df = df.iloc[1:20]
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
    df = best_clusters[0][0].copy()
    y_col_values = ['0']
    
    for i in range(1, k): 
        y_col_values.append(str(i)) 
        print(best_clusters[0][i])
        df = df.append(best_clusters[0][i], ignore_index=True)
        
        
    
    plot_clusters(df, best_clusters, folder)
    df = df.drop("phi", axis=1)
    df = df.drop("theta", axis=1)
    df = df.drop("Distance_to_Cluster", axis=1)
    
    
    
    cluster_assignments = np.asarray(df["Cluster"])
    cluster_assignments = to_categorical(cluster_assignments)
    cluster_assignment_df = pd.DataFrame(cluster_assignments)
    cluster_assignment_df.columns = y_col_values
    df = df.drop("Cluster", axis=1)
    df = pd.concat([df, cluster_assignment_df], axis=1)
    #df = df.reset_index(drop=True)
    
    for key in best_clusters[0].keys():
        for row in best_clusters[0][key].iterrows():
            filepath = row[1]["Filepath"]
            correct_cluster = row[1]["Cluster"]
            df_row = df.loc[df["Filepath"] == filepath]
            if(not float(df_row[str(correct_cluster)])== 1.0):
                print("wrong")

            
            
    
    
    return df, best_clusters, y_col_values

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

""" Removes any cluster sets that have 0 examples in any of its clusters """
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

    new_df, cluster_arrays = cluster_points(df.copy(), cluster_centroids)

    while(not old_cluster_centroids.__contains__(cluster_centroids)):
        old_cluster_centroids.append(cluster_centroids)
        old_clusters = cluster_centroids
        

        cluster_centroids = find_mean_clusters(new_df, cluster_arrays, old_clusters)

        new_df, cluster_dfs = cluster_points(new_df, cluster_centroids)
    
    try:
        total_SSE = find_SSE(cluster_dfs)
    except:
        total_SSE = 1000
    return np.array([cluster_dfs, total_SSE, cluster_centroids], dtype=object)

def plot_all_points(df, clusters):
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

""" plots all of the points the same color on a polar coordinate system as well as the centroids of the clusters """
def plot_clusters(df, clusters, folder):
    k = len(clusters[2])
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
        path = folder + f"/cluster_num{i}"
        # Show the plot
        plt.savefig(path)
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


""" shows a bar chart of the nubmer of examples per bin and the number of misses per bin. """
def Plot_Bins_and_misses(clusters, test_predictions, df, folder):
    if(not os.path.isdir(folder + "/hits_and_misses")):
        os.mkdir(folder + "/hits_and_misses")
    if(not os.path.isdir(folder + "/right_clusters_and_predictions")):
        os.mkdir(folder + "/right_clusters_and_predictions")
    if(not os.path.isdir(folder + "/wrong_clusters_and_predictions")):
        os.mkdir(folder + "/wrong_clusters_and_predictions")
        
    misses = []
    totals = []
    false_positives = []
    x = []
    for i in range(df.shape[1]):
        misses.append(0)
        totals.append(0)
        false_positives.append(0)
        x.append(i)
    
    for kfold in range(len(test_predictions)):
        for test_example in range(len(test_predictions[kfold][1])):
            prediction = test_predictions[kfold][0][test_example]
            Filepath = test_predictions[kfold][1][test_example]
            predicted_bin = np.where(prediction == np.max(prediction))[0][0]
            row = df.loc[df["Filepath"] == Filepath]
            #print(row)
            true_bin = np.where(np.delete(row.to_numpy(), 0) == 1.0)[0][0]
            #print("true bin = " + str(true_bin))
            totals[true_bin] = totals[true_bin] + 1
            #print(f"predicted bin = {predicted_bin}")
            example = clusters[0][true_bin].loc[clusters[0][true_bin]['Filepath'] == Filepath]
            if(len(example) == 0):
                print("wtf")
            elif(predicted_bin != true_bin):
                misses[true_bin] = misses[true_bin] + 1
                false_positives[predicted_bin] = false_positives[predicted_bin] + 1
                plot_prediction_vs_cluster(clusters[0][true_bin], example, predicted_cluster=clusters[0][predicted_bin], folder=folder)
            else:
                plot_prediction_vs_cluster(clusters[0][true_bin], example, folder=folder)

    
    X_axis = np.arange(len(x))
    

    
    plt.bar(X_axis - 0.2, misses, 0.4, label = 'misses')
    plt.bar(X_axis + 0.2, totals, 0.4, label = 'totals')
    
    plt.xticks(X_axis, x)
    plt.xlabel("Bin #")
    plt.ylabel("Number examples per bin")
    plt.title(f"Misses per bin \nTotal misses = {sum(misses)} Total accuracy = {100 * ((len(df) - sum(misses))/len(df))}%")
    plt.legend()
    fig_name = folder + "/hits_and_misses/" + "per_bin_misses.png"
    plt.savefig(fig_name)
    plt.close()
    
    plt.bar(X_axis - 0.2, false_positives, 0.4, label = 'false_positives')
    plt.bar(X_axis + 0.2, totals, 0.4, label = 'totals')
    
    plt.xticks(X_axis, x)
    plt.xlabel("Bin #")
    plt.ylabel("Number examples per bin")
    plt.title("False positives per bin")
    plt.legend()
    fig_name = folder + "/hits_and_misses/" + "per_bin_false_positives.png"
    plt.savefig(fig_name)
    plt.close()
    

def plot_prediction_vs_cluster(correct_cluster, example, predicted_cluster=[] , folder=None):
    # Plot the data in polar coordinates
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    
    
    r_arr = []
    theta_arr = []
    for row in correct_cluster.iterrows():
        row = row[1]
        """ to make this 2d, we plot phi as r and theta as theta"""
        r = row["phi"] 
        theta = row["theta"]
        r_arr.append(r)
        theta_arr.append(theta)
    
    ax.set_thetamin(0)
    ax.set_thetamax(360)
    ax.set_rmax(60)
    ax.scatter(theta_arr, r_arr, c='green', marker="x", label='correct bin')
    
    #plotting the falsely predicted cluster
    if(len(predicted_cluster) != 0): 
        r_arr = []
        theta_arr = []
        for row in predicted_cluster.iterrows():
            row = row[1]
            """ to make this 2d, we plot phi as r and theta as theta"""
            r = row["phi"] 
            theta = row["theta"]
            r_arr.append(r)
            theta_arr.append(theta)

        ax.scatter(theta_arr, r_arr, c='red', marker="x", label='incorrect bin')
    
        #plotting the example in blue
        ax.scatter(float(example["theta"]), float(example["phi"]), c="blue", label='true value')
        
        ax.set_title(f"Predicted and Correct bin plot")
        ax.legend(loc='upper right')
        # Add gridlines
        ax.grid(True)

        path = folder + f"/wrong_clusters_and_predictions/" + example["Filepath"].values[0].split("/")[-2]
        # Show the plot
        plt.savefig(path)
        plt.close()
    else:
        #plotting the example in blue
        ax.scatter(float(example["theta"]), float(example["phi"]), c="blue", label='true value')
        
        ax.set_title(f"Predicted and Correct bin plot")
        ax.legend(loc='upper right')
        # Add gridlines
        ax.grid(True)

        path = folder + f"/right_clusters_and_predictions/" + example["Filepath"].values[0].split("/")[-2]
        # Show the plot
        plt.savefig(path)
        plt.close()