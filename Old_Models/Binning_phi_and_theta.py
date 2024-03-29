import pandas as pd
import numpy as np
import math as m
from keras.utils import to_categorical
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import seaborn as sns
from sklearn import metrics


"""Bins phi and theta into num_phi_bins and num_theta_bins"""
def Bin_phi_and_theta(df, num_phi_bins, num_theta_bins):
    theta_spacing = 361 / num_theta_bins
    phi_spacing = 60 / num_phi_bins
    bin_num = 0
    bin = np.zeros(len(df))
    phis = np.array(df["phi"])
    thetas = np.array(df["theta"])
    y_col_values = []
    bins_and_values = {}

    for j in range(num_theta_bins*num_phi_bins):
        y_col_values.append(str(int(j)))

    for phi_bin in range(num_phi_bins):
        phi_low = phi_bin * phi_spacing #creating a high and low value for phi for this respective bin
        phi_high = (phi_bin + 1) * phi_spacing  
        print(f"phi low and high = {phi_low} and {phi_high}")

        for theta_bin in range(num_theta_bins):
            theta_low = theta_bin * theta_spacing #creating a high and low value for theta for this respective bin
            theta_high = (theta_bin + 1) * theta_spacing
            print(f"theta low and high = {theta_low} and {theta_high}")

            for i in range(len(df)):
                if(phis[i] >= phi_low and phis[i] < phi_high and thetas[i] >= theta_low and thetas[i] < theta_high):
                    if(bin[i] > 0): 
                        print("This index was catergorized twice")
                        break
                    bin[i] = str(int(bin_num))
                
            bins_and_values[bin_num] = {"phi":[phi_low, phi_high], "theta":[theta_low, theta_high]}
            bin_num += 1
            
    
    bin = to_categorical(bin)
    bin_df = pd.DataFrame(bin)
    
    bin_df.columns = y_col_values
    df = pd.concat([df, bin_df], axis=1)

    df = df.drop("phi", axis=1)
    df = df.drop("theta", axis=1)

    return df, y_col_values, bins_and_values

"""Bins phi and theta into num_phi_bins and num_theta_bins. the first couple of bins are combined so a solid circle is in the middle"""
def Bin_phi_and_theta_center_target(df, num_phi_bins, num_theta_bins):
    theta_spacing = 361 / num_theta_bins
    phi_spacing = 60 / num_phi_bins
    bin_num = 0
    bin = np.zeros(len(df))
    phis = np.array(df["phi"])
    thetas = np.array(df["theta"])
    y_col_values = ['0']
    bins_and_values = {}

    #binning all values in the center circle first
    for i in range(len(df)):
        if(phis[i] < phi_spacing):
            bin[i] = str(int(bin_num))
    bins_and_values[bin_num] = {"phi":[0, phi_spacing], "theta":[0, 361]}
    bin_num += 1


    for phi_bin in range(1, num_phi_bins):
        phi_low = phi_bin * phi_spacing #creating a high and low value for phi for this respective bin
        phi_high = (phi_bin + 1) * phi_spacing  
        #print(f"phi low and high = {phi_low} and {phi_high}")

        for theta_bin in range(num_theta_bins):
            theta_low = theta_bin * theta_spacing #creating a high and low value for theta for this respective bin
            theta_high = (theta_bin + 1) * theta_spacing
            # print(f"theta low and high = {theta_low} and {theta_high}")
            # print(f"bin number = {bin_num}")

            #goes through all of the examples and sees if any examples fall within the current bin
            for i in range(len(df)):
                if(phis[i] >= phi_low and phis[i] < phi_high and thetas[i] >= theta_low and thetas[i] < theta_high):
                    if(bin[i] > 0): 
                        print("This index was catergorized twice")
                        break
                    print(f"phi = {phis[i]} theta = {thetas[i]} bin = {bin_num}")
                    bin[i] = str(int(bin_num))
                
            bins_and_values[bin_num] = {"phi":[phi_low, phi_high], "theta":[theta_low, theta_high]}
            y_col_values.append(str(int(bin_num)))
            bin_num += 1
            
    #changes the bin array to a categorical matrix with 1's in each examples corresponding bin column
    bin = to_categorical(bin)
    while(bin.shape[1] < int(y_col_values[-1]) + 1):
        bin = np.hstack((bin,np.zeros((bin.shape[0],1))))
    
    bin_df = pd.DataFrame(bin)
    
    bin_df.columns = y_col_values
    df = pd.concat([df, bin_df], axis=1)

    # for i in range(len(df)):
    #     print("\nphi = " + str(df.iloc[i]["phi"]))
    #     print("theta = " + str(df.iloc[i]["theta"]))
    #     print("bin = " + str(np.where(bin[i]== 1.0)[0][0]))

    df = df.drop("phi", axis=1)
    df = df.drop("theta", axis=1)

    
    return df, y_col_values, bins_and_values

""" bins phi and theta into just theta bins, no phi. so it looks like a bunch of pizza slices."""
def Bin_just_theta(df, num_theta_bins):
    theta_spacing = 361 / num_theta_bins
    bin_num = 0
    bin = np.zeros(len(df))
    thetas = np.array(df["theta"])
    y_col_values = []
    bins_and_values = {}
    bins_and_frequencies = {}
    
    for theta_bin in range(num_theta_bins):
        theta_low = theta_bin * theta_spacing #creating a high and low value for theta for this respective bin
        theta_high = (theta_bin + 1) * theta_spacing

        for i in range(len(df)):
            if(thetas[i] >= theta_low and thetas[i] < theta_high):
                if(bin[i] > 0): 
                    print("This index was catergorized twice")
                    break
                print(f"theta = {thetas[i]} bin = {bin_num}")
                assigned_bin = str(int(bin_num))
                bin[i] = assigned_bin
                
                if(not bins_and_frequencies.keys().__contains__(assigned_bin)):
                    bins_and_frequencies[assigned_bin] = 0
                else:
                    bins_and_frequencies[assigned_bin] = bins_and_frequencies[assigned_bin] + 1
            
        bins_and_values[bin_num] = {"phi":[0, 60], "theta":[theta_low, theta_high]}
        y_col_values.append(str(int(bin_num)))
        bin_num += 1
            
    
    bin = to_categorical(bin)
    bin_df = pd.DataFrame(bin)
    
    bin_df.columns = y_col_values
    df = pd.concat([df, bin_df], axis=1)

    # for i in range(len(df)):
    #     print("\nphi = " + str(df.iloc[i]["phi"]))
    #     print("theta = " + str(df.iloc[i]["theta"]))
    #     print("bin = " + str(np.where(bin[i]== 1.0)[0][0]))

    df = df.drop("phi", axis=1)
    df = df.drop("theta", axis=1)

    return df, y_col_values, bins_and_values
    
""" bins phi and theta into just phi bins, no theta. so it looks like a bunch of circles."""
def Bin_just_phi(df, num_phi_bins):
    phi_spacing = 60 / num_phi_bins
    bin_num = 0
    bin = np.zeros(len(df))
    phis = np.array(df["phi"])
    y_col_values = []
    bins_and_values = {}
    bins_and_frequencies = {}


    for phi_bin in range(0, num_phi_bins):
        phi_low = phi_bin * phi_spacing #creating a high and low value for phi for this respective bin
        phi_high = (phi_bin + 1) * phi_spacing  
        #print(f"phi low and high = {phi_low} and {phi_high}")
        for i in range(len(df)):
            if(phis[i] >= phi_low and phis[i] < phi_high):
                if(bin[i] > 0): 
                    print("This index was catergorized twice")
                    break
                print(f"phi = {phis[i]} bin = {phi_bin}")
                assigned_bin = str(int(phi_bin))
                bin[i] = assigned_bin
                
                if(not bins_and_frequencies.keys().__contains__(assigned_bin)):
                    bins_and_frequencies[assigned_bin] = 0
                else:
                    bins_and_frequencies[assigned_bin] = bins_and_frequencies[assigned_bin] + 1
                    
                                
        bins_and_values[bin_num] = {"phi":[phi_low, phi_high], "theta":[0, 361]}
        y_col_values.append(str(int(phi_bin)))
        bin_num += 1
        
    
    bin = to_categorical(bin)
    bin_df = pd.DataFrame(bin)
    
    bin_df.columns = y_col_values
    df = pd.concat([df, bin_df], axis=1)

    # for i in range(len(df)):
    #     print("\nphi = " + str(df.iloc[i]["phi"]))
    #     print("theta = " + str(df.iloc[i]["theta"]))
    #     print("bin = " + str(np.where(bin[i]== 1.0)[0][0]))

    df = df.drop("phi", axis=1)
    df = df.drop("theta", axis=1)

    return df, y_col_values, bins_and_values

def turn_filepath_to_nparray(x):
    images = np.empty(0)
    for i in range(len(x)):
        path = x[i]
        img = Image.open(path)
        nump = asarray(img, dtype="float32")/255.0
        x[i] = nump
    return x

"""plots a heat map of a particular example's prediction probabilities across each bin and 
   plots the location of the example as a comparison"""
def make_sphere(bins_and_values, test_prediction ,true_value, filepath):

    heats = test_prediction
    #theta inclination angle
    #phi azimuthal angle
    n_phi = 60 # number of values for phi
    n_theta = 360  # number of values for theta
    r = 2        #radius of sphere

    phi, theta = np.mgrid[0.0:0.3*np.pi:n_phi*1j, 0.0:2.0*np.pi:n_theta*1j]

    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)

    # mimic the input array
    # array columns phi, theta, value
    # first n_theta entries: phi=0, second n_theta entries: phi=0.0315..
    inp = []

    for j in theta[0,:]:
        for i in phi[:,0]:
            for bin_num in bins_and_values.keys():

                min_phi = bins_and_values[bin_num]["phi"][0]
                max_phi = bins_and_values[bin_num]["phi"][1]
                min_theta = bins_and_values[bin_num]["theta"][0]
                max_theta = bins_and_values[bin_num]["theta"][1]

                current_phi = i*180/np.pi
                current_theta = j*180/np.pi
                #print(current_phi)

                if(current_theta > 360): 
                    current_theta = current_theta - 360
                if(current_phi > 60): 
                    current_phi = current_phi - 60

                if(current_phi >= min_phi and current_phi < max_phi and current_theta >= min_theta and current_theta < max_theta):
                    val = heats[bin_num]
                    inp.append([j, i, val])
                    # print(f"Sent into bin {bin_num}")
                    # print(f"phi = {current_phi} falls inside {min_phi} and {max_phi}")
                    # print(f"theta = {current_theta} falls inside {min_theta} and {max_theta}")
                    break
                if(bin_num == list(bins_and_values.keys())[-1]):
                    print(f"phi = {current_phi}")
                    print(f"theta = {current_theta}")
                    print("didnt find bin for this example")
                    


    # for j in theta[0,:]:
    #     for i in phi[:,0]:
    #         val = 0.7+np.cos(j)*np.sin(i+np.pi/4.)# put something useful here
    #         inp.append([j, i, val])

    inp = np.array(inp)
    #print(inp.shape)
    #print(inp[49:60, :])

    #reshape the input array to the shape of the x,y,z arrays. 
    c = inp[:,2].reshape((n_theta,n_phi)).T
    #print(z.shape)
    #print(c.shape)
    
    #in lab
    # point_theta = int(true_value.split("\\")[-2].split("_")[-1])* np.pi/180
    # point_phi = int(true_value.split("\\")[-2].split("_")[-3])* np.pi/180
    
    #at home
    point_theta = int(true_value.split("/")[-2].split("_")[-1])* np.pi/180
    point_phi = int(true_value.split("/")[-2].split("_")[-3])* np.pi/180

    point_x = (r+.1)*np.sin(point_phi)*np.cos(point_theta)
    point_y = (r+.1)*np.sin(point_phi)*np.sin(point_theta)
    point_z = (r+.1)*np.cos(point_phi)

    #Set colours and render
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    #use facecolors argument, provide array of same shape as z
    # cm.<cmapname>() allows to get rgba color from array.
    # array must be normalized between 0 and 1
    ax.scatter(np.array(point_x), np.array(point_y), np.array(point_z), marker='x',c="blue", label="true value")
    surf = ax.plot_surface(
        x,y,z,  rstride=1, cstride=1, facecolors=cm.Reds(c), cmap=cm.Reds, alpha=0.5, linewidth=1) 
    ax.set_xlim([-2.2,2.2])
    ax.set_ylim([-2.2,2.2])
    ax.set_zlim([0,4.4])
    ax.set_aspect("auto")
    ax.legend()
    ax.view_init(90, 90)
    #ax.plot_wireframe(x, y, z, color="k") #not needed?!
    
    #in lab
    # if(not os.path.isdir(filepath.split("\\fold")[0])):
    #     os.mkdir(filepath.split("\\fold")[0])
    # if not os.path.isdir(filepath.removesuffix("\\")):
    #     os.mkdir(filepath.removesuffix("\\"))
    # fig.colorbar(surf)
    # plt.savefig(filepath + true_value.split("\\")[-2] +".png")
    
    #at home

    fig.colorbar(surf)
    plt.savefig(filepath + true_value.split("/")[-2] +".png")
    
    plt.close(fig)
    #plt.show()

""" shows a bar chart of the nubmer of examples per bin and the number of misses per bin. """
def Plot_Bins_and_misses(bins_and_values, test_predictions, df, folder):
    folder = folder + "/fold5/"
    #TODO: df actually needs to be the original df with all of the examples in it
    if(not os.path.exists(folder.removesuffix("fold5/") + "/hits_and_misses")):
        os.mkdir(folder.removesuffix("fold5/") + "/hits_and_misses")
    df = df.drop("index", axis=1)
    
    if(df.columns.__contains__("image_path")):
        df.rename(columns={"image_path": "Filepath"}, inplace=True)
    
    if(df.columns.__contains__("phi" or "theta" or "height")):
        bins = []
        y_col_values = []
        for row in df.iterrows():
            for bin in bins_and_values.keys():
                phi_low = bins_and_values[bin]["phi"][0]
                phi_high = bins_and_values[bin]["phi"][1]
                theta_low = bins_and_values[bin]["theta"][0]
                theta_high = bins_and_values[bin]["theta"][1]
                if((phi_low <= row[1]["phi"] < phi_high) and (theta_low <= row[1]["theta"] < theta_high)):
                    bins.append(bin)
                    break
        for bin in bins_and_values.keys():
            y_col_values.append(str(bin))
        
        
        # for i in range(len(df)): 
        #     row = df.iloc[i] 
        #     print(f"\n phi = {row['phi']} \n theta = {row['theta']} \n bin_vals = {bins_and_values[bins[i]]}\n")
        bins = to_categorical(bins)
        bin_df = pd.DataFrame(bins)
        bin_df.columns = y_col_values
        df = pd.concat([df, bin_df], axis=1)
        labels = df.columns[df.columns.to_list().index('0'):].to_list()
        if(df.columns.__contains__("phi")): df = df.drop("phi", axis=1)
        if(df.columns.__contains__("theta")): df = df.drop("theta", axis=1)
        if(df.columns.__contains__("height")): df = df.drop("height", axis=1)
        
    
    # df = df[labels]
    misses = []
    totals = []
    false_positives = []
    x = []
    for i in range(len(labels)):
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
            
            if(predicted_bin != true_bin):
                misses[true_bin] = misses[true_bin] + 1
                false_positives[predicted_bin] = false_positives[predicted_bin] + 1
    
    X_axis = np.arange(len(x))
    

    plt.bar(X_axis - 0.2, misses, 0.4, label = 'misses')
    plt.bar(X_axis + 0.2, totals, 0.4, label = 'totals')
    
    plt.xticks(X_axis, x)
    plt.xlabel("Bin #")
    plt.ylabel("Number examples per bin")
    plt.title(f"Misses per bin \nTotal misses = {sum(misses)} Total accuracy = {100 * ((len(df) - sum(misses))/len(df))}%")
    plt.legend()
    fig_name = folder.removesuffix("fold5/") + "/hits_and_misses/per_bin_misses.png"
    plt.savefig(fig_name)
    plt.close()
    
    plt.bar(X_axis - 0.2, false_positives, 0.4, label = 'false_positives')
    plt.bar(X_axis + 0.2, totals, 0.4, label = 'totals')
    
    plt.xticks(X_axis, x)
    plt.xlabel("Bin #")
    plt.ylabel("Number examples per bin")
    plt.title("False positives per bin")
    plt.legend()
    fig_name = folder.removesuffix("fold5/") + "hits_and_misses/per_bin_false_positives.png"
    plt.savefig(fig_name)
    plt.close()
    return df

""" plots a confusion matrix with the text data from all kfolds """  
def confusion_matrix(test_predictions, df, folder):
    y_pred = []
    y_true = []
    for fold in range(len(test_predictions)):
        test_fold = test_predictions[fold]
        for i in range(len(test_fold[1])):
            image_path = test_fold[1][i]
            prediction_cluster = np.argmax(test_fold[0][i])
            print(f"predicted cluster = {prediction_cluster}")
            #gets the row of the filepath and turns it into a dict
            true_cluster_row = df.loc[df['Filepath'] == image_path] 
            true_cluster = np.where((true_cluster_row.to_numpy())[0] == 1.0)[0][0] - 1
            print(f"true cluster = {true_cluster}")
            y_pred.append(prediction_cluster)
            y_true.append(true_cluster)
            
    
    cm = metrics.confusion_matrix(np.asarray(y_true), np.asarray(y_pred))
    axes = np.arange(cm.shape[0])
    
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(cm,
                        index = axes, 
                        columns = axes)
    #Plotting the confusion matrix
    plt.figure(figsize=(9,9))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    figname = "/hits_and_misses/Cumulative_Confusion_matrix.png"
    if(not os.path.exists(folder + "/hits_and_misses")):
        os.mkdir(folder + "/hits_and_misses")
    
    plt.savefig(folder + figname)
    plt.close()
    return
    

"""plots a heat map of a particular example's prediction probabilities across each bin and 
   plots the location of the example as a comparison"""
def make_circle(bins_and_values, test_prediction ,true_value, filepath):
    heats = test_prediction
    #theta inclination angle
    #phi azimuthal angle
    n_phi = 60 # number of values for phi
    n_theta = 360  # number of values for theta
    r = n_phi     #radius of sphere

    phi, theta = np.mgrid[0.0:0.3*np.pi:n_phi*1j, 0.0:2.0*np.pi:n_theta*1j]
    # convert to Cartesian coordinates
    x = phi * np.cos(theta)
    y = phi * np.sin(theta)

    # mimic the input array
    # array columns phi, theta, value
    # first n_theta entries: phi=0, second n_theta entries: phi=0.0315..
    inp = []

    for j in theta[0,:]:
        for i in phi[:,0]:
            for bin_num in bins_and_values.keys():

                min_phi = bins_and_values[bin_num]["phi"][0]
                max_phi = bins_and_values[bin_num]["phi"][1]
                min_theta = bins_and_values[bin_num]["theta"][0]
                max_theta = bins_and_values[bin_num]["theta"][1]

                current_phi = i*180/np.pi
                current_theta = j*180/np.pi
                #print(current_phi)

                if(current_theta > 360): 
                    current_theta = current_theta - 360
                if(current_phi > 60): 
                    current_phi = current_phi - 60

                if(current_phi >= min_phi and current_phi < max_phi and current_theta >= min_theta and current_theta < max_theta):
                    val = heats[bin_num]
                    inp.append([j, i, val])
                    # print(f"Sent into bin {bin_num}")
                    # print(f"phi = {current_phi} falls inside {min_phi} and {max_phi}")
                    # print(f"theta = {current_theta} falls inside {min_theta} and {max_theta}")
                    break
                if(bin_num == list(bins_and_values.keys())[-1]):
                    print(f"phi = {current_phi}")
                    print(f"theta = {current_theta}")
                    print("didnt find bin for this example")
                    
    # for j in theta[0,:]:
    #     for i in phi[:,0]:
    #         val = 0.7+np.cos(j)*np.sin(i+np.pi/4.)# put something useful here
    #         inp.append([j, i, val])

    inp = np.array(inp)

    #reshape the input array to the shape of the x,y,z arrays. 
    c = inp[:,2].reshape((n_theta,n_phi)).T
    
    #at home
    point_theta = int(true_value.split("/")[-2].split("_")[-1])* np.pi/180
    point_phi = int(true_value.split("/")[-2].split("_")[-3])* np.pi/180
    
    point_x = point_phi * np.cos(point_theta)
    point_y = point_phi * np.sin(point_theta)
    
    # Create the polar heatmap plot with a red colormap
    # fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(projection='polar'))
    
    # ax.set_ylim(0, np.max(phi))
    # color = ax.pcolormesh(theta, phi, c, cmap='Reds_r', vmin=np.max(c), vmax=np.min(c))
    # ax.scatter(np.array(point_x), np.array(point_y), marker='x',c="cyan", s= 150,  label="true value")
    # # ax.plot(point_x, point_y, marker='o',c="cyan", markersize= 150,  label="true value")

    # fig.text(.5, .001, f"prediction = {test_prediction.tolist()}", ha='center')

    # cbar = plt.colorbar(color)
    # # Set the colorbar label and range
    # cbar.set_label('Probabilities')
    # plt.show()
    
    # # plt.savefig(filepath + true_value.split("/")[-2] +".png")
    
    # plt.close(fig)
    
    print(np.max(phi))
    print(point_x)
    print(point_y)
    # Create the polar heatmap plot with a red colormap
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15,10), subplot_kw=dict(projection='polar'), sharey=True)
    ax1.set_ylim(0, np.max(phi))
    color = ax1.pcolormesh(theta, phi, c, cmap='Reds_r', vmin=np.max(c), vmax=np.min(c), zorder=-1)
    ax1.set_ylim(0, np.max(phi))
    ax2.scatter(np.array(point_theta), np.array(point_phi), marker='x',c="cyan", s= 150,  label="true value", zorder=200.5)

    fig.text(.5, .001, f"prediction = {test_prediction.tolist()}", ha='center')

    cbar = plt.colorbar(color)
    # Set the colorbar label and range
    cbar.set_label('Probabilities')
    # plt.show()
    
    plt.savefig(filepath + true_value.split("/")[-2] +".png")
    
    plt.close(fig)

            
    

#make_sphere()
    