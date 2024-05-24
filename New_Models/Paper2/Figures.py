
'''Figures showing UQ for impact site'''
import sys
sys.path.append('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/New_Models')
import numpy as np
import pandas as pd
from GPR import *
from Bagging_models import *
from Single_UQ_models import *
from matplotlib.patches import Ellipse
from scipy import stats
import matplotlib.colors as mcolors


'''
Plots the RPA bone as grey scatter points
also plots the mean prediction, and the true value of the impact site
Finally, plots cyan scatter points of samples of the distribution of the predictions.
'''
def plot_impact_site_with_uncertainty(CIs_wanted, x_pred, x_std, y_pred, y_std, x_true, y_true, saving_path=None):
    #material basis vectors for RPA bone
    Material_X = np.array([-0.87491124, -0.44839274,  0.18295974])
    Material_Y = np.array([ 0.23213791, -0.71986519, -0.65414532])
    Material_Z = np.array([ 0.42502036, -0.5298472,   0.7339071 ])
    #Center of mass of the RPA bone in abaqus basis
    CM = np.array([106.55,72.79,56.64])
    # #Ossification center of the RPA bone in abaqus basis
    OC = np.array([130.395996,46.6063,98.649696])
    
    parietal_node_location_df = pd.read_csv('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/parital_node_locations.csv')
    RPA_x = parietal_node_location_df['RPA nodes x']; RPA_y = parietal_node_location_df['RPA nodes y']; RPA_z = parietal_node_location_df['RPA nodes z']
    
    #converting the RPA node locations into Jimmy's reference frame
    RPA_x, RPA_y, RPA_z = convert_coordinates_to_new_basis(Material_X, Material_Y, Material_Z, CM, RPA_x, RPA_y, RPA_z)
    
    
    '''3d plot of RPA nodes and the predicted x, y and z values in '''
    # Create a 3D figure
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    
    ax.scatter(RPA_z, RPA_x, RPA_y, c='grey', alpha=0.025)
    # x_dist = np.random.normal(x_predictions[i], x_stds[i], 100)
    # y_dist = np.random.normal(y_predictions[i], y_stds[i], 100)
    num_points = 5000
    x_dist = np.random.normal(x_pred, x_std, num_points)
    y_dist = np.random.normal(y_pred, y_std, num_points)
    
    point_size = 50
    
    z_val=15
    z_dist = np.random.normal(z_val, 3, num_points) #TODO find a good value for z
    ax.scatter(z_dist, x_dist, y_dist, c='cyan', alpha=0.01)
    ax.scatter(z_val, x_pred, y_pred, c='blue', label='Mean predicted impact location', s=point_size)
    ax.scatter(z_val, x_true, y_true, c='orange', label='True impact location', s=point_size)
    
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_xlim3d(-60, 60)
    ax.set_ylim3d(-80, 80)
    ax.set_zlim3d(-60, 60)
    ax.set_title('GPR prediction for impact site in right parietal bone', fontweight='bold')
    ax.legend()
    
    normal_vector = np.array([0,1,0])
    # Calculate the azimuthal and polar angles
    azimuth = np.arctan2(normal_vector[1], normal_vector[0])
    polar = np.arccos(normal_vector[2])
    # Convert angles to degrees
    azimuth = np.degrees(azimuth)
    polar = np.degrees(polar)
    # # Set the camera direction using the angles (customizing a bit)
    ax.view_init(elev=-5, azim=azimuth + 270)
    # # Show the plot
    if(saving_path == None):
        plt.show()
    else:
        plt.savefig(saving_path)
    plt.close()
    return


'''
Plots the RPA bone as grey scatter points
also plots the mean prediction, and the true value of the impact site
Finally, plots cyan scatter points of samples of the distribution of the predictions.
'''
def plot_impact_site_with_uncertainty_2D(CIs_wanted, x_pred, x_std, y_pred, y_std, x_true, y_true, saving_path=None):
    #material basis vectors for RPA bone
    Material_X = np.array([-0.87491124, -0.44839274,  0.18295974])
    Material_Y = np.array([ 0.23213791, -0.71986519, -0.65414532])
    Material_Z = np.array([ 0.42502036, -0.5298472,   0.7339071 ])
    #Center of mass of the RPA bone in abaqus basis
    CM = np.array([106.55,72.79,56.64])
    # #Ossification center of the RPA bone in abaqus basis
    OC = np.array([130.395996,46.6063,98.649696])

    # Assuming convert_coordinates_to_new_basis is a function you've defined elsewhere
    # and it correctly converts 3D coordinates to 2D in this new context.

    # Load parietal bone node locations (already done in your original code)
    parietal_node_location_df = pd.read_csv('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/parital_node_locations.csv')
    RPA_x = parietal_node_location_df['RPA nodes x']
    RPA_y = parietal_node_location_df['RPA nodes y']
    RPA_z = parietal_node_location_df['RPA nodes z']
    
    # No need to convert Z coordinates or use them in plotting
    RPA_x, RPA_y, RPA_z = convert_coordinates_to_new_basis(Material_X, Material_Y, Material_Z, CM, RPA_x, RPA_y, RPA_z)

    # Your predictive model code remains the same, just ensure it doesn't include Z predictions

    # Create a 2D figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot for the parietal bone nodes
    ax.scatter(RPA_x, RPA_y, c='grey', alpha=.15, label='Parietal bone nodes')

    num_points = 5000
    x_dist = np.random.normal(x_pred, x_std, num_points)
    y_dist = np.random.normal(y_pred, y_std, num_points)

    point_size = 50

    # Scatter plot for the distribution of predicted impact locations
    ax.scatter(x_dist, y_dist, c='cyan', alpha=0.01)

    # Scatter plot for the mean predicted and true impact locations
    ax.scatter(x_pred, y_pred, c='blue', label='Mean predicted impact location', s=point_size)
    ax.scatter(x_true, y_true, c='orange', label='True impact location', s=point_size)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('GPR prediction for impact site in right parietal bone', fontweight='bold')
    ax.set_ylim(-60, 60)
    ax.set_xlim(-80, 80)
    ax.legend()
    

    # Show or save the plot
    if saving_path is None:
        plt.show()
    else:
        plt.savefig(saving_path)
    plt.close()
    
    
def plot_impact_site_with_uncertainty_2D_with_ellipse_before_recalibration(CIs_wanted, x_pred, x_std, y_pred, y_std, x_true, y_true, saving_path=None):
    def plot_confidence_ellipses(ax, x_pred, y_pred, x_std, y_std, confidence_intervals):
        """
        Adds confidence interval ellipses to the provided axes object.
        
        :param ax: The matplotlib axes to add ellipses to.
        :param x_pred: The x coordinate of the predicted mean.
        :param y_pred: The y coordinate of the predicted mean.
        :param x_std: The standard deviation in the x direction.
        :param y_std: The standard deviation in the y direction.
        :param confidence_intervals: A list of confidence interval percentages.
        """
        # colors = list(mcolors.CSS4_COLORS.values())  # Get a list of color names
        # np.random.shuffle(colors)  # Shuffle the list to randomize color selection
        colors = ['red','yellow','green']
        # For a normal distribution, the number of standard deviations for a given confidence interval can be found
        # by using the inverse of the cumulative distribution function (CDF), often referred to as the z-value.
        # The z-value times the standard deviation gives you the radius of the confidence interval for that percentage.
        for i, confidence in enumerate(confidence_intervals):
            z_value = np.abs(np.array([stats.norm.ppf((1 + (confidence / 100)) / 2)]))
            width = z_value * x_std * 2  # 2x for the diameter
            height = z_value * y_std * 2  # 2x for the diameter
            # ellipse = Ellipse((x_pred, y_pred), width, height, edgecolor='blue', facecolor='none', label=f'{confidence}% CI')
            # ax.add_patch(ellipse) 
            ellipse_color = colors[i % len(colors)]  # Use modulo to loop over colors if necessary
            ellipse = Ellipse((x_pred, y_pred), width, height, edgecolor=ellipse_color, facecolor='none', label=f'{confidence}% Confidence interval', linewidth=3)
            ax.add_patch(ellipse)
              
    #material basis vectors for RPA bone
    Material_X = np.array([-0.87491124, -0.44839274,  0.18295974])
    Material_Y = np.array([ 0.23213791, -0.71986519, -0.65414532])
    Material_Z = np.array([ 0.42502036, -0.5298472,   0.7339071 ])
    #Center of mass of the RPA bone in abaqus basis
    CM = np.array([106.55,72.79,56.64])
    # #Ossification center of the RPA bone in abaqus basis
    OC = np.array([130.395996,46.6063,98.649696])

    # Assuming convert_coordinates_to_new_basis is a function you've defined elsewhere
    # and it correctly converts 3D coordinates to 2D in this new context.

    # Load parietal bone node locations (already done in your original code)
    parietal_node_location_df = pd.read_csv('/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/parital_node_locations.csv')
    RPA_x = parietal_node_location_df['RPA nodes x']
    RPA_y = parietal_node_location_df['RPA nodes y']
    RPA_z = parietal_node_location_df['RPA nodes z']
    
    # No need to convert Z coordinates or use them in plotting
    RPA_x, RPA_y, RPA_z = convert_coordinates_to_new_basis(Material_X, Material_Y, Material_Z, CM, RPA_x, RPA_y, RPA_z)

    # Your predictive model code remains the same, just ensure it doesn't include Z predictions


    # Existing code for loading data and converting coordinates remains the same...

    # Create a 2D figure
    fig, ax = plt.subplots(figsize=(20, 16))

    # Scatter plot for the parietal bone nodes
    ax.scatter(RPA_x, RPA_y, s=100,  c='grey', alpha=.15, label='Parietal bone', zorder=1)

    point_size = 100

    # Scatter plot for the mean predicted and true impact locations
    ax.scatter(x_pred, y_pred, c='cyan', label='Mean prediction', s=point_size*2, zorder=3)
    ax.scatter(x_true, y_true, c='orange', label='True impact location', s=point_size*2, zorder=3)

    # Add confidence interval ellipses
    plot_confidence_ellipses(ax, x_pred, y_pred, x_std, y_std, CIs_wanted)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('GPR prediction for impact site in right parietal bone', fontweight='bold')
    ax.set_ylim(-60, 60)
    ax.set_xlim(-80, 80)
    ax.legend()

    # Show or save the plot
    if saving_path is None:
        plt.show()
    else:
        plt.savefig(saving_path)
    plt.close()