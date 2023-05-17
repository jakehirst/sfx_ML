import numpy as np
import math as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from thickness import *
from matplotlib.image import imread


FOLDER = ""
SIMULATION = ""    
MAX_STEP = ""
MAX_UCI = ""
CRACK_TYPE = None


def get_reference_vector(folder, simulation):
    reference_vector = np.array([-10,0,0])
    reference_vector = np.array([ 0.57651577, -0.74140302, -0.34344014]) * 10
    return reference_vector




def get_crack_vectors(final_front_locations, initiation_cite):
    cv0 = final_front_locations[0] - np.array(initiation_cite)
    cv1 = final_front_locations[1] - np.array(initiation_cite)
    return [cv0, cv1]


def find_average_orientation(reference_vector, crack_vectors, centroid):
    cv0 = crack_vectors[0]
    cv1 = crack_vectors[1]
    centroid_vector = centroid / np.linalg.norm(centroid)
    cv0, cv1, reference_vector, plane_normal = project_vectors_onto_plane(cv0, cv1, reference_vector, centroid_vector)

    norm_cv0 = cv0 / np.linalg.norm(cv0)
    norm_cv1 = cv1 / np.linalg.norm(cv1)
    norm_reference = reference_vector / np.linalg.norm(reference_vector)

    orientation_0 = m.acos(np.vdot(norm_cv0, norm_reference)) * 180 /m.pi
    orientation_1 = m.acos(np.vdot(norm_cv1, norm_reference)) * 180 /m.pi


    # plot_vectors(cv0, cv1, reference_vector)

    if(np.equal(cv1, np.zeros(3)).all()):
        if(orientation_0 > 90):
            orientation_0  = 180 - orientation_0
        return orientation_0
    if(orientation_0 > 90):
        orientation_0  = 180 - orientation_0
    if(orientation_1 > 90):
        orientation_1 = 180 - orientation_1

    #print(f"orientation_0 = {orientation_0}")
    #print(f"orientation_1 = {orientation_1}")

    return orientation_0 + orientation_1 / 2

def find_orientation_edge_crack(final_front_locations, reference_vector, centroid):
    centroid_vector = centroid / np.linalg.norm(centroid)
    initial_crack_vector = final_front_locations[0] - final_front_locations[1]
    cv0, cv1, reference_vector, plane_normal = project_vectors_onto_plane(initial_crack_vector, np.array([0,0,0]), reference_vector, centroid_vector)

    norm_crack_vector = cv0 / np.linalg.norm(cv0)
    norm_reference = reference_vector / np.linalg.norm(reference_vector)
    orientation = m.acos(np.vdot(norm_crack_vector, norm_reference)) * 180 /m.pi
    if(orientation > 90):
        orientation = 180 - orientation
    #print(f"orientation = {orientation}")
    crack_vectors = [cv0, cv1]
    return orientation, crack_vectors

        

#TODO need to fix this... probably the wrong plane_normal
def project_vectors_onto_plane(cv0, cv1, reference_vector, plane_normal):
    # plane_normal = np.array([0.175828, -0.672908, -0.71852])
    # plane_normal = np.array([123.103, 44.3063, 69.9004]) / np.linalg.norm(np.array([123.103, 44.3063, 69.9004]) )
    #plane_normal = np.array([289.861, -151.741, 544.96]) / np.linalg.norm(np.array([289.861, -151.741, 544.96]) ) 

    #n is the normal vector to the plane, a is the vector to be projected
    # proj_n_a = a - np.dot(a, n)/np.linalg.norm(n)**2 * n

    # cv0 = cv0 / np.linalg.norm(cv0)
    # cv1 = cv1 / np.linalg.norm(cv1)
    # reference_vector = reference_vector / np.linalg.norm(reference_vector)



    cv0 = cv0 - np.dot(cv0, plane_normal)/np.linalg.norm(plane_normal)**2 * plane_normal
    cv1 = cv1 - np.dot(cv1, plane_normal)/np.linalg.norm(plane_normal)**2 * plane_normal
    reference_vector = reference_vector - np.dot(reference_vector, plane_normal)/np.linalg.norm(plane_normal)**2 * plane_normal


    return cv0, cv1, reference_vector, plane_normal

""" finds the angle between the two crack vectors. THIS WORKS DONT MESS WITH IT"""
def find_diff_crack_vectors(crack_vectors, reference_vector ,plane_normal):
    global CRACK_TYPE
    if(CRACK_TYPE == 0):
        hi , bye, reference_vector, plane_normal = project_vectors_onto_plane(crack_vectors[0], crack_vectors[1], reference_vector,plane_normal)
        cv0 = crack_vectors[0]
        cv1 = crack_vectors[1]
        cv0 = cv0 / np.linalg.norm(cv0)
        cv1 = cv1 / np.linalg.norm(cv1)
        #plot_vectors(cv0, cv1, reference_vector / np.linalg.norm(reference_vector), plane_normal)

        return 0.0
    else:
        cv0 , cv1, reference_vector, plane_normal = project_vectors_onto_plane(crack_vectors[0], crack_vectors[1], reference_vector,plane_normal)

        cv0 = cv0 / np.linalg.norm(cv0)
        cv1 = cv1 / np.linalg.norm(cv1)

        """ printing angle between crack vectors"""
        angle = np.arccos(np.clip(np.dot(cv0, cv1), -1.0, 1.0)) * 180 / m.pi
        #print(f"angle between = {angle}")

        """ plots the vectors in 3d space, as well as a picture of the actual crack"""
        #plot_vectors(cv0, cv1, reference_vector / np.linalg.norm(reference_vector), plane_normal)


        return angle

def plot_vectors(cv0, cv1, reference_vector, normal_vector):
    global FOLDER
    global SIMULATION

    # Create the figure and axes
    fig = plt.figure(figsize=(15,8))
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot the vectors
    ax1.quiver(0, 0, 0, cv0[0], cv0[1], cv0[2], color='r', label='cv0')
    ax1.quiver(0, 0, 0, cv1[0], cv1[1], cv1[2], color='g', label='cv1')
    ax1.quiver(0, 0, 0, reference_vector[0], reference_vector[1], reference_vector[2], color='b', label='reference_vector')
    ax1.quiver(0, 0, 0, normal_vector[0], normal_vector[1], normal_vector[2], color='y', label='normal_vector')


    # Set the limits of the plot
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([-3, 3])
    ax1.set_zlim([-3, 3])

    # Add a legend
    ax1.legend( loc='upper right')

    radius = np.linalg.norm(normal_vector)
    elevation = np.arctan2(normal_vector[2], np.sqrt(normal_vector[0]**2 + normal_vector[1]**2))
    azimuth = np.arctan2(normal_vector[1], normal_vector[0])

    # Add a title and axis labels
    plt.title('Three 3D Vectors')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    # Orient plot with spherical coordinates
    ax1.view_init(elev=np.rad2deg(elevation), azim=np.rad2deg(azimuth))
    ax1.dist = 3

    img_path = "C:\\Users\\u1056\\sfx\\images_sfx\\Visible_cracks_new_dataset_\\OG\\" + SIMULATION + f"\\Step{MAX_STEP}_UCI_{MAX_UCI}_Dynamic.png"

    if(os.path.exists(img_path)):
        ax2 = fig.add_subplot(122)
        img = imread(img_path)
        # Display the image on the second subplot
        ax2.imshow(img)
        ax2.axis("off")
        # ax2.set_xlim([100, 200])
        # ax2.set_ylim([100, 200])
        ax2.set_title('Title for plot 2')


    # Show the plot
    plt.show()

""" gets the centroid for all of the outer surface nodes on the parietal bone. acts as a vector for a plane
    normal as well as a starting point for the crack vectors cv0 and cv1"""
def get_centroid_of_outer_surface_nodes(folder_path, simulation, max_step_uci):
    outer_surface_nodes = []
    outer = False
    node_locations = {}
    locations=False

    f = open(folder_path + simulation + f"\\Step{max_step_uci[0]}_UCI_{max_step_uci[1]}_Dynamic.inp",'r')
    for line in f.readlines(): 
        #""" getting all node locations """
        if(locations and line.startswith("*Element")):
            locations = False
        if(locations):
            node_and_location = line.replace(" ", "").replace("\n", "").split(",")
            node_locations[node_and_location[0]] = [float(node_and_location[1]), float(node_and_location[2]), float(node_and_location[3])]        
        if(line.startswith("*Node") and len(node_locations.keys()) == 0):
            locations = True

        #""" getting the nodes on the outer side """
        if(outer and line.startswith("*")):
            outer = False
        if(outer):
            nodes = line.replace(" ", "").replace("\n", "").split(",")
            outer_surface_nodes.extend(nodes)
        if(line.startswith("*Nset, nset=SKULL_EDGE, instance=PART-1-1")):
            outer = True
        if(outer_surface_nodes.__contains__("")):outer_surface_nodes.remove("")


    centroid = []
    num = 0
    for osn in outer_surface_nodes:
        if(centroid == []):
            centroid = node_locations[osn]
        else:
            centroid[0] += node_locations[osn][0] 
            centroid[1] += node_locations[osn][1] 
            centroid[2] += node_locations[osn][2] 
        num +=1
    
    centroid = np.array(centroid)/num
    return centroid


def find_orientation(folder, simulation, final_front_locations, initiation_cite, crack_type):
    global FOLDER
    global SIMULATION
    global MAX_STEP
    global MAX_UCI
    global CRACK_TYPE

    CRACK_TYPE = crack_type
    FOLDER = folder
    SIMULATION = simulation
    max_step_uci = get_max_dynamic_step(folder, simulation)
    MAX_STEP = str(max_step_uci[0])
    MAX_UCI = str(max_step_uci[1])

    centroid = get_centroid_of_outer_surface_nodes(folder, simulation, max_step_uci)
    norm_centroid = centroid / np.linalg.norm(centroid)

    reference_vector = get_reference_vector(folder, simulation)
    crack_vectors = get_crack_vectors(final_front_locations, centroid)
    if(crack_type == 1):
        average_orientation = find_average_orientation(reference_vector, crack_vectors, centroid)
    elif(crack_type == 0):
        average_orientation, crack_vectors = find_orientation_edge_crack(final_front_locations, reference_vector, centroid)
    angle_between_crack_vectors = find_diff_crack_vectors(crack_vectors, reference_vector, norm_centroid)
    return average_orientation, angle_between_crack_vectors

""" 1 is interior crack, 0 is an edge crack """
def get_crack_type(folder, simulation):
    filepath = folder + simulation + "\\Analyze_Crack_Package.json"
    f = open(filepath)
    data = json.load(f)
    crack_type = data["NUM_CRACKS"]-1
    return crack_type