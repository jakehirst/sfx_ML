from front_locations import *
import math as m
import pandas as pd

# """ gets the total crack length based on euclidean distance between each front location """
# def get_distances_from_initiation(simulation_folder, simulation):
#     prev_front_0 = get_initiation_cite(simulation_folder, simulation)
#     prev_front_1 = get_initiation_cite(simulation_folder, simulation)
#     front_locations = get_all_front_locations(simulation_folder, simulation)
#     dist_frt_0 = 0.0
#     dist_frt_1 = 0.0

#     for locations in front_locations:
#         dist_frt_0 += get_arc_len(prev_front_0, locations[2])
#         dist_frt_1 += get_arc_len(prev_front_1, locations[3])
#         prev_front_0 = locations[2]
#         prev_front_1 = locations[3]

#     return dist_frt_0, dist_frt_1


""" gets the total crack length based on euclidean distance between each front location """
def get_distances_from_initiation(simulation_folder, simulation):
    prev_front_0 = get_initiation_cite(simulation_folder, simulation)
    prev_front_1 = get_initiation_cite(simulation_folder, simulation)
    front_locations = get_all_front_locations(simulation_folder, simulation)
    dist_frt_0 = 0.0
    dist_frt_1 = 0.0

    for locations in front_locations:
        dist_frt_0 += get_arc_len(prev_front_0, locations[2]) 
        if get_arc_len(prev_front_0, locations[2]) > .04:
            dist_frt_0 -= get_arc_len(prev_front_0, locations[2])
            locations[3]=locations[2]
            locations[2]= prev_front_0
            #print('happened 0')

        dist_frt_1 += get_arc_len(prev_front_1, locations[3])
        if get_arc_len(prev_front_1, locations[3]) > .04: # if front 2 ends it will insert a random previous startin point, this changes that entry to the previous entry.
            dist_frt_1 -= get_arc_len(prev_front_1, locations[3]) 
            locations[3]=prev_front_1
            #print('happened 1')
        prev_front_0 = locations[2]
        prev_front_1 = locations[3]
        front_0_final_location = locations[2]
        front_1_final_location = locations[3]

    #print('hillooo')
    return dist_frt_0, dist_frt_1,front_0_final_location,front_1_final_location


""" Turns cartesian coordinates [x,y,z] into spherical coordinates [r, phi, theta] """
#https://vvvv.org/blog/polar-spherical-and-geographic-coordinates#:~:text=The%20definition%20of%20the%20spherical%20coordinates%20has%20two,longitude%20does%20not%20match%20with%20the%20two%20angles.
def cart_to_spherical(cartesian):
    x = cartesian[0]; y = cartesian[1]; z = cartesian[2]
    azimuthal = m.atan2(y, x) #theta
    r = m.sqrt(x**2 + y**2 + z**2)
    polar = m.acos(z/r) #phi
    return [r, polar, azimuthal]


""" gets the arclength between two points with a constant radius r. """
def get_arc_len(pt_a, pt_b):
    a = cart_to_spherical(pt_a)
    b = cart_to_spherical(pt_b)

    phi = a[1] * 180 / m.pi
    theta = a[2] * 180 / m.pi
    #print("\nphi = " + str(phi))
    #print("theta = " + str(theta) + "\n")
    #lattitudes and longitudes are in degrees right here
    latitude_a = (a[1] * 180 / m.pi) - 90
    latitude_b = (b[1] * 180 / m.pi) - 90
    longitude_a = a[2]* 180 / m.pi
    longitude_b = b[2]* 180 / m.pi
    return haversine(latitude_a, longitude_a, latitude_b, longitude_b)


""" uses haversine formula to get the distance between 2 points on a circle """
#https://www.geeksforgeeks.org/haversine-formula-to-find-distance-between-two-points-on-a-sphere/
def haversine(lat1, lon1, lat2, lon2):
    # distance between latitudes
    # and longitudes
    dLat = (lat2 - lat1) * m.pi / 180.0
    dLon = (lon2 - lon1) * m.pi / 180.0
    # convert to radians
    lat1 = (lat1) * m.pi / 180.0
    lat2 = (lat2) * m.pi / 180.0
    # apply formulae
    a = (pow(m.sin(dLat / 2), 2) +
         pow(m.sin(dLon / 2), 2) *
             m.cos(lat1) * m.cos(lat2))
    rad = 1 
    c = 2 * m.asin(m.sqrt(a))
    return rad * c

""" gets approximate final front location for both fronts. cannot get exact due to the front locations resetting to the intiation site """
def get_final_front_locations(simulation_folder, simulation):
    front_locations = get_all_front_locations(simulation_folder, simulation)
    front_0 = np.empty(0)
    front_1 = np.empty(0)

    front_0 = np.array(front_locations[0][2])
    front_1 = np.array(front_locations[0][3])
    for location in front_locations:
            front_0 = np.vstack([front_0, location[2]])
            front_1 = np.vstack([front_1, location[3]])
    
    #getting the unique front locations in the order that they are propogated
    indexes = np.unique(front_0, axis=0, return_index=True)[1]
    unique_front_0 = np.array([front_0[index] for index in sorted(indexes)])
    indexes = np.unique(front_1, axis=0, return_index=True)[1]
    unique_front_1 = np.array([front_1[index] for index in sorted(indexes)])


    #removing any huge jumps to different parts of the skull that are not supposed to be there
    temp = unique_front_0[0]
    i = 0
    for loc in unique_front_0:
        x = get_euclidean_distance(temp, loc)
        #print(x)
        if(x > 6):
            unique_front_0 = np.delete(unique_front_0, i, axis=0)
            continue
        temp = loc
        i+=1
    
    temp = unique_front_1[0]
    i = 0
    for loc in unique_front_1:
        x = get_euclidean_distance(temp, loc)
        #print(x)
        if(x > 6):
            unique_front_1 = np.delete(unique_front_1, i, axis=0)
            continue
        temp = loc
        i+=1


    # if(len(unique_front_0) > 1):
    #     unique_front_0 = np.delete(unique_front_0, -1, axis=0)
    # if(len(unique_front_1) > 1):
    #     unique_front_1 = np.delete(unique_front_1, -1, axis=0)
    #print(unique_front_0)
    return unique_front_0[-1], unique_front_1[-1]




# """ gets the ratio of the distance between the initiation site and the end points of the fronts, over the
# distances of all of the front locations combined, assuming that the skull is a perfect sphere with radius of 1 so that
# we can hopefully capture the non-linearity of the crack. linearity of 1 = perfectly straight crack. """
# def get_linearity(simulation_folder, simulation):
#     #print(simulation)
#     d0, d1 = get_distances_from_initiation(simulation_folder, simulation)
#     init_cite = get_initiation_cite(simulation_folder, simulation)
#     front_0_endpoint, front_1_endpoint = get_final_front_locations(simulation_folder, simulation)
#     True_len0 = get_arc_len(init_cite, front_0_endpoint)
#     True_len1 = get_arc_len(init_cite, front_1_endpoint)

#     linearity = (True_len0 + True_len1) / (d0 + d1)
#     return linearity

""" gets the ratio of the distance between the initiation site and the end points of the fronts, over the
distances of all of the front locations combined, assuming that the skull is a perfect sphere with radius of 1 so that
we can hopefully capture the non-linearity of the crack. linearity of 1 = perfectly straight crack. """
def get_linearity(simulation_folder, simulation):
    #print(simulation)
    d0, d1,front_0_endpoint,front_1_endpoint = get_distances_from_initiation(simulation_folder, simulation)
    init_cite = get_initiation_cite(simulation_folder, simulation)
    #front_0_endpoint, front_1_endpoint = get_final_front_locations(simulation_folder, simulation)
    True_len0 = get_arc_len(init_cite, front_0_endpoint)
    True_len1 = get_arc_len(init_cite, front_1_endpoint)

    linearity = (True_len0 + True_len1) / (d0 + d1)
    #print(linearity)
    return linearity

# get_linearity('F:\\Jake\\good_simies\\', 'Para_1-5ft_PHI_0_THETA_0')
# x = pd.read_csv("C:\\Users\\u1056\\sfx\\ML\\Feature_gathering\\OG_dataframe.csv")
# print(x)
# print(x.iloc[35])
# print(x.iloc[18])
# print(x.iloc[14])
# print(x.iloc[11])
# print(x.iloc[12])
