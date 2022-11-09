from front_locations import *
import math as m

""" gets the total crack length based on euclidean distance between each front location """
def get_distances_from_initiation(simulation_folder, simulation):
    prev_front_0 = get_initiation_cite(simulation_folder, simulation)
    prev_front_1 = get_initiation_cite(simulation_folder, simulation)
    front_locations = get_all_front_locations(simulation_folder, simulation)
    dist_frt_0 = 0.0
    dist_frt_1 = 0.0

    for locations in front_locations:
        dist_frt_0 += get_arc_len(prev_front_0, locations[2])
        dist_frt_1 += get_arc_len(prev_front_1, locations[3])
        prev_front_0 = locations[2]
        prev_front_1 = locations[3]

    return dist_frt_0, dist_frt_1

""" Turns cartesian coordinates [x,y,z] into spherical coordinates [r, phi, theta] """
#https://vvvv.org/blog/polar-spherical-and-geographic-coordinates#:~:text=The%20definition%20of%20the%20spherical%20coordinates%20has%20two,longitude%20does%20not%20match%20with%20the%20two%20angles.
def cart_to_spherical(cartesian):
    x = cartesian[0]; y = cartesian[1]; z = cartesian[2]
    azimuthal = m.atan(y/x)
    r = m.sqrt(x**2 + y**2 + z**2)
    polar = m.acos(z/r)
    return [r, polar, azimuthal]

""" gets the arclength between two points with a constant radius r. """
def get_arc_len(pt_a, pt_b):
    a = cart_to_spherical(pt_a)
    b = cart_to_spherical(pt_b)

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

""" gets the ratio of the distance between the initiation site and the end points of the fronts, over the
distances of all of the front locations combined, assuming that the skull is a perfect sphere with radius of 1 so that
we can hopefully capture the non-linearity of the crack. linearity of 1 = perfectly straight crack. """
def get_linearity(simulation_folder, simulation):
    d0, d1 = get_distances_from_initiation(simulation_folder, simulation)
    init_cite = get_initiation_cite(simulation_folder, simulation)
    front_locations = get_all_front_locations(simulation_folder, simulation)
    front_0_endpoint = front_locations[-1][2]
    front_1_endpoint = front_locations[-1][3]
    True_len0 = get_arc_len(init_cite, front_0_endpoint)
    True_len1 = get_arc_len(init_cite, front_1_endpoint)

    linearity = (True_len0 + True_len1) / (d0 + d1)
    return linearity

# simulation_folder = 'C:\\Users\\u1056\\sfx\\good_simies\\'
# simulation = 'Para_1-5ft_PHI_40_THETA_20'
# get_linearity(simulation_folder, simulation)
