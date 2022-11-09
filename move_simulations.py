import os
import sys

simulation_folder_filepath = "C:\\Users\\u1056\\sfx\\simulation_results\\"
simulations_to_move = ["Para_3ft_PHI_30_THETA_230", "Para_1-5ft_PHI_40_THETA_240"]
#destination_filepath = "C:\\Users\\u1056\\sfx\\bad_simulation_results\\"
destination_filepath = "C:\\Users\\u1056\\sfx\\2_initiation_site_simulations\\"

for sim in simulations_to_move:
    try:
        os.rename(simulation_folder_filepath + sim, destination_filepath + sim)
    except:
        print("couldnt move " + sim)