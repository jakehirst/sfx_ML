""" This is what we use to populate our test matrix and generate new data. """
from pyDOE import lhs
import numpy as np

# Number of samples
num_samples = 1000

# Variable ranges
height_range = (1.0, 4.0)
phi_range = (5, 50)
theta_range = (0, 360)

# Generate Latin Hypercube samples
variable_ranges = [height_range, phi_range, theta_range]
lhs_samples = lhs(len(variable_ranges), samples=num_samples)

# Scale and convert LHS samples to variable types
scaled_samples = []
for i, (lower, upper) in enumerate(variable_ranges):
    if i == 0:
        scaled_samples.append(np.around(lhs_samples[:, i] * (upper - lower) + lower, decimals=4))
    else:
        scaled_samples.append(lhs_samples[:, i] * (upper - lower) + lower)
        scaled_samples[i] = scaled_samples[i].astype(int)

# Transpose the scaled samples
lhs_samples_transposed = list(map(list, zip(*scaled_samples)))

# Print the LHS samples
for sample in lhs_samples_transposed:
    print(str(sample).replace("[","").replace("]",""))