import numpy as np
import pandas as pd

height_bounds = (1.0, 5.0)
phi_bounds = (0.0, 50.0)
theta_bounds = (0.0, 360.0)


def latin_hypercube_samples(n, ranges):
    num_vars = len(ranges)
    samples = np.zeros((n, num_vars))

    for i in range(num_vars):
        samples[:, i] = np.random.uniform(ranges[i][0], ranges[i][1], n)

    for i in range(num_vars):
        np.random.shuffle(samples[:, i])

    return samples

# Define the number of samples and variable ranges
num_samples = 1000
variable_ranges = [(height_bounds[0], height_bounds[1]), (phi_bounds[0], phi_bounds[1]), (theta_bounds[0], theta_bounds[1])]  # (min, max) for each variable

# Generate Latin Hypercube Samples
samples = latin_hypercube_samples(num_samples, variable_ranges)

# Assign the samples to height, phi, and theta
height, phi, theta = samples[:, 0], samples[:, 1], samples[:, 2]

# Print the generated samples
print(f'Generated Height values: {height}')
print(f'Generated Phi values: {phi}')
print(f'Generated Theta values: {theta}')

dataframe = pd.DataFrame(data=samples, columns = ['height', 'phi', 'theta'])
print('here')