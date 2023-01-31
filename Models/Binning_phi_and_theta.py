import pandas as pd
import numpy as np
import math as m
from keras.utils import to_categorical
from PIL import Image
from numpy import asarray


def Bin_phi_and_theta(df, num_bins):
    theta_spacing = 361 / num_bins
    phi_spacing = 60 / num_bins
    num = 0
    bin = np.zeros(len(df))
    phis = np.array(df["phi"])
    thetas = np.array(df["theta"])
    y_col_values = []

    for j in range(num_bins*num_bins):
        y_col_values.append(str(int(j)))

    for phi_bin in range(num_bins):
        for theta_bin in range(num_bins):
            for i in range(len(df)):

                phi_low = phi_bin * phi_spacing
                phi_high = (phi_bin + 1) * phi_spacing
                theta_low = theta_bin * theta_spacing
                theta_high = (theta_bin + 1) * theta_spacing

                if(phis[i] >= phi_low and phis[i] < phi_high and thetas[i] >= theta_low and thetas[i] < theta_high):
                    if(bin[i] > 0): 
                        print("This index was catergorized twice")
                        break
                    bin[i] = str(int(num))
                
            num += 1
    
    bin = to_categorical(bin)
    bin_df = pd.DataFrame(bin)
    
    bin_df.columns = y_col_values
    df = pd.concat([df, bin_df], axis=1)

    df = df.drop("phi", axis=1)
    df = df.drop("theta", axis=1)

    return df, y_col_values


def turn_filepath_to_nparray(x):
    images = np.empty(0)
    for i in range(len(x)):
        path = x[i]
        img = Image.open(path)
        nump = asarray(img, dtype="float32")/255.0
        x[i] = nump
    return x