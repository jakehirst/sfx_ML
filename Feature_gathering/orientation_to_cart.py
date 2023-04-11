import pandas as pd
import numpy as np
import math as m
import os

filename = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/OG_dataframe.csv"
df = pd.read_csv(filename)

phis = df["phi"]
thetas = df["theta"]
print("done")

