import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append("C:\\Users\\u1056\\sfx\\ML")
sys.path.append("C:\\Users\\u1056\\sfx\\ML\\Feature_gathering")
from Feature_gathering.features_to_df import create_df
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import keras
from keras import layers
import math as m
import quaternionic

def spherical_to_cartesian(theta, phi):
    x = m.cos(phi) * m.sin(theta)
    y = m.sin(phi) * m.sin(theta)
    z = m.cos(theta)
    return [x, y, z]

# linear regression for multioutput regression
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
# create datasets
#X, y = make_regression(n_samples=24, n_features=12, n_informative=5, n_targets=2, random_state=1)

df = create_df()
df = df.drop("height", axis=1)
train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

train_y = []
train_y.append(train_dataset.pop("phi"))
train_y.append(train_dataset.pop("theta"))
train_y = np.array(train_y).T
train_x = np.array(train_dataset)

test_y = []
test_y.append(test_dataset.pop("phi"))
test_y.append(test_dataset.pop("theta"))
test_y = np.array(test_y).T
test_x = np.array(test_dataset)

# define model
model = LinearRegression()
# fit model
model.fit(train_x, train_y)

for i in range(len(test_x)):
    # make a prediction
    data_in = test_x[i]
    prediction = model.predict([data_in])
    # summarize prediction
    print("\nlabels = " + str(test_y[i]))
    print("prediction = " + str(prediction[0]))