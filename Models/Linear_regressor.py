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
import keras
from keras import layers


np.set_printoptions(precision=3, suppress=True) #makes numpy easier to read with prints

df = create_df()

""" drop whatever you are not predicting or predicting with here"""
#df = df.drop("phi", axis=1)
df = df.drop("theta", axis=1)
df = df.drop("height", axis=1)
# df = df.drop("front 0 x", axis=1)
# df = df.drop("front 0 y", axis=1)
# df = df.drop("front 0 z", axis=1)
# df = df.drop("front 1 x", axis=1)
# df = df.drop("front 1 y", axis=1)
# df = df.drop("front 1 z", axis=1)
# df = df.drop("init x", axis=1)
# df = df.drop("init y", axis=1)
#df = df.drop("init z", axis=1)
df = df.drop("dist btw frts", axis=1)
df = df.drop("linearity", axis=1)

""" sampling the dataset randomly """
train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)
columns = df.columns
#sns.pairplot(train_dataset[columns], diag_kind='kde') #doesnt work...
print(train_dataset.describe().transpose())
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('phi')
test_labels = test_features.pop('phi')
print(train_dataset.describe().transpose()[['mean', 'std']])

#quote from tensorflow:
"""One reason this is important is because the features are multiplied by the model weights. So, the scale of the outputs and the scale of the gradients are affected by the scale of the inputs.
Although a model might converge without feature normalization, normalization makes training much more stable"""

""" normalizing features and labels """
normalizer = tf.keras.layers.Normalization(axis=-1) #creating normalization layer
normalizer.adapt(np.array(train_features)) #fitting the state of the preprocessing layer
print(normalizer.mean.numpy())

""" Just an example of how it is normalizing """
first = np.array(train_features[:1])
with np.printoptions(precision=3, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())



linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.predict(train_features[:10])

print(linear_model.layers[1].kernel)

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


history = linear_model.fit(
    train_features,
    train_labels,
    epochs=500,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

print("minimum MAE: ")
print(min(history.history['loss']))
print("minimum validation MAE: ")
print(min(history.history['val_loss']))

plt.plot(history.history['loss'], label='loss (mean absolute error)')
plt.plot(history.history['val_loss'], label='val_loss')
#plt.ylim([0, 4])
plt.xlabel('Epoch')
plt.ylabel('Error [height]')
plt.legend()
plt.grid(True)
plt.show()

print(df)