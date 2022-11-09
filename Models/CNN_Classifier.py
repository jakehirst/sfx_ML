#code similar to https://www.kaggle.com/code/gcdatkin/concrete-crack-image-detection/notebook

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report

# positive_dir = Path("C:\\Users\\u1056\\sfx\\ML\\Binary_phi_CNN\\phi_GT_15") #phi
positive_dir = Path("C:\\Users\\u1056\\sfx\\ML\\Binary_phi_CNN\\Height_GT_2.5") #height

# negative_dir = Path("C:\\Users\\u1056\\sfx\\ML\\Binary_phi_CNN\\phi_not_GT_15") #phi
negative_dir = Path("C:\\Users\\u1056\\sfx\\ML\\Binary_phi_CNN\\Height_notGT_2.5") #height




#creating dataframes
#The filepaths to the images as the only parameter (first column), and then the binary Label (height, phi, or theta) being the other column 
def generate_df(image_dir, label):
    #.glob finds certain patterns in the folder
    filepaths = pd.Series(list(image_dir.glob(r'*.png')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=filepaths.index)
    df = pd.concat([filepaths, labels], axis=1)
    return df


positive_df = generate_df(positive_dir, label="POSITIVE")
negative_df = generate_df(negative_dir, label="NEGATIVE")
all_df = pd.concat([positive_df, negative_df], axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)

#splitting the all_df into training and testing dfs, where the training df is 0.7 (70%) the size of the all_df
train_df, test_df = train_test_split(
    all_df,
    train_size=0.8, #fraction of the training size
    shuffle=True, #randomizing what goes in what
    random_state=1
)

''' use with greyscale images'''
# train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
#     #might not want rescale here
#     rescale=1./255, #rescales all of the pixel values in the image to be between 1 and 0
#     validation_split=0.2
# )

# test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1./255
# )

'''can use for binary images'''
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.2
)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
)


train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(642, 642), #resizes the data, we dont want that... already have square images. so I just put the size that the image is already in there
    color_mode='grayscale', #could be rgb, but we will use greyscale images here
    class_mode='binary', #what type of labels do you have? could be "categorical" but then youd need a y_col, which would be the categorical labels
    batch_size=2, #This represents how many images are passed into the CNN at once... almost like a parallel 
    #programming thing. Training time will reduce with the larger the amount of images you pass in at once.
    shuffle=True,
    seed=42,#TODO: what is this??
    subset='training'
)

val_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(642, 642),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=2,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_data = train_gen.flow_from_dataframe(
    test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(642, 642),
    color_mode='grayscale',
    class_mode='binary',
    batch_size=2,
    shuffle=False, #dont want to shuffle test data ever
    seed=42
)


"""     ......START TRAINING......      """
inputs = tf.keras.Input(shape=(642, 642, 1)) #shape = (pixelHeight, pixelWidth, number_of_color_inputs) number_of_color_inputs = 3 for rgb, 1 for greyscale

"""layers look at the images and try to take out some sort of parameter to later classify."""

"""   feature extraction   """
#filters = how many times a layer looks at the image
#kernel_size = size of the kernel that is moving across the image. (3x3 is default)
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs) #first layer (gets easy features such as edges or lines in the image)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
#in the second layer, we increase the filter size to 32, in order to get more complex features. (e.g. kinks in the cracks)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x) #second layer
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
# x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x) #third layer
# x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
# x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x) #fourth layer
# x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
"""   feature extraction   """

"""   classification aspect   """
x = tf.keras.layers.GlobalAveragePooling2D()(x) #reducing the output into one dimension 
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
"""   classification aspect   """

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

#keep trying to fit the model based on 
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
)


fig = px.line(
    history.history,
    y=['loss', 'val_loss'],
    labels={'index': "Epoch", 'value': "Loss"},
    title="Training and Validation Loss Over Time"
)

fig.show()



def evaluate_model(model, test_data):
    
    results = model.evaluate(test_data, verbose=0)
    loss = results[0]
    acc = results[1]
    
    print("    Test Loss: {:.5f}".format(loss))
    print("Test Accuracy: {:.2f}%".format(acc * 100))
    
    y_pred = np.squeeze((model.predict(test_data) >= 0.5).astype(np.int))
    cm = confusion_matrix(test_data.labels, y_pred)
    clr = classification_report(test_data.labels, y_pred, target_names=["NEGATIVE", "POSITIVE"])
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
    plt.xticks(ticks=np.arange(2) + 0.5, labels=["NEGATIVE", "POSITIVE"])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=["NEGATIVE", "POSITIVE"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    print("Classification Report:\n----------------------\n", clr)

print("done")
