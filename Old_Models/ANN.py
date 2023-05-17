import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append("C:\\Users\\u1056\\sfx\\ML")
sys.path.append("C:\\Users\\u1056\\sfx\\ML\\Feature_gathering")
#from Feature_gathering.features_to_df import create_df
import pandas as pd
# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
# import keras
# from keras import layers
import math as m
#import tensorflow_probability as tfp

def prepare_data(df_filename, label_to_predict, epochs, features_to_drop=None):
    print("----- PREDICTING " + label_to_predict + " -----")
    np.set_printoptions(precision=3, suppress=True) #makes numpy easier to read with prints

    df = pd.read_csv("C:\\Users\\u1056\\sfx\\ML\\Feature_gathering\\" + df_filename, index_col = [0])

    """ drops all of the labesl that are not the one we are trying to predict """
    labels = ["height", "phi", "theta", "x", "y", "z"]
    for label in labels:
        if((not label == label_to_predict) and df.columns.__contains__(label)):
            df = df.drop(label, axis=1)

    """ drop whatever you are not predicting or predicting with here"""
    if(not features_to_drop == None):
        for feature in features_to_drop:
            df = df.drop(feature, axis=1)
        print(df)

    """ sampling the dataset randomly """
    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)
    columns = df.columns
    #sns.pairplot(train_dataset[columns], diag_kind='kde') #doesnt work...
    #print(train_dataset.describe().transpose())
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop(label_to_predict)
    test_labels = test_features.pop(label_to_predict)
    #print(train_dataset.describe().transpose()[['mean', 'std']])
    return [train_features, train_labels, test_features, test_labels, epochs]


def prepare_data_Kfold(folder, dataset, label_to_predict, epochs, numfolds=5, features_to_drop=None):
    df = pd.read_csv(folder+dataset, index_col = [0])

    """ drops all of the labesl that are not the one we are trying to predict """
    labels = ["height", "phi", "theta", "x", "y", "z"]
    for label in labels:
        if((not label == label_to_predict) and df.columns.__contains__(label)):
            df = df.drop(label, axis=1)
    
    """ drop whatever you are not predicting or predicting with here"""
    if(not features_to_drop == None):
        for feature in features_to_drop:
            df = df.drop(feature, axis=1)
        print(df)
    
    """ shuffling the dataset randomly """
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)

    """ creating numfolds number of folds in the dataset """
    examples_per_fold = m.ceil(len(df) / numfolds)
    folds = []
    for i in range(numfolds):
        if(i == numfolds-1):
            fold = df.iloc[(i*examples_per_fold):]
            folds.append(fold)
        else:
            fold = df.iloc[(i*examples_per_fold):((i+1) * examples_per_fold)]
            folds.append(fold)

    """ putting all fold combinations into a list to run the machin e learning model with """
    args = []
    for fold in folds:
        test_dataset = fold
        train_dataset = df.drop(fold.index)

        train_features = train_dataset.copy()
        test_features = test_dataset.copy()

        train_labels = train_features.pop(label_to_predict)
        test_labels = test_features.pop(label_to_predict)

        args.append([train_features, train_labels, test_features, test_labels, epochs])

    return args


def run_Kfold_ANN(args, Full_df, activation='relu', show=False):
    results = []
    for arg in args:
        results.append(make_ANN(*arg))
    
    train_r2 = np.empty(0)
    test_r2 = np.empty(0)
    min_loss = np.empty(0)
    min_val_loss = np.empty(0)
    loss = np.empty(0)
    val_loss = np.empty(0)
    models = []

    for result in results:
        train_r2 = np.append(train_r2, result["Training R^2"])
        test_r2 = np.append(test_r2, result["Test R^2"])
        min_loss = np.append(min_loss, min(result["history"].history['loss']))
        min_val_loss = np.append(min_val_loss, min(result["history"].history['val_loss']))
        models.append(result["model"])
        if(show == True):
            if(len(loss) == 0):
                loss = np.array([result["history"].history['loss']])
                val_loss = np.array([result["history"].history['val_loss']])
            else:
                loss = np.vstack([loss, result["history"].history['loss']])
                val_loss = np.vstack([val_loss, result["history"].history['val_loss']])

        # loss = np.append(loss, [result["history"].history['loss']])
        # val_loss = np.append(val_loss, [result["history"].history['val_loss']])
    


    avg_train_r2 = np.sum(train_r2) / len(train_r2)
    avg_test_r2 = np.sum(test_r2) / len(test_r2)
    avg_min_loss = np.sum(min_loss) / len(min_loss)
    avg_min_val_loss = np.sum(min_val_loss) /  len(min_val_loss)
    
    print("avg_train_r2 = " + str(avg_train_r2))
    print("avg_test_r2 = " + str(avg_test_r2))
    print("avg_min_loss = " + str(avg_min_loss))
    print("avg_min_val_loss = " + str(avg_min_val_loss))


    #TODO: Not sure about the axis here
    if(show == True):
        avg_loss = np.sum(loss, axis=0) / len(loss)
        avg_val_loss = np.sum(val_loss, axis=0) / len(val_loss)


        plt.plot(avg_loss, label='avg loss (mean absolute error)')
        plt.plot(avg_val_loss, label='avg validation loss')
        #plt.ylim([0, 4])
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)
        plt.show()
        print("done")


def make_ANN(train_features, train_labels, test_features, test_labels, epochs, activation='relu', show=False):
    #quote from tensorflow:
    """One reason this is important is because the features are multiplied by the model weights. So, the scale of the outputs and the scale of the gradients are affected by the scale of the inputs.
    Although a model might converge without feature normalization, normalization makes training much more stable"""
    """ normalizing features and labels """

    normalizer = tf.keras.layers.Normalization(axis=-1) #creating normalization layer
    normalizer.adapt(np.array(train_features)) #fitting the state of the preprocessing layer
    #print(normalizer.mean.numpy())

    """ Just an example of how it is normalizing """
    first = np.array(train_features[:1])
    with np.printoptions(precision=3, suppress=True):
        # print('First example:', first)
        print()
        # print('Normalized:', normalizer(first).numpy())

    numfeatures = len(train_features.columns)

    # model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=numfeatures),
    #                              tf.keras.layers.Dense(16, activation = activation),
    #                              tf.keras.layers.Dense(16, activation = activation),
    #                              tf.keras.layers.Dense(32, activation = activation),
    #                              tf.keras.layers.Dense(32, activation = activation),
    #                              tf.keras.layers.Dense(32, activation = activation),
    #                              tf.keras.layers.Dense(1)])
    
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=numfeatures),
                                 tf.keras.layers.Dense(128, activation = activation),
                                 tf.keras.layers.Dense(64, activation = activation),
                                 tf.keras.layers.Dense(32, activation = activation),
                                 tf.keras.layers.Dense(16, activation = activation),
                                 tf.keras.layers.Dense(8, activation = activation),
                                 tf.keras.layers.Dense(1)])
    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        # loss='mean_absolute_error')
        # loss=mean_distance_error_phi_theta)
        # loss='mean_absolute_error')
        loss='mean_squared_error')
        # loss='mean_squared_logarithmic_error')
    
    history = model.fit(tf.expand_dims(train_features, axis=-1), 
                                       train_labels, 
                                       epochs=epochs,
                                       validation_split = 0.2,  
                                       callbacks=[
                                            tf.keras.callbacks.EarlyStopping(
                                                monitor='val_loss',
                                                patience=500,
                                                restore_best_weights=True
                                            )
                                        ]
                                       )

    print("minimum MAE: ")
    print(min(history.history['loss']))
    print("minimum validation MAE: ")
    print(min(history.history['val_loss']))
    
    
    plt.plot(history.history['loss'], label='loss (mean absolute error)')
    plt.plot(history.history['val_loss'], label='val_loss')
    #plt.ylim([0, 4])
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title("theta")
    plt.legend()
    plt.grid(True)
    plt.show()

    """ makes predictions with the test dataset and plots them. Good predictions should lie on the line. """
    test_predictions = model.predict(test_features).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.title("theta")
    lims = [0, max(test_labels)]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()


    """ gets r^2 value of the test dataset with the predictions made from above ^ """
    metric = tfa.metrics.r_square.RSquare()
    metric.update_state(test_labels, test_predictions)
    test_result = metric.result()
    print("Test R^2 = " + str(test_result.numpy()))


    """ gets r^2 value of the training dataset """
    training_predictions = model.predict(train_features).flatten()
    metric = tfa.metrics.r_square.RSquare()
    metric.update_state(train_labels, training_predictions)
    train_result = metric.result()
    print("Train R^2 = " + str(train_result.numpy()))
    return {"Training R^2": train_result.numpy(), "Test R^2": test_result.numpy(), "history": history, "model": model}


def Prepare_Full_Df(folder, dataset, label_to_predict):
    df = pd.read_csv(folder + dataset, index_col = [0])
    """ drops all of the labesl that are not the one we are trying to predict """
    labels = ["height", "phi", "theta", "x", "y", "z"]
    for label in labels:
        if((not label == label_to_predict) and df.columns.__contains__(label)):
            df = df.drop(label, axis=1)
    return df


def remove_features(df, features_to_remove=[], features_to_keep=[]):
    labels = ["theta", "phi", "height", "x", "y", "z"]
    if(len(features_to_remove) == 0 and len(features_to_keep)>0):
        print("remove all features except listed")
        new_df = df.copy()
        for feature in df.columns:
            if(features_to_keep.__contains__(feature) or labels.__contains__(feature)):
                continue
            else:
                new_df = new_df.drop(feature, axis=1)
        
    elif(len(features_to_remove) > 0 and len(features_to_keep)==0):
        print("remove listed features")
        new_df = df.copy()
        for feature in features_to_remove:
            new_df = new_df.drop(feature, axis=1)
        
    else:
        print("keep all features")
        new_df = df
        
    return new_df
    
    
    


"""feature options = "front 0 x", "front 0 y", "front 0 z", "front 1 x", "front 1 y", 
                    "front 1 z", "init x", "init y", "init z", "dist btw frts", "crack len", 
                    "linearity", "max thickness", "mean thickness"  """
          
          
"""   ********* theta **********   """
# folder = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/"
# dataset = "OG_dataframe.csv"
# Full_df = Prepare_Full_Df(folder, dataset, "theta")          
# theta_p_less_point5 = ["front 0 y", "front 0 z", "init y", "init z", "dist btw frts"]

# simple_df = remove_features(Full_df, features_to_keep=theta_p_less_point5)
# args = prepare_data_Kfold(folder, dataset, "theta", epochs=5000)
# run_Kfold_ANN(args, simple_df, activation="relu")




# """   ********* phi **********   """
# folder = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/"
# dataset = "OG_dataframe.csv"
# Full_df = Prepare_Full_Df(folder, dataset, "phi")    
# phi_p_less_point5 = ["front 0 x", "front 0 z", "front 1 z"]

# simple_df = remove_features(Full_df, features_to_keep=phi_p_less_point5)
# args = prepare_data_Kfold(folder, dataset, "phi", epochs=5000)
# run_Kfold_ANN(args, simple_df, activation="relu")




"""   ********* height **********   """
folder = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/"
dataset = "OG_dataframe.csv"
Full_df = Prepare_Full_Df(folder, dataset, "height")    
height_p_less_point05 = ["init y", "crack len", "max thickness",  "abs_val_mean_kink"]
height_p_closeto_point05 = ["init y", "dist btw fts", "crack len", "max thickness", "max_kink", "abs_val_mean_kink", "abs_val_sum_kink"]


simple_df = remove_features(Full_df, features_to_keep=height_p_less_point05)
simple_df = remove_features(Full_df, features_to_remove=[])
args = prepare_data_Kfold(folder, dataset, "height", epochs=5000)
run_Kfold_ANN(args, simple_df, activation="relu")

