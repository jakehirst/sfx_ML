import tensorflow as tf
print(tf.version.VERSION)
from tensorflow import keras
import pandas as pd
from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
import tensorflow as tf #COMMENT can only get tensorflow version 2.9 on mac m1, so that caps the tensorflow_probability version we can use
import tensorflow_addons as tfa
import tensorflow_probability as tfp #COMMMENT can only use version 0.17.0 with tensorflow 2.9
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import KFold
from CNN_function_library import *

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tf.keras.backend.set_floatx("float64")
tfd = tfp.distributions
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

TOTAL_SAMPLES = 1 #TODO: make sure you update this later

scaler = StandardScaler()
#detector = IsolationForest(n_estimators=1000, behaviour="deprecated", contamination="auto", random_state=0)
neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / TOTAL_SAMPLES


def get_kl_regularizer(prior_distribution):
    return tfpl.KLDivergenceRegularizer(prior_distribution,
                                        weight=1.0,
                                        use_exact_kl=False,
                                        test_points_fn=lambda q: q.sample(3),
                                        test_points_reduce_axis=(0,1))        
 

#COMMENT https://towardsdatascience.com/uncertainty-in-deep-learning-bayesian-cnn-tensorflow-probability-758d7482bef6

#https://towardsdatascience.com/uncertainty-in-deep-learning-bayesian-cnn-tensorflow-probability-758d7482bef6
def custom_normal_prior(dtype, shape, name, trainable, add_variable_fn, prior_mean, prior_std):
    distribution = tfd.Normal(loc = prior_mean * tf.ones(shape, dtype),
                              scale = prior_std * tf.ones(shape, dtype))
    batch_ndims = tf.size(distribution.batch_shape_tensor())
    
    distribution = tfd.Independent(distribution,
                                   reinterpreted_batch_ndims = batch_ndims)
    return distribution


def conv_reparameterization_layer(filters, kernel_size, activation):
    # For simplicity, we use default prior and posterior.
    # In the next parts, we will use custom mixture prior and posteriors.
    return tfpl.Convolution2DReparameterization(
            filters = filters, #not entirely sure what these filters are in a convolutional layer
            kernel_size = kernel_size,
            activation = activation, 
            kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular=False), #if this is set to true, we will get a point estimate instead of a probability
            kernel_prior_fn = tfpl.default_multivariate_normal_fn, #could change these to the custom_normal_prior
            
            bias_prior_fn = tfpl.default_multivariate_normal_fn,
            bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular=False),
            
            kernel_divergence_fn = divergence_fn,
            bias_divergence_fn = divergence_fn)
    

def approximate_kl(q, p, q_tensor):
    return tf.reduce_mean(q.log_prob(q_tensor) - p.log_prob(q_tensor))

def make_CNN(args, label_to_predict, batch_size=5, patience=25, max_epochs=1000, optimizer="adam", activation="relu", kernel_size=(3,3), plot = True, augmentation_list = [], num_folds=5):
    results = []
    #TODO: do this for phi and theta later
    #shuffles the dataset and puts it into a dataframe
    if(label_to_predict == "height"):
        df = pd.concat([args[0], args[1]], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
        df.columns = ["Filepath", "height"]
        kfold = KFold(n_splits=num_folds, shuffle=True)
        inputs = np.array(df["Filepath"])
        outputs = np.array(df["height"])
    elif(label_to_predict == "phi"):
        df = pd.concat([args[0], args[2]], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
        df.columns = ["Filepath", "phi"]
        kfold = KFold(n_splits=num_folds, shuffle=True)
        inputs = np.array(df["Filepath"])
        outputs = np.array(df["phi"])
        prior_mean = 20
        prior_std = 2
    elif(label_to_predict == "theta"):
        df = pd.concat([args[0], args[3]], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
        df.columns = ["Filepath", "theta"]
        kfold = KFold(n_splits=num_folds, shuffle=True)
        inputs = np.array(df["Filepath"])
        outputs = np.array(df["theta"])
    elif(label_to_predict == "x"):
        df = pd.concat([args[0], args[2], args[3]], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
        df.columns = ["Filepath", "phi", "theta"]
        print("convert phi and theta to cartesian x here")
    elif(label_to_predict == "y"):
        df = pd.concat([args[0], args[2], args[3]], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
        df.columns = ["Filepath", "phi", "theta"]
        print("convert phi and theta to cartesian y here")



    fold_no = 1
    for train, test in kfold.split(inputs, outputs):
        train_df = df.iloc[train]
        test_df = df.iloc[test]


        
        #train_df, test_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=1)

        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255, #all pixel values set between 0 and 1 instead of 0 and 255.
            validation_split=0.2 #creating a validation split in our training generator
        )

        val_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255,
        )

        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale = 1./255, #all pixel values set between 0 and 1 instead of 0 and 255.
        )

        #creating a validation set
        val_df = train_df.sample(frac=0.2)
        #removing the validation set from the training set
        train_df = train_df.drop(val_df.index)
        #adding data augmentation to training dataset
        train_df = add_augmentations(train_df, augmentation_list)


        #flow the images through the generators
        train_images = train_generator.flow_from_dataframe(
            dataframe=train_df,
            x_col = 'Filepath',
            y_col = label_to_predict,
            target_size=args[4][:2],  #can reduce the images to a certain size to reduce training time. 120x120 for example here
            color_mode='rgb',
            class_mode = 'raw', #keeps the classes of our labels the same after flowing
            batch_size=batch_size, #can increase this to up to like 10 or so for how much data we have
            shuffle=True,
            seed=42,
        )

        val_images = val_generator.flow_from_dataframe(
            dataframe=val_df,
            x_col = 'Filepath',
            y_col = label_to_predict,
            target_size=args[4][:2], #can reduce the images to a certain size to reduce training time. 120x120 for example here
            color_mode='rgb',
            class_mode = 'raw', #keeps the classes of our labels the same after flowing
            batch_size=batch_size, #can increase this to up to like 10 or so for how much data we have
            shuffle=True,
            seed=42,
        )

        test_images = test_generator.flow_from_dataframe(
            dataframe=test_df,
            x_col = 'Filepath',
            y_col = label_to_predict,
            target_size=args[4][:2],  #can reduce the images to a certain size to reduce training time. 120x120 for example here
            color_mode='rgb',
            class_mode = 'raw', #keeps the classes of our labels the same after flowing
            batch_size=batch_size, #can increase this to up to like 10 or so for how much data we have
            shuffle=False
        )

        len_outputs = 1 #TODO: need to figure out what this actually is... its a MultivariateNormalTriL parameter
        n_batches = 5 #TODO: need to figure out what this actually is too
        
        prior = tfd.Independent(tfd.Normal(loc=prior_mean * tf.ones(shape=len_outputs, dtype=tf.float64),
                                scale = prior_std * tf.ones(shape=len_outputs, dtype=tf.float64), ), reinterpreted_batch_ndims=1)
        kl_regularizer = get_kl_regularizer(prior)


        def nll(y_true, y_pred):
            return -y_pred.log_prob(y_true)

        def my_dist(params):
            return tfd.Normal(loc=params, scale=1)
        
        #https://datascience.stackexchange.com/questions/106600/how-to-perform-regression-on-image-data-using-tensorflow

        # conv = tf.keras.Sequential([
        #     tf.keras.layers.InputLayer((642, 802, 3)),
            
        #     tf.keras.layers.Conv2D(filters = 16, kernel_size=3, activation='relu'),
        #     tf.keras.layers.MaxPooling2D(2),
            
        #     tf.keras.layers.Conv2D(filters = 32, kernel_size=3, activation='relu'),
        #     tf.keras.layers.MaxPooling2D(2),
            
        #     tf.keras.layers.Conv2D(filters = 64, kernel_size=3, activation='relu'),
        #     tf.keras.layers.MaxPooling2D(2),
    
        #     tf.keras.layers.Flatten()
            
        #     # tf.keras.layers.Dense(units=64, activation='relu'),
            
        #     # tf.keras.layers.Dense(tfpl.MultivariateNormalTriL.params_size(len_outputs)),
            
        #     # tfpl.MultivariateNormalTriL(len_outputs, activity_regularizer=kl_regularizer)
        #     # tfp.layers.MultivariateNormalTriL(len_outputs, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1/n_batches), name="output")
        # ])
        
    
        
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer((642, 802, 3)),
            
            conv_reparameterization_layer(16, 3, 'swish'),
            tf.keras.layers.MaxPooling2D(2),
            
            conv_reparameterization_layer(32, 3, 'swish'),
            tf.keras.layers.MaxPooling2D(2),

            conv_reparameterization_layer(64, 3, 'swish'),
            tf.keras.layers.MaxPooling2D(2),

            conv_reparameterization_layer(128, 3, 'swish'),
            tf.keras.layers.GlobalMaxPooling2D(),
            
            tf.keras.layers.Flatten(),

            
           tfk.layers.Dense(10, activation="relu", name="dense_1"),
           
           tfk.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(len(outputs)), 
                            activation=None, 
                            name="distribution_weights"),
           
           tfp.layers.MultivariateNormalTriL(len(outputs), 
                                             activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=1/n_batches), 
                                             name="output")
        ])
        
        
        model.compile(optimizer="adam", loss=neg_log_likelihood)
        # Run training session.
        history = model.fit(train_images, epochs=max_epochs, validation_data=val_images, verbose=False)
        # Describe model.
        model.summary()
        
        
        # def nll(y_true, y_pred):
        #     return -y_pred.log_prob(y_true)

        # model.compile(loss=nll,
        #             optimizer=tf.keras.optimizers.Adam(0.001),
        #             metrics=['accuracy'])

        # model.summary() # Total Params: 196,884

        # history = model.fit(
        #     train_images,
        #     validation_data=val_images,
        #     epochs=max_epochs, #max number of epochs to go over data
        #     #this callback makes the learning stop if the validation loss stops improving for 'patience' epochs in a row. very useful tool should use in other models
        #     callbacks=[
        #         tf.keras.callbacks.EarlyStopping(
        #             monitor='val_loss',
        #             patience=patience,
        #             restore_best_weights=True
        #         )
        #     ],
        #     verbose=1
        # )


        test_labels = test_images.labels
        train_labels = train_images.labels

        # """ makes predictions with the test dataset and plots them. Good predictions should lie on the line. """
        test_predictions = np.squeeze(model.predict(test_images))

        """ gets r^2 value of the test dataset with the predictions made from above ^ """
        metric = tfa.metrics.r_square.RSquare()
        metric.update_state(test_labels, test_predictions)
        training_result = metric.result()
        print("Train R^2 = " + str(training_result.numpy()))

        """ gets r^2 value of the training dataset """
        training_predictions = model.predict(train_images).flatten()
        metric = tfa.metrics.r_square.RSquare()
        metric.update_state(train_labels, training_predictions)
        test_result = metric.result()
        print("Test R^2 = " + str(test_result.numpy()))
        #return {"Training R^2": training_result.numpy(), "Test R^2": test_result.numpy(), "history": history}
        print("min val loss = " + str(min(history.history['val_loss'])))
        print("min loss = " + str(min(history.history['loss'])))

        if(plot == True):
            plot_stuff(test_images, train_images, history, training_result.numpy() , test_result.numpy(), test_predictions, augmentation_list, label_to_predict)
        results.append([training_result.numpy(), test_result.numpy(), min(history.history['val_loss']), min(history.history['loss'])])
    results = np.array(results)
    avg_training_r = np.average(results[:,0])
    avg_test_r = np.average(results[:,1])
    avg_val_loss = np.average(results[:,2])
    avg_train_loss = np.average(results[:,3])
    print("AVERAGE TRAINING R^2: " + str(np.average(results[:,0])))
    print("AVERAGE TEST R^2: " + str(np.average(results[:,1])))
    print("AVERAGE MIN VALIDATION LOSS: " + str(np.average(results[:,2])))
    print("AVERAGE MIN TRAINING LOSS: " + str(np.average(results[:,3])))

    return [history, model, avg_training_r, avg_test_r, avg_val_loss, avg_train_loss]




def plot_stuff(test_images, train_images, history, trainr2, testr2, test_predictions, augmentation_list, label_to_predict):
    test_labels = test_images.labels
    train_labels = train_images.labels
    date_and_time = (str(datetime.datetime.now())[:10]).replace("-", "_")

    fig = plt.figure()
    plt.plot(history.history['loss'], label='loss (mean absolute error)')
    plt.plot(history.history['val_loss'], label='val_loss')
    #plt.ylim([0, 4])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title("loss over epochs predicting " + label_to_predict)
    plt.legend()
    plt.grid(True)
    fig.text(.5, .007, "Best loss = " + str(round(min(history.history['loss'])*10000)/10000) + " Best val_loss = " + str(round(min(history.history['val_loss'])*10000)/10000) + " train r^2 = " + str(round(trainr2*10000)/10000) + " test r^2 = " + str(round(testr2*10000)/10000),fontsize = 7, ha='center')
    fig_name = "C:\\Users\\u1056\\sfx\\Result_plots_after_fixing_augmentation\\" + label_to_predict + "_loss_vs_epochs_" + date_and_time + ".png"
    plt.savefig(fig_name)
    #plt.show()

    fig = plt.figure()
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.title("predictions vs true labels predicting " + label_to_predict)
    lims = [0, max(test_labels)]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims) 
    fig.text(.5, .007, "Best loss = " + str(round(min(history.history['loss'])*10000)/10000) + " Best val_loss = " + str(round(min(history.history['val_loss'])*10000)/10000) + " train r^2 = " + str(round(trainr2*10000)/10000) + " test r^2 = " + str(round(testr2*10000)/10000), fontsize = 8, ha='center')
    fig_name = "C:\\Users\\u1056\\sfx\\Result_plots_after_fixing_augmentation\\" + label_to_predict + "_predictions_vs_true_" + date_and_time + ".png"
    plt.savefig(fig_name)
    #plt.show()




# parent_folder_name = "Highlighted_only_Parietal"
""" limited augmentations below"""
# augmentation_list = ["OG", "Posterize", "Color", "Flipping", "Rotation", "Solarize"]
""" all augmentations below """
# augmentation_list = ["OG", "autoContrast", "Brightness Manipulation", "Color", "Contrast", "Equalize", "Flipping", "Gaussian Noise", "Identity", "Posterize", "Rotation", "Sharpness", "Shearing", "Shifting", "Solarize", "Zooming"]


dataset = "old_dataset/Original"
dataset = "new_dataset/Original"
dataset = "new_dataset/Visible_cracks"
augmentation_list = ["OG", "Posterize", "Color", "Flipping", "Rotation", "Solarize"]

label_to_predict = "phi"
args = prepare_data(dataset, augmentation_list)
phi_result = make_CNN(args, label_to_predict, batch_size=5, patience=20, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True) 

# parent_folder_name = "Original_from_test_matrix"
# parent_folder_name = "Highlighted_only_Parietal"
# label_to_predict = "height"
# args = prepare_data(parent_folder_name, augmentation_list)
# height_result = make_CNN(args, label_to_predict, batch_size=5, patience=20, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True)
#print(result)

# label_to_predict = "phi"
# args = prepare_data(parent_folder_name, augmentation_list)
# phi_result = make_CNN(args, label_to_predict, batch_size=5, patience=20, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True) 

# label_to_predict = "theta"
# args = prepare_data(parent_folder_name, augmentation_list)
# theta_result = make_CNN(args, label_to_predict, batch_size=5, patience=20, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True) 


# print(height_result)
# print(phi_result)
# print(theta_result)
# print("done")
# label_to_predict = "phi"
# parent_folder_name = "With_Width_only_parietal"
# augmentations = ["OG", "Posterize", "Color", "Flipping", "Rotation", "Solarize"]

# results = {}
# for i in range(1, len(augmentations)):
#     augmentation_list = [augmentations[0], augmentations[i]]
#     args = prepare_data(parent_folder_name, augmentation_list)
#     result = make_CNN(args, label_to_predict, batch_size=5, patience=20, max_epochs=100, optimizer="Nadam", activation="relu", kernel_size=(10,10))
#     results[augmentations[i]] = [min(result[0].history['val_loss']), min(result[0].history['loss']), result[2], result[3]]

# print(results)
# print("done")