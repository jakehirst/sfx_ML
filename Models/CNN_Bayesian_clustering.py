import pandas as pd
from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import KFold
from Binning_phi_and_theta import *
#from PIL import Image
import os
from k_means_clustering import *
from tensorflow import keras
from CNN_function_library import *
import tensorflow_probability as tfp
import imageio

tfd = tfp.distributions
tfpl = tfp.layers

#https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/bayesian_neural_network.py
#https://www.tensorflow.org/probability/examples/Probabilistic_Layers_Regression



# For Reparameterization Layers
def custom_normal_prior(dtype, shape, name, trainable, add_variable_fn):
    distribution = tfd.Normal(loc = 0.1 * tf.ones(shape, dtype),
                              scale = 0.003 * tf.ones(shape, dtype))
    batch_ndims = tf.size(distribution.batch_shape_tensor())
    
    distribution = tfd.Independent(distribution,
                                   reinterpreted_batch_ndims = batch_ndims)
    return distribution
    
# def laplace_prior(dtype, shape, name, trainable, add_variable_fn):
#     distribution = tfd.Laplace(loc = tf.zeros(shape, dtype),
#                                scale = tf.ones(shape, dtype))
#     batch_ndims = tf.size(distribution.batch_shape_tensor())
    
#     distribution = tfd.Independent(distribution,
#                                    reinterpreted_batch_ndims = batch_ndims)
#     return distribution 

def approximate_kl(q, p, q_tensor):
    return tf.reduce_mean(q.log_prob(q_tensor) - p.log_prob(q_tensor))

total_samples = 100
divergence_fn = lambda q, p, q_tensor : approximate_kl(q, p, q_tensor) / total_samples


def conv_reparameterization_layer(filters, kernel_size, activation):
    # For simplicity, we use default prior and posterior.
    # In the next parts, we will use custom mixture prior and posteriors.
    return tfpl.Convolution2DReparameterization(
            filters = filters,
            kernel_size = kernel_size,
            activation = activation, 
            padding = 'same',
            kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_prior_fn = tfpl.default_multivariate_normal_fn,
            
            bias_prior_fn = tfpl.default_multivariate_normal_fn,
            bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular=False),
            
            kernel_divergence_fn = divergence_fn,
            bias_divergence_fn = divergence_fn)


def make_CNN(args, label_to_predict, batch_size=1, patience=25, max_epochs=1000, optimizer="adam", activation="relu", kernel_size=(5,5), plot = True, augmentation_list = [], num_folds=5, k=5, folder="No folder applied", num_tries = 10):
    results = []
    
    if(folder == "No folder applied"): print("NEED TO ASSIGN A FOLDER")
    
    folder = folder + f"/{k}_k"
    if(not os.path.exists(folder)):
        os.mkdir(folder)
    
    #TODO: do this for phi and theta later
    #shuffles the dataset and puts it into a dataframe
    phiandtheta_df = pd.concat([args[0], args[2], args[3]], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
    phiandtheta_df.columns = ["Filepath", "phi", "theta"]
    
    #phiandtheta_df = phiandtheta_df.iloc[0:10]
    #find_clustering_elbow(phiandtheta_df, 2000)
    #main_clustering_call(phiandtheta_df, 5, 1000)

    df, clusters, y_col_values = main_clustering_call(phiandtheta_df, k, num_tries, folder)


    kfold = KFold(n_splits=num_folds, shuffle=True)
    inputs = np.array(df["Filepath"])
    outputs = np.array(df[y_col_values])

    all_test_predictions = [] #test predictions from each of the folds

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
        #train_df = add_augmentations_bins(train_df, augmentation_list)



        #flow the images through the generators
        train_images = train_generator.flow_from_dataframe(
            dataframe=train_df,
            x_col = 'Filepath',
            y_col = y_col_values,
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
            y_col = y_col_values,
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
            y_col = y_col_values,
            target_size=args[4][:2],  #can reduce the images to a certain size to reduce training time. 120x120 for example here
            color_mode='rgb',
            class_mode = 'raw', #keeps the classes of our labels the same after flowing
            batch_size=batch_size, #can increase this to up to like 10 or so for how much data we have
            shuffle=False,
        )

        # #pulling the image matricies from the filepaths of images
        # test_inputs = np.empty((0,args[4][0], args[4][1], args[4][2]))
        # test_outputs = np.empty((0,4))
        # train_inputs = np.empty((0,args[4][0], args[4][1], args[4][2]))
        # train_outputs = np.empty((0,4))
        # val_inputs = np.empty((0,args[4][0], args[4][1], args[4][2]))
        # val_outputs = np.empty((0,4))
        
        # for i in range(test_images.__len__()):
        #     example = test_images[i]
        #     test_outputs = np.append(test_outputs, example[1], axis=0)
        #     test_inputs = np.append(test_inputs, example[0], axis=0)
        # for i in range(train_images.__len__()):
        #     example = train_images[i]
        #     train_outputs = np.append(train_outputs, example[1], axis=0)
        #     train_inputs = np.append(train_inputs, example[0], axis=0)
        # for i in range(val_images.__len__()):
        #     example = val_images[i]
        #     val_outputs = np.append(val_outputs, example[1], axis=0)
        #     val_inputs = np.append(val_inputs, example[0], axis=0)
        # test_outputs = test_outputs.astype(np.float32)
        # test_inputs = test_inputs.astype(np.float32)
        # train_outputs = train_outputs.astype(np.float32)
        # train_inputs = train_inputs.astype(np.float32)
        # val_outputs = val_outputs.astype(np.float32)
        # val_inputs = val_inputs.astype(np.float32)
        # test_inputs = []
        # train_inputs = []
        # val_inputs = []

        # for example in test_images:
        #     image = example[0]
        #     label = example[1]
        #     print("hi")
        # #puttin the rgb matricies into train_inputs, test_inputs, and val_inputs
        # #to see the image, use plt.imshow(arr)
        # for image in test_images._filepaths:
        #     arr = imageio.v2.imread(image)[:,:,0:3] / 255.0
        #     test_inputs.append(arr)
        # for image in train_images._filepaths:
        #     arr = imageio.v2.imread(image)[:,:,0:3] / 255.0
        #     train_inputs.append(arr)
        # for image in val_images._filepaths:
        #     arr = imageio.v2.imread(image)[:,:,0:3] / 255.0
        #     val_inputs.append(arr)

        # test_inputs = np.array(test_inputs)
        # train_inputs = np.array(train_inputs)
        # val_inputs = np.array(val_inputs)
        #https://datascience.stackexchange.com/questions/106600/how-to-perform-regression-on-image-data-using-tensorflow

        #COMMENT this one works but not very well
        
        """ regular cnn model """
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(16, 3, activation = 'relu', 
        #                         input_shape = (642, 802, 3)),
        #     tf.keras.layers.MaxPooling2D(2),
            
        #     tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
        #     tf.keras.layers.MaxPooling2D(2),

        #     tf.keras.layers.Conv2D(64, 3, activation = 'relu'),
        #     tf.keras.layers.MaxPooling2D(2),

        #     tf.keras.layers.Conv2D(128, 3, activation = 'relu'),
        #     tf.keras.layers.MaxPooling2D(),
        #     tf.keras.layers.Flatten(),
            
        #     tf.keras.layers.Dense(units=128, activation='relu'),

        #     tf.keras.layers.Dense(k, activation = 'softmax')
        # ])
        """ regular cnn model """

        """ bayesian cnn model """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer((642, 802, 3)),
            
            conv_reparameterization_layer(8, 3, 'relu'),
            tf.keras.layers.MaxPooling2D(2),
            
            conv_reparameterization_layer(16, 3, 'relu'),
            tf.keras.layers.MaxPooling2D(),   
                     
            conv_reparameterization_layer(32, 3, 'relu'),
            tf.keras.layers.MaxPooling2D(2),

            conv_reparameterization_layer(64, 3, 'relu'),
            tf.keras.layers.MaxPooling2D(2),

            conv_reparameterization_layer(128, 3, 'relu'),
            tf.keras.layers.GlobalMaxPooling2D(),
            
            tfpl.DenseReparameterization(
                units = tfpl.OneHotCategorical.params_size(k), activation = None,
                kernel_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular=False),
                kernel_prior_fn = tfpl.default_multivariate_normal_fn,
                
                bias_prior_fn = tfpl.default_multivariate_normal_fn,
                bias_posterior_fn = tfpl.default_mean_field_normal_fn(is_singular=False),
                
                kernel_divergence_fn = divergence_fn,
                bias_divergence_fn = divergence_fn),
            tfpl.OneHotCategorical(k)
        ])
    
        """ bayesian cnn model """

        
        def nll(y_true, y_pred):
            # print("y_true = " + str(y_true))
            # print("y_pred = " + str(y_pred))
            return -y_pred.log_prob(y_true)

        model.compile(loss="categorical_crossentropy",#loss=nll,
                    optimizer=tf.keras.optimizers.Adam(0.001),
                    metrics=['accuracy'])

        model.summary()
        
        history = model.fit(
            train_images,
            validation_data=val_images,
            epochs=max_epochs, #max number of epochs to go over data
            #this callback makes the learning stop if the validation loss stops improving for 'patience' epochs in a row. very useful tool should use in other models
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True
                )
                #,tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 30))
            ],
            verbose=1 #prints training process
        )
        #COMMENT this one works but not very well

        
        #COMMENT NEW TRY
        # NUM_TRAIN_EXAMPLES = train_images.__len__()
        # NUM_CLASSES = k
        # kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
        #                     tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32))
        
        # model = tf.keras.models.Sequential([
        #     tfp.layers.Convolution2DFlipout(
        #         6, kernel_size=5, padding='SAME',
        #         kernel_divergence_fn=kl_divergence_function,
        #         activation=tf.nn.relu),
        #     tf.keras.layers.MaxPooling2D(
        #         pool_size=[2, 2], strides=[2, 2],
        #         padding='SAME'),
        #     tfp.layers.Convolution2DFlipout(
        #         16, kernel_size=5, padding='SAME',
        #         kernel_divergence_fn=kl_divergence_function,
        #         activation=tf.nn.relu),
        #     tf.keras.layers.MaxPooling2D(
        #         pool_size=[2, 2], strides=[2, 2],
        #         padding='SAME'),
        #     tfp.layers.Convolution2DFlipout(
        #         120, kernel_size=5, padding='SAME',
        #         kernel_divergence_fn=kl_divergence_function,
        #         activation=tf.nn.relu),
        #     tf.keras.layers.Flatten(),
        #     tfp.layers.DenseFlipout(
        #         84, kernel_divergence_fn=kl_divergence_function,
        #         activation=tf.nn.relu),
        #     tfp.layers.DenseFlipout(
        #         NUM_CLASSES, kernel_divergence_fn=kl_divergence_function,
        #         activation=tf.nn.softmax)
        # ])

        # # Model compilation.
        # optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        # # We use the categorical_crossentropy loss since the MNIST dataset contains
        # # ten labels. The Keras API will then automatically add the
        # # Kullback-Leibler divergence (contained on the individual layers of
        # # the model), to the cross entropy loss, effectively
        # # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
        # model.compile(optimizer, loss='categorical_crossentropy',
        #         metrics=['accuracy'], experimental_run_tf_function=False)

        # model.build(input_shape=[None, 642, 802, 3])
        
        # history = model.fit(
        #     train_images,
        #     validation_data=val_images,
        #     epochs=max_epochs, #max number of epochs to go over data
        #     #this callback makes the learning stop if the validation loss stops improving for 'patience' epochs in a row. very useful tool should use in other models
        #     # callbacks=[
        #     #     tf.keras.callbacks.EarlyStopping(
        #     #         monitor='val_loss',
        #     #         patience=patience,
        #     #         restore_best_weights=True
        #     #     )
        #     #     #,tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 ** (epoch / 30))
        #     # ],
        #     verbose=1 #prints training process
        # )
        
        # print(' ... Training convolutional neural network')
        # for epoch in range(FLAGS.num_epochs):
        #     epoch_accuracy, epoch_loss = [], []
        #     for step, (batch_x, batch_y) in enumerate(train_seq):
        #         batch_loss, batch_accuracy = model.train_on_batch(
        #             batch_x, batch_y)
        #         epoch_accuracy.append(batch_accuracy)
        #         epoch_loss.append(batch_loss)

        #         if step % 100 == 0:
        #             print('Epoch: {}, Batch index: {}, '
        #                 'Loss: {:.3f}, Accuracy: {:.3f}'.format(
        #                     epoch, step,
        #                     tf.reduce_mean(epoch_loss),
        #                     tf.reduce_mean(epoch_accuracy)))

        #         if (step+1) % FLAGS.viz_steps == 0:
        #             # Compute log prob of heldout set by averaging draws from the model:
        #             # p(heldout | train) = int_model p(heldout|model) p(model|train)
        #             #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
        #             # where model_i is a draw from the posterior p(model|train).
        #             print(' ... Running monte carlo inference')
        #             probs = tf.stack([model.predict(heldout_seq, verbose=1)
        #                             for _ in range(FLAGS.num_monte_carlo)], axis=0)
        #             mean_probs = tf.reduce_mean(probs, axis=0)
        #             heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
        #             print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))

        #             if HAS_SEABORN:
        #                 names = [layer.name for layer in model.layers
        #                         if 'flipout' in layer.name]
        #                 qm_vals = [layer.kernel_posterior.mean().numpy()
        #                             for layer in model.layers
        #                             if 'flipout' in layer.name]
        #                 qs_vals = [layer.kernel_posterior.stddev().numpy()
        #                             for layer in model.layers
        #                             if 'flipout' in layer.name]
        #                 plot_weight_posteriors(names, qm_vals, qs_vals,
        #                                         fname=os.path.join(
        #                                             FLAGS.model_dir,
        #                                             'epoch{}_step{:05d}_weights.png'.format(
        #                                                 epoch, step)))
        #                 plot_heldout_prediction(heldout_seq.images, probs.numpy(),
        #                                         fname=os.path.join(
        #                                             FLAGS.model_dir,
        #                                             'epoch{}_step{}_pred.png'.format(
        #                                                 epoch, step)),
        #                                         title='mean heldout logprob {:.2f}'
        #                                         .format(heldout_log_prob))



        #COMMENT NEW TRY

        test_labels = test_images.labels
        train_labels = train_images.labels

        # """ makes predictions with the test dataset and plots them. Good predictions should lie on the line. """
        test_predictions = np.squeeze(model.predict(test_images))
        all_test_predictions.append((test_predictions, test_images._filepaths))
        
        print("min val loss = " + str(min(history.history['val_loss'])))

        #making nice plots to look at :)
        label_to_predict = "Orientation via bins"


    #     for i in range(len(test_predictions)):
    #         make_sphere(bins_and_values, test_predictions[i], test_images._filepaths[i], folder)
        plot_stuff(history, label_to_predict, folder, fold_no) #this has to be after make_sphere because make_sphere makes the folder duh

        fold_no += 1
    
    confusion_matrix(all_test_predictions, df, folder)
    Plot_Bins_and_misses(clusters, all_test_predictions, df, folder)
    
    # print(prediction_sheet)
    print("done")



    #     """ gets r^2 value of the test dataset with the predictions made from above ^ """
    #     metric = tfa.metrics.r_square.RSquare()
    #     metric.update_state(test_labels, test_predictions)
    #     training_result = metric.result()
    #     print("Train R^2 = " + str(training_result.numpy()))

    #     """ gets r^2 value of the training dataset """
    #     training_predictions = model.predict(train_images).flatten()
    #     metric = tfa.metrics.r_square.RSquare()
    #     metric.update_state(train_labels, training_predictions)
    #     test_result = metric.result()
    #     print("Test R^2 = " + str(test_result.numpy()))
    #     #return {"Training R^2": training_result.numpy(), "Test R^2": test_result.numpy(), "history": history}
    #     print("min val loss = " + str(min(history.history['val_loss'])))
    #     print("min loss = " + str(min(history.history['loss'])))

    #     if(plot == True):
    #         plot_stuff(test_images, train_images, history, training_result.numpy() , test_result.numpy(), test_predictions, augmentation_list, label_to_predict, fold_no)
    #     results.append([training_result.numpy(), test_result.numpy(), min(history.history['val_loss']), min(history.history['loss'])])
    #     fold_no += 1

    # results = np.array(results)
    # avg_training_r = np.average(results[:,0])
    # avg_test_r = np.average(results[:,1])
    # avg_val_loss = np.average(results[:,2])
    # avg_train_loss = np.average(results[:,3])
    # print("AVERAGE TRAINING R^2: " + str(np.average(results[:,0])))
    # print("AVERAGE TEST R^2: " + str(np.average(results[:,1])))
    # print("AVERAGE MIN VALIDATION LOSS: " + str(np.average(results[:,2])))
    # print("AVERAGE MIN TRAINING LOSS: " + str(np.average(results[:,3])))

    return model
    # return [history, model, avg_training_r, avg_test_r, avg_train_loss, avg_val_loss]

def plot_stuff(history, label_to_predict, folder, fold_no):
    folder = folder + f"/fold {fold_no}"
    if(not os.path.exists(folder)):
        os.mkdir(folder)
    
    fig = plt.figure()
    plt.plot(history.history['loss'], label='loss (categorical cross entropy)')
    plt.plot(history.history['val_loss'], label='val_loss')
    #plt.ylim([0, 4])
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title(f"fold {fold_no} loss over epochs predicting " + label_to_predict)
    plt.legend()
    plt.grid(True)
    plt.ylim(0, max(history.history['val_loss'])+2)
    fig.text(.5, .007, "Best val_loss = " + str(round(min(history.history['val_loss'])*10000)/10000),fontsize = 7, ha='center')
    fig_name = folder + "/loss_vs_epochs.png"
    plt.savefig(fig_name)
    plt.close()

    fig = plt.figure()
    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label='validation accuracy')
    #plt.ylim([0, 4])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f"fold {fold_no} Accuracy over epochs predicting " + label_to_predict)
    plt.legend()
    plt.grid(True)
    fig.text(.5, .007, "Best validation accuracy = " + str(round(max(history.history['val_acc'])*10000)/10000),fontsize = 7, ha='center')
    fig_name = folder + "/accuracy_vs_epochs.png"
    plt.savefig(fig_name)
    plt.close()
    

def check_bins_plot(hit_arr, num_tests, bins_checked_arr, fold, num_bins):
    misses_arr = []
    for i in range(len(hit_arr)):
        misses_arr.append(num_tests - hit_arr[i])
    
    x = np.arange(len(bins_checked_arr))
    width = 0.35

    fig, ax = plt.subplots()
    hit_bar = ax.bar(x - width/2, hit_arr, width, label="hits")
    miss_bar = ax.bar(x + width/2, misses_arr, width, label="misses")

    ax.set_xticks(x)
    ax.set_xticklabels(bins_checked_arr)
    ax.legend()
    ax.set_title("#hits vs #bins checked FOLD " + str(fold))
    ax.set_xlabel("#bins checked")
    ax.set_ylabel("#hits/misses")

    plt.savefig("C:\\Users\\u1056\\sfx\\bin_plots_" + str(num_bins) + "_bins_3\\Bins_plot_FOLD" + str(fold) +".png")  
    plt.close()

    return





# parent_folder_name = "Highlighted_only_Parietal"
""" limited augmentations below"""
# augmentation_list = ["OG", "Posterize", "Color", "Flipping", "Rotation", "Solarize"]
""" all augmentations below """
# augmentation_list = ["OG", "autoContrast", "Brightness Manipulation", "Color", "Contrast", "Equalize", "Flipping", "Gaussian Noise", "Identity", "Posterize", "Rotation", "Sharpness", "Shearing", "Shifting", "Solarize", "Zooming"]


parent_folder_name = "Original"
# parent_folder_name = "Original_from_test_matrix"
#parent_folder_name = "Highlighted_only_Parietal"

label_to_predict = "binned_orientation"
augmentation_list = ["OG", "Posterize", "Color", "Flipping", "Rotation", "Solarize"]
args = prepare_data(parent_folder_name, augmentation_list)

#in lab
# folder = "some folder"

#at home
folder = "/Users/jakehirst/Desktop/sfx/bayesian_clustering_test"
num_tries = 100

make_CNN(args, label_to_predict, batch_size=5, patience=10, max_epochs=100, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, k=2, folder=folder, num_tries=num_tries)
# make_CNN(args, label_to_predict, batch_size=5, patience=50, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, k=3, folder=folder, num_tries=num_tries)
# make_CNN(args, label_to_predict, batch_size=5, patience=50, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, k=4, folder=folder, num_tries=num_tries)
# make_CNN(args, label_to_predict, batch_size=5, patience=50, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, k=5, folder=folder, num_tries=num_tries)
# make_CNN(args, label_to_predict, batch_size=5, patience=50, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, k=6, folder=folder, num_tries=num_tries)
# make_CNN(args, label_to_predict, batch_size=5, patience=50, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, k=7, folder=folder, num_tries=num_tries)
# make_CNN(args, label_to_predict, batch_size=5, patience=50, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, k=8, folder=folder, num_tries=num_tries)
# make_CNN(args, label_to_predict, batch_size=5, patience=50, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, k=9, folder=folder, num_tries=num_tries)
# make_CNN(args, label_to_predict, batch_size=5, patience=50, max_epochs=500, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, k=10, folder=folder, num_tries=num_tries)


print("done")
#make_CNN(args, label_to_predict, batch_size=5, patience=3, max_epochs=20, optimizer="Nadam", activation="relu", kernel_size=(3,3), augmentation_list=augmentation_list, plot=True, num_bins=6)