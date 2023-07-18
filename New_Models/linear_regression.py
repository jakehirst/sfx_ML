from sklearn.linear_model import LinearRegression
import numpy as np
from GPR import *

def save_model(model, fold_no, saving_folder, model_type):
    # Save the model to a file
    filename = saving_folder + f'/{model_type}_model_fold{fold_no}.sav'
    pickle.dump(model, open(filename, 'wb'))

'''
makes a parody plot of the predictions from GPR including the standard deviations
'''
def parody_plot(y_test, y_pred_test, fold_no, saving_folder, label_to_predict, model_type):
    plt.figure()
    plt.plot(y_test, y_test, c='r')
    plt.plot(y_test, y_pred_test, 'o')
    plt.title('Fold ' + str(fold_no) + f' {model_type}' + ', R2=%.2f' % r2_score(y_test, y_pred_test))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(saving_folder +  f'/{label_to_predict}_fold_{fold_no}_parody_plot.png')
    # plt.show()
    plt.close()

'''
splits the data into 5 different k-folds of test and training sets
then runs Linear Regression on each of the training sets
then evaluates the models based on their respective test sets.
'''
def Kfold_Linear_Regression(full_dataset, full_dataset_labels, important_features, saving_folder, label_to_predict, save_data=True, num_training_points=False): #TODO change title for different models
    # correlated_featureset, raw_images, full_dataset_labels = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.01)
    #full_dataset = remove_ABAQUS_features(full_dataset)
    models = []
    
    rnge = range(1, len(full_dataset)+1)
    kf5 = KFold(n_splits=5, shuffle=True)
    fold_no = 1
    for train_index, test_index in kf5.split(rnge):
        train_df = full_dataset.iloc[train_index]
        # train_images = raw_images[train_index]
        y_train = full_dataset_labels[train_index]
        test_df = full_dataset.iloc[test_index]
        # test_images = raw_images[test_index]
        y_test = full_dataset_labels[test_index]
        
        
        """ if we want to limit the number of training datapoints """
        if(not num_training_points == False):
            train_df.reset_index(drop=True, inplace=True)
            train_indicies = np.random.choice(np.arange(0, len(train_df)), size=num_training_points, replace=False)
            train_df = train_df.iloc[train_indicies]
            y_train = y_train[train_indicies]
            
            
        model = LinearRegression() #TODO change model type for different models

        model.fit(train_df.to_numpy(), y_train)
        y_pred_train  = model.predict(train_df.to_numpy())
        y_pred_test = model.predict(test_df.to_numpy())
        
        if(save_data):
            save_model(model, fold_no, saving_folder, model_type='linear_reg') #TODO change model type for different models
            # collect_and_save_metrics(y_test, y_pred_test, train_df.__len__(), len(train_df.columns), full_dataset.columns.to_list(), fold_no, saving_folder)
            collect_and_save_metrics(y_train, y_pred_train, y_test, y_pred_test, list(train_df.columns), fold_no, saving_folder)
            #plot_test_predictions_heatmap(y_test, y_pred_test, y_pred_test_std, fold_no, saving_folder)
            parody_plot(y_test, y_pred_test, fold_no, saving_folder, label_to_predict, model_type='Linear Regression')
            
        models.append((model, y_test, test_df))
        fold_no += 1
        


