from linear_regression import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
'''
splits the data into 5 different k-folds of test and training sets
then runs GPR on each of the training sets
then evaluates the models based on their respective test sets.
'''
def Kfold_Ridge_Regression(alpha, full_dataset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict, save_data=True): #TODO change title for different models
    # correlated_featureset, raw_images, full_dataset_labels = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.01)

    full_dataset = remove_ABAQUS_features(full_dataset)
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
        
        # Create the Ridge regression model
        # alpha = Regularization strength
        model = Ridge(alpha=alpha)
        
        model.fit(train_df.to_numpy(), y_train)        
        
        y_pred_train  = model.predict(train_df.to_numpy())
        y_pred_test = model.predict(test_df.to_numpy())
        

        if(save_data):
            save_model(model, fold_no, saving_folder, model_type=f'ridge_reg_alpha') #TODO change model type for different models
            # collect_and_save_metrics(y_test, y_pred_test, train_df.__len__(), len(train_df.columns), full_dataset.columns.to_list(), fold_no, saving_folder)
            collect_and_save_metrics(y_train, y_pred_train, y_test, y_pred_test, list(train_df.columns), fold_no, saving_folder)
            #plot_test_predictions_heatmap(y_test, y_pred_test, y_pred_test_std, fold_no, saving_folder)
            parody_plot(y_test, y_pred_test, fold_no, saving_folder, label_to_predict, model_type=f'Ridge Regression alpha={alpha}')
            
            
        models.append((model, y_test, test_df))
        fold_no += 1
        
        


