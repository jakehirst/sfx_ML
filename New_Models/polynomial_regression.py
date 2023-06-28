from linear_regression import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

'''
splits the data into 5 different k-folds of test and training sets
then runs GPR on each of the training sets
then evaluates the models based on their respective test sets.
'''
def Kfold_Polynomial_Regression(degree, full_dataset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict, save_data=True): #TODO change title for different models
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
        
        model = make_pipeline(PolynomialFeatures(degree),LinearRegression())
        model.fit(train_df.to_numpy(), y_train)
        
        y_pred_train  = model.predict(train_df.to_numpy())
        y_pred_test = model.predict(test_df.to_numpy())
        

        if(save_data):
            save_model(model, fold_no, saving_folder, model_type=f'poly_reg_degree') #TODO change model type for different models
            collect_and_save_metrics(y_test, y_pred_test, train_df.__len__(), len(train_df.columns), full_dataset.columns.to_list(), fold_no, saving_folder)
            #plot_test_predictions_heatmap(y_test, y_pred_test, y_pred_test_std, fold_no, saving_folder)
            parody_plot(y_test, y_pred_test, fold_no, saving_folder, label_to_predict, model_type=f'Polynomial Regression degree {degree}')
            
        models.append((model, y_test, test_df))
        fold_no += 1
        
        


full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/New_Crack_Len_FULL_OG_dataframe.csv"
image_folder = '/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx/new_dataset/Visible_cracks'
all_labels = ['height', 'phi', 'theta', 
              'impact site x', 'impact site y', 'impact site z', 
              'impact site r', 'impact site phi', 'impact site theta']

    
    
label_to_predict = 'impact site x'
degree = 2
saving_folder=f'/Users/jakehirst/Desktop/model_results/NEW_DATASET_Poly_Regression_{label_to_predict}_degree{degree}/'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.05)
#correlated_featureset = remove_ABAQUS_features(correlated_featureset)
top_5_features = ['init phi', 'front 1 r', 'avg_ori', 'init r', 'init x']
correlated_featureset = correlated_featureset[top_5_features]
raw_images = []
Kfold_Polynomial_Regression(degree, correlated_featureset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict)


label_to_predict = 'impact site y'
degree = 2
saving_folder=f'/Users/jakehirst/Desktop/model_results/NEW_DATASET_Poly_Regression_{label_to_predict}_degree{degree}/'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.05)
# correlated_featureset = remove_ABAQUS_features(correlated_featureset)
top_5_features = ['front_0_z', 'max_kink', 'angle_btw', 'init theta', 'init y']
correlated_featureset = correlated_featureset[top_5_features]
raw_images = []
Kfold_Polynomial_Regression(degree, correlated_featureset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict)


label_to_predict = 'impact site z'
degree = 2
saving_folder=f'/Users/jakehirst/Desktop/model_results/NEW_DATASET_Poly_Regression_{label_to_predict}_degree{degree}/'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, full_dataset_labels, important_features = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, maximum_p_value=0.05)
# correlated_featureset = remove_ABAQUS_features(correlated_featureset)
top_5_features = ['front_0_x', 'init phi', 'init y', 'init z', 'init theta']
correlated_featureset = correlated_featureset[top_5_features]
raw_images = []
Kfold_Polynomial_Regression(degree, correlated_featureset, raw_images, full_dataset_labels, important_features, saving_folder, label_to_predict)
