from sklearn.model_selection import train_test_split
from prepare_data import *
from CNN import prepare_dataset_Single_Output_Regression
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, precision_score, confusion_matrix, recall_score, f1_score
import os

def split_test_and_trainging_datasets(full_dataset, raw_images, full_dataset_labels):
    #setting aside a test dataset
    np.random.seed(6) #this should reset the randomness to the same randomness so that the test_indicies are the same throughout the tests
    test_indicies = np.random.choice(np.arange(0, len(full_dataset)), size=30, replace=False) #30 for the test dataset
    test_df = full_dataset.iloc[test_indicies]
    test_images = raw_images[test_indicies]
    y_test = full_dataset_labels[test_indicies]
    full_df = full_dataset.drop(test_indicies, axis=0)
    full_images = np.delete(raw_images, test_indicies, axis=0)
    full_dataset_labels = np.delete(full_dataset_labels, test_indicies, axis=0)
    return full_df, test_df, full_images, test_images, full_dataset_labels, y_test

def parody_plot_with_std(y_test, y_pred_test, y_pred_test_std, fold_no, saving_folder):
    plt.figure()
    plt.errorbar(y_test, y_pred_test, yerr=y_pred_test_std, fmt='o')
    plt.plot(y_test, y_test, c='r')
    plt.title('Fold ' + str(fold_no) + ' Gaussian Process Regression, R2=%.2f' % r2_score(y_test, y_pred_test))
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig(saving_folder +  f'/fold_{fold_no}_parody_plot.png')
    # plt.show()
    plt.close()

def Kfold_Gaussian_Process_Regression(full_dataset, raw_images, full_dataset_labels, saving_folder):
    models = []    

    rnge = range(1, len(full_dataset)+1)
    kf5 = KFold(n_splits=5, shuffle=True)
    fold_no = 1
    for train_index, test_index in kf5.split(rnge):
        train_df = full_dataset.iloc[train_index]
        train_images = raw_images[train_index]
        y_train = full_dataset_labels[train_index]
        test_df = full_dataset.iloc[test_index]
        test_images = raw_images[test_index]
        y_test = full_dataset_labels[test_index]
        
        kernel = ConstantKernel(1.0) + ConstantKernel(1.0) * RBF(10)  + WhiteKernel(5) #TODO experiment with the kernel... but this one seems to work.
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(train_df.to_numpy(), y_train)
        y_pred_train, y_pred_train_std = model.predict(train_df.to_numpy(), return_std=True)
        y_pred_test, y_pred_test_std = model.predict(test_df.to_numpy(), return_std=True)
        
        parody_plot_with_std(y_test, y_pred_test, y_pred_test_std, fold_no, saving_folder)
        
        fold_no += 1



full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_code/sfx_ML/Feature_gathering/FULL_OG_dataframe_with_impact_sites_and_Jimmy_RF.csv"
image_folder = '/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx/new_dataset/Visible_cracks'
all_labels = ['height', 'phi', 'theta', 'impact site x', 'impact site y', 'impact site z', 'Jimmy_impact site x', 'Jimmy_impact site y', 
              'Jimmy_impact site z', 'Jimmy_impact site r', 'Jimmy_impact site phi', 'Jimmy_impact site theta']

# label_to_predict = 'Jimmy_impact site x'
# saving_folder=f'/Users/jakehirst/Desktop/model_results/GPR_{label_to_predict}/'
# if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
# correlated_featureset, raw_images, full_dataset_labels = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, =0.01)
# correlated_featureset = remove_ABAQUS_features(correlated_featureset)
# Kfold_Gaussian_Process_Regression(correlated_featureset, raw_images, full_dataset_labels, saving_folder)


label_to_predict = 'Jimmy_impact site y'
saving_folder=f'/Users/jakehirst/Desktop/model_results/GPR_{label_to_predict}/'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, raw_images, full_dataset_labels = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, minimum_p_value=0.01)
correlated_featureset = remove_ABAQUS_features(correlated_featureset)
Kfold_Gaussian_Process_Regression(correlated_featureset, raw_images, full_dataset_labels, saving_folder)


label_to_predict = 'Jimmy_impact site z'
saving_folder=f'/Users/jakehirst/Desktop/model_results/GPR_{label_to_predict}/'
if(not os.path.exists(saving_folder)): os.mkdir(saving_folder)
correlated_featureset, raw_images, full_dataset_labels = prepare_dataset_Single_Output_Regression(full_dataset_pathname, image_folder, label_to_predict, all_labels, saving_folder=None, minimum_p_value=0.01)
correlated_featureset = remove_ABAQUS_features(correlated_featureset)
Kfold_Gaussian_Process_Regression(correlated_featureset, raw_images, full_dataset_labels, saving_folder)