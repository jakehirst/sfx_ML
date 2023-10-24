import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import os
import datetime


full_dataset_pathname = "/Users/jakehirst/Desktop/sfx/sfx_ML_data/New_Crack_Len_FULL_OG_dataframe_2023_10_23.csv"
image_folder = '/Users/jakehirst/Desktop/sfx/sfx_pics/jake/images_sfx/new_dataset/Visible_cracks'
all_labels = ['height', 'phi', 'theta', 
            'impact site x', 'impact site y', 'impact site z', 
            'impact site r', 'impact site phi', 'impact site theta']

# features = ['init theta', 'init phi',
#        'init r', 'init z', 'init y', 'init x', 'dist btw frts', 'crack len',
#        'linearity', 'max thickness', 'mean thickness', 'max_kink',
#        'abs_val_mean_kink', 'mean_kink', 'sum_kink', 'abs_val_sum_kink',
#        'avg_ori', 'angle_btw', 'height', 'phi', 'theta', 'impact site theta',
#        'impact site phi', 'impact site r', 'impact site z', 'impact site y',
#        'impact site x']

def plot_feature_vs_label(df, feature, label):
    x = df[label].to_numpy()
    y = df[feature].to_numpy()
    corr, p_val = pearsonr(x, y)
    plt.scatter(x, y, c='r', alpha=0.5)
    plt.xlabel(label)
    plt.ylabel(feature)
    plt.subplots_adjust(top=0.85)
    plt.title(f'{feature} vs {label}\nCorrelation = {np.round(corr, 3)}\nP-value = {"{:e}".format(p_val)}')
    plt.show()
    return

def add_feature_to_df(df, column_to_add, column_name):
    new_df = df.copy()
    column_to_add.columns = [column_name]
    new_df[column_to_add.columns[0]] = column_to_add
    return new_df




def multiply_two_features(df, feature1, feature2):
    column = df[feature1] * df[feature2]
    column_name = f'{feature1} * {feature2}'
    return column, column_name

def divide_two_features(df, feature1, feature2):
    columns = []
    column_names = []
    columns.append(df[feature1] / (df[feature2]+0.1))
    column_names.append(f'{feature1} / {feature2}')
    columns.append(df[feature2] / (df[feature1]+0.1))
    column_names.append(f'{feature2} / {feature1}')
    return columns, column_names

def get_exp_of_two_df_columns(df, base_feat, exp_feat, negative_exp=False):
    threshold = 1e37 #preventing inf's from happening
    if(negative_exp):
        result = df.apply(lambda row: max(min(row[base_feat] ** -row[exp_feat], threshold), -threshold), axis=1)
    else:
        result = df.apply(lambda row: max(min(row[base_feat] ** row[exp_feat], threshold), -threshold), axis=1)

    # Replace NaN with 0.0
    result = result.fillna(threshold)

    return result






def exp_two_features(df, feature1, feature2):
    columns = []
    column_names = []
    columns.append(get_exp_of_two_df_columns(df, feature1, feature2, negative_exp=False))
    column_names.append(f'{feature1} ^ {feature2}')
    columns.append(get_exp_of_two_df_columns(df, feature2, feature1, negative_exp=False))
    column_names.append(f'{feature2} ^ {feature1}')
    
    columns.append(get_exp_of_two_df_columns(df, feature1, feature2, negative_exp=True))
    column_names.append(f'{feature1} ^ -{feature2}')
    columns.append(get_exp_of_two_df_columns(df, feature2, feature1, negative_exp=True))
    column_names.append(f'{feature2} ^ -{feature1}')
    return columns, column_names

def square_feature(df, feature):
    column = df[feature]**2
    column_name = f'{feature}^2'
    return column, column_name

def cube_feature(df, feature):
    column = df[feature]**3
    column_name = f'{feature}^3'
    return column, column_name

def sqrt_feature(df, feature):
    column = np.sqrt(df[feature])
    column_name = f'sqrt({feature})'
    return pd.Series(column), column_name

def exp_feature(df, feature):
    column = np.exp(df[feature])
    column_name = f'exp({feature})'
    return pd.Series(column), column_name

def log_feature(df, feature):
    column = np.log(df[feature])
    column_name = f'log({feature})'
    column = pd.Series(column)
    column = column.clip(lower=-100000) #limits the lowest value to be -100000
    return column, column_name

def save_new_features(folder, new_features, new_features_df, label, combination_type):
    feature_list = []
    corr_list = []
    p_val_list = []
    # Iterate through features
    for feature in new_features:
        corr, p_val = pearsonr(new_features_df[feature], new_features_df[label])
        feature_list.append(feature)
        corr_list.append(corr)
        p_val_list.append(p_val)
        

    result_df = pd.DataFrame({'Feature': feature_list, 'Correlation': corr_list, 'P-Value': p_val_list})
    # Create a new column with absolute values of 'Age'
    result_df['Abs_val_corr'] = result_df['Correlation'].abs()
    # Sort the DataFrame by the absolute values of 'Age' in ascending order
    sorted_df = result_df.sort_values(by='Abs_val_corr', ascending=False)
    
    sorted_df.to_csv(folder + f'/{combination_type}_feature_correlation_results.csv', index=False)
    return

''' 
does a feature interaction between two features, and then saves the correlation between the label and the new feature 

combination_type could be one of 3 things:
- multiply_two_feats (x * y)
- divide_two_feats (x/y and y/x)
- exponential_two_feats (x^y and y^x and x^-y and y^-x)
'''
def get_feature_interactions(df, list_of_multiplying_features, combination_type, label, saving_folder):
    all_labels = ['height', 'phi', 'theta', 
            'impact site x', 'impact site y', 'impact site z', 
            'impact site r', 'impact site phi', 'impact site theta']
    original_features = df.columns
    original_features = original_features.drop(all_labels)

    new_features_df = df.copy()
    already_done = []
    for i in range(len(list_of_multiplying_features)):
        for j in range(len(list_of_multiplying_features)):
            feat1 = list_of_multiplying_features[i]
            feat2 = list_of_multiplying_features[j]
            
            if(i == j): 
                continue #dont want to repeat features being made
            if(already_done.__contains__((feat1, feat2)) or already_done.__contains__((feat2, feat1))): 
                continue
            already_done.append((feat1, feat2))
            
            if(combination_type == 'multiply_two_feats'):
                new_column, column_name = multiply_two_features(new_features_df, feat1, feat2)
                new_features_df = add_feature_to_df(new_features_df, new_column, column_name)
            elif(combination_type == 'divide_two_feats'):
                new_columns, column_names = divide_two_features(new_features_df, feat1, feat2)
                for i in range(len(new_columns)):
                    new_features_df = add_feature_to_df(new_features_df, new_columns[i], column_names[i])
            elif(combination_type == 'exponential_two_feats'):
                new_columns, column_names = exp_two_features(new_features_df, feat1, feat2)
                for i in range(len(new_columns)):
                    new_features_df = add_feature_to_df(new_features_df, new_columns[i], column_names[i])
                print('here')
                    
    new_features_df = new_features_df.drop(columns=original_features)
    folder = saving_folder + f'/{label}/{combination_type}'
    if(not os.path.exists(folder)): os.makedirs(folder, exist_ok=True)
    new_features_df.to_csv(folder + f'/{combination_type}_feature_dataframe.csv', index=False)

    new_features = new_features_df.columns.to_numpy()
    #remove all labels from the new features
    mask = np.isin(new_features, all_labels, invert=True)
    new_features = new_features[mask]
    
    save_new_features(folder, new_features, new_features_df, label, combination_type)
    
    return


''' 
does a feature transformation, and then saves the correlation between the label and the new feature 

transformation_type could be one of 5 things:
- square (feature ** 2)
- cube (feature ** 3)
- sqrt (feature ** .5)
- exp (exp(feature))
- log (log(feature))
'''
def get_feature_transformation(df, list_of_multiplying_features, transformation_type, label, saving_folder):
    all_labels = ['height', 'phi', 'theta', 
            'impact site x', 'impact site y', 'impact site z', 
            'impact site r', 'impact site phi', 'impact site theta']
    original_features = df.columns
    original_features = original_features.drop(all_labels)

    new_features_df = df.copy()
    for feature in list_of_multiplying_features:        
        if(transformation_type == 'square'):
            new_column, column_name = square_feature(new_features_df, feature)
        elif(transformation_type == 'cube'):
            new_column, column_name = cube_feature(new_features_df, feature)
        elif(transformation_type == 'sqrt'):
            if((new_features_df[feature] < 0).any()): continue #TODO could do a transformation here... and make everything positive by subtracting the minimum
            else: new_column, column_name = sqrt_feature(new_features_df, feature)
        elif(transformation_type == 'exp'):
            new_column, column_name = exp_feature(new_features_df, feature)
        elif(transformation_type == 'log'):
            if((new_features_df[feature] < 0).any()): continue #TODO could do a transformation here... and make everything positive by subtracting the minimum
            else: new_column, column_name = log_feature(new_features_df, feature)
        elif(transformation_type == 'nothin'):
            new_column = df[feature]
            column_name = f'{feature} (unchanged)'

            
        new_features_df = add_feature_to_df(new_features_df, new_column, column_name)

    new_features_df = new_features_df.drop(columns=original_features)
    folder = saving_folder + f'/{label}/{transformation_type}'
    if(not os.path.exists(folder)): os.makedirs(folder, exist_ok=True)
    new_features_df.to_csv(folder + f'/{transformation_type}_feature_dataframe.csv', index=False)

    new_features = new_features_df.columns.to_numpy()
    #remove all labels from the new features
    mask = np.isin(new_features, all_labels, invert=True)
    new_features = new_features[mask]
    
    save_new_features(folder, new_features, new_features_df, label, transformation_type)
    return

def put_everything_into_a_single_csv(folder, label):
    all_labels = ['height', 'phi', 'theta', 
            'impact site x', 'impact site y', 'impact site z', 
            'impact site r', 'impact site phi', 'impact site theta']
    label_folder = folder + f'/{label}'
    dirs = os.listdir(label_folder)
    dirs = [dir for dir in dirs if not dir.endswith('.csv')]
    
    all_correlations = pd.DataFrame()
    all_features = pd.DataFrame()
    for dir in dirs:
        correlations = pd.read_csv(label_folder + f'/{dir}/{dir}_feature_correlation_results.csv')
        features = pd.read_csv(label_folder + f'/{dir}/{dir}_feature_dataframe.csv')
        
        all_correlations = pd.concat([all_correlations, correlations], axis=0, ignore_index=True)
        all_features = pd.concat([all_features, features], axis=1)

    all_features = all_features.loc[:,~all_features.columns.duplicated()] #removes all features or label columns that have been duplicated
    all_correlations = all_correlations.sort_values(by='Abs_val_corr', ascending=False) #re-sorting the correlations by the abs_val

    all_features.to_csv(label_folder + f'/{label.upper()}ALL_TRANSFORMED_FEATURES.csv', index=False)
    all_correlations.to_csv(label_folder + f'/{label.upper()}_ALL_TRANSFORMED_FEATURE_CORRELATIONS.csv', index=False)

    return


''' 
For a given label, this takes all of the features in the df and calculates the pearson correlation with the label. 
- ranks the features by pearson correlation on the label
- gets rid of any features that have a poor correlation with the label
- gets rid of any redundant features based on the maximum_redundancy measurement
- returns the best features 
'''
def get_best_features_to_use(folder, label, all_labels, maximum_redundancy, minimum_corr_to_label):
    df = pd.read_csv(folder + f'/{label}/{label.upper()}ALL_TRANSFORMED_FEATURES.csv')
    corr_df = pd.read_csv(folder + f'/{label}/{label.upper()}_ALL_TRANSFORMED_FEATURE_CORRELATIONS.csv')
    label_df = df.copy()[all_labels]
    feature_df = df.copy().drop(columns=all_labels)
    
    '''first get rid of any features that have a correlation value with the label that is less than minimum_corr_to_label'''
    well_corr_df = corr_df.copy()[corr_df['Abs_val_corr'] >= minimum_corr_to_label]
    well_corr_features_df = feature_df.copy().loc[:, well_corr_df['Feature'].to_numpy()]

    best_features = []
    '''now get rid of any redundant features. Going in order from most correlated to the label to least'''
    for feature1 in well_corr_df['Feature']:
        best_features.append(feature1)
        if(not list(well_corr_df['Feature']).__contains__(feature1)): 
            continue
        # best_features.append(feature1)
        correlation_matrix = well_corr_features_df.corr(method='pearson')
        series_of_interest = correlation_matrix[feature1]
        
        #getting all non-redundant features to the feature of interest
        features_that_arent_redundant = series_of_interest[series_of_interest < maximum_redundancy].index.tolist()
        #add the best features so far to the features that arent redunant
        features_that_arent_redundant += list(set(best_features) - set(features_that_arent_redundant))


        #trimming the feature_dataframe to only have the non-redundant features
        well_corr_features_df[features_that_arent_redundant]
        #trimming the correaltion dataframe to only have the non-redundant features
        well_corr_df = well_corr_df[well_corr_df['Feature'].isin(features_that_arent_redundant)]

    print(f'\nThere are {len(well_corr_df)} non-redundant correlated features. :\n {well_corr_df[["Feature", "Correlation"]]}')
    
    # for feature in well_corr_df['Feature']:
    #     # if(feature == 'log(crack len)'): 
    #     #     print('here')
    #     plt.scatter(df[label], df[feature])
    #     plt.ylabel(feature)
    #     plt.xlabel(label)
    #     plt.show()
    well_corr_features_df = well_corr_features_df[well_corr_df['Feature'].to_numpy()]
    return well_corr_features_df, label_df


list_of_multiplying_features = ['dist btw frts', 'crack len', 'max thickness', 'max_kink', 'abs_val_sum_kink', 'avg_ori', 'angle_btw']
df = pd.read_csv(full_dataset_pathname, index_col=0)
cols = df.columns
feat_cols = cols.difference(all_labels)
list_of_multiplying_features = list(feat_cols[~feat_cols.str.contains('front')])



# saving_folder changes with the date
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
saving_folder = f'/Volumes/Jake_ssd/OCTOBER_DATASET/feature_transformations_{current_date}'

label = 'height'
# label = 'impact site x'
# label = 'impact site y'


get_feature_transformation(df, list_of_multiplying_features, 'square', label, saving_folder)
get_feature_transformation(df, list_of_multiplying_features, 'cube', label, saving_folder)
get_feature_transformation(df, list_of_multiplying_features, 'sqrt', label, saving_folder)
get_feature_transformation(df, list_of_multiplying_features, 'exp', label, saving_folder)
get_feature_transformation(df, list_of_multiplying_features, 'log', label, saving_folder)
get_feature_transformation(df, list_of_multiplying_features, 'nothin', label, saving_folder)

# saving_folder = '/Volumes/Jake_ssd/OCTOBER_DATASET/feature_interactions'

get_feature_interactions(df, list_of_multiplying_features, 'multiply_two_feats', label, saving_folder)
get_feature_interactions(df, list_of_multiplying_features, 'divide_two_feats', label, saving_folder)
get_feature_interactions(df, list_of_multiplying_features, 'exponential_two_feats', label, saving_folder)

# get_feature_interactions(df, list_of_multiplying_features, 'multiply_two_feats', 'impact site x', saving_folder)
# get_feature_interactions(df, list_of_multiplying_features, 'divide_two_feats', 'impact site x', saving_folder)
# get_feature_interactions(df, list_of_multiplying_features, 'exponential_two_feats', 'impact site x', saving_folder)


# get_feature_interactions(df, list_of_multiplying_features, 'multiply_two_feats', 'impact site y', saving_folder)
# get_feature_interactions(df, list_of_multiplying_features, 'divide_two_feats', 'impact site y', saving_folder)
# get_feature_interactions(df, list_of_multiplying_features, 'exponential_two_feats', 'impact site y', saving_folder)

put_everything_into_a_single_csv(saving_folder, 'height')
# put_everything_into_a_single_csv(saving_folder, 'impact site x')
# put_everything_into_a_single_csv(saving_folder, 'impact site y')

get_best_features_to_use(saving_folder, label, all_labels, 0.8, 0.25)




