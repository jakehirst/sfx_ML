from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sdv.evaluation.single_table import evaluate_quality
from Backward_feature_selection import *
from sklearn.metrics import r2_score
import plotly.graph_objs as go
from plotly.subplots import make_subplots



'''trains a CTGAN based on the train_df provided. the batch sizer and num_epochs are customizable.'''
def train_CTGAN(train_df, batch_size, num_epochs):
    #metadata should define the pandas dataframe of the data you want to use to generate new synthetic data.
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=train_df)

    '''showing what metadata does. It evaluates the type of data for each row in the pandas dataframe. This can be edited if it doesnt get it correct the first time.'''
    # print(metadata.to_dict()) 
    
    synthesizer = CTGANSynthesizer(
    metadata, # required
    enforce_rounding=False,
    epochs=num_epochs,
    verbose=True, #Print out the Generator and Discriminator loss values per epoch. The loss values indicate how well the GAN is currently performing, lower values indicating higher quality.
    enforce_min_max_values = True, #If true, the synthetic data will contain values within the ranges of the real data
    cuda= True, #should speed up training time if cuda is available.
    batch_size = batch_size #must be even and dividsible by pac, which pac defaults at 10
    )

    synthesizer.fit(train_df)
    return synthesizer

'''saves the metrics of the synthetic data'''
def save_quality_metrics(train_df, synthetic_data, path):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=train_df)
    
    quality_report = evaluate_quality(
    real_data=train_df,
    synthetic_data=synthetic_data,
    metadata=metadata)

    '''
    Column shapes compares the gaussian curve fits of the synthetic vs the real datasets
    
    Column Pair Trends compares the pearson correlations between all columns in the synthetic vs 
    real datasets. How similar are the correlation matricies in the real and synthetic datasets?
    '''
    quality_df = quality_report.get_properties()
    quality_df.to_csv(path)
    # print(quality_report.get_details(property_name='Column Shapes'))
    # print(quality_report.get_details(property_name='Column Pair Trends'))
    # print('save the quality metrics here')
    return

'''saves the CTGAN to a pickle file example: desktop/my_synthesizer.pkl'''
def save_CTGAN(synthesizer, path):
    synthesizer.save(
    filepath=path
    )
    return

'''loads a CTGAN from a pickle file example: desktop/my_synthesizer.pkl'''
def load_CTGAN(path):
    synthesizer = CTGANSynthesizer.load(
        filepath=path
    )
    return synthesizer
'''
Uses the inputted synthesizer to make the same number of synthetic datapoints as there are training datapoints
Then fits each model type to both the real training dataset, and the synthetic dataset.
Then records both the R2 of the synthetic and real training datasets to the desired path in a .csv
'''
def analyze_R2_performance(synthesizer, train_df, test_df, path, 
                           all_labels=['impact site x', 'impact site y', 'height'], 
                           model_types = ['linear', 'RF', 'lasso', 'ridge', 'poly2', 'GPR'], 
                           features=None):
    if(features == None):
        features = ['init z',
            'init y',
            'init x',
            'max_prop_speed',
            'avg_prop_speed',
            'dist btw frts',
            'crack len',
            'linearity',
            'max thickness',
            'mean thickness',
            'median_thickness',
            'var_thickness',
            'std_thickness',
            'thickness_at_init',
            'max_kink',
            'abs_val_mean_kink',
            'mean_kink',
            'sum_kink',
            'abs_val_sum_kink',
            'median_kink',
            'std_kink',
            'var_kink',
            'avg_ori',
            'angle_btw']
    
    if(not os.path.exists(path)): os.makedirs(path)
    
    for label in all_labels:
        saving_path  = path + f'/{label}'
        print(f'\n***label being predicted = {label}***')
        synthetic_data = synthesizer.sample(num_rows=len(train_df))
        synthetic_training_features = synthetic_data[features]
        synthetic_training_labels = synthetic_data[label]
        true_training_features = train_df[features]
        true_training_labels = train_df[label]
        test_features = test_df[features]
        test_labels = test_df[label]

        performances = {}
        for model_type in model_types:
            print(f'\nmodel type: {model_type}')
            synthetic_model = train_model(model_type, synthetic_training_features, synthetic_training_labels)
            synthetic_model.fit(synthetic_training_features, synthetic_training_labels)
            preds = synthetic_model.predict(test_features)
            synthetic_r2 = r2_score(test_labels, preds)
            print(f'synthetic data model test r2 = {synthetic_r2}')
            
            true_model = train_model(model_type, true_training_features, true_training_labels)
            true_model.fit(true_training_features, true_training_labels)
            preds = true_model.predict(test_features)
            real_r2 = r2_score(test_labels, preds)
            print(f'real data model test r2 = {real_r2}')
            
            performances[model_type] = {'real_data test r2':real_r2, 'synthetic_data test r2':synthetic_r2}
            
            df = pd.DataFrame.from_dict(performances, orient='index')
            df.to_csv(saving_path + '.csv')

            
        # print('need to save performances as a .csv here')
    return


'''plots the CTGAN's discriminator and generator losses over the total epochs'''
def Plot_CTGAN_losses(synthesizer, metadata, train_df, synthetic_data):
    loss_values = synthesizer._model.loss_values
    # Plot loss function
    fig = go.Figure(data=[go.Scatter(x=loss_values['Epoch'], y=loss_values['Generator Loss'], name='Generator Loss'),
                        go.Scatter(x=loss_values['Epoch'], y=loss_values['Discriminator Loss'], name='Discriminator Loss')])


    # Update the layout for best viewing
    fig.update_layout(template='plotly_white',
                        legend_orientation="h",
                        legend=dict(x=0, y=1.1))

    title = 'CTGAN loss functions'
    fig.update_layout(title=title, xaxis_title='Epoch', yaxis_title='Loss')
    fig.show()
    return


'''plots the confusion matricies and the data quality for each of the columns in the training data as they relate to the synthetic data.'''
def Plot_CTGAN_quality_reports(train_df, synthetic_data):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=train_df)
    
    quality_report = evaluate_quality(
    real_data=train_df,
    synthetic_data=synthetic_data,
    metadata=metadata)

    quality_report.get_properties()
    # print(quality_report.get_details(property_name='Column Shapes'))
    # print(quality_report.get_details(property_name='Column Pair Trends'))
    fig = quality_report.get_visualization(property_name='Column Shapes')
    fig.show()
    fig = quality_report.get_visualization(property_name='Column Pair Trends')
    fig.show()
    return

'''a helper function to get the quality metric of choice. used for plotting mostly.'''
def get_quality_metric(metric, path):
    metrics = pd.read_csv(path + '/quality_metrics.csv')
    if(metric == 'Column Shapes'):
        metric_val = metrics.loc[metrics['Property'] == 'Column Shapes']['Score'].iloc[0]
    elif(metric == 'Column Pair Trends'):
        metric_val = metrics.loc[metrics['Property'] == 'Column Pair Trends']['Score'].iloc[0]
    elif(metric == 'Quality metric sum'):
        metric_val = metrics['Score'].sum()
    return metric_val



def make_performance_figures(label, epochs_batchsizes_scores, model_symbols):
    # Create subplots: one for real scores and one for abs(difference)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Synthetic Data Test R2', 'Synthetic - Real) Test R2\n(literature says this should always be negative)'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]  # Specify 3D subplots
    )
    # Create subplots
    # fig = make_subplots(rows=1, cols=2, subplot_titles=("d values", "abs(c-d) values"))

    # Loop through each model type and add a scatter plot to the subplots
    for model_type, tuples in epochs_batchsizes_scores.items():
        x, y, c, d = zip(*tuples)
        d_minus_c = [di - ci for ci, di in zip(c, d)]
        
        symbol, color = model_symbols[model_type]
        
        # Left subplot for d values
        fig.add_trace(
            go.Scatter3d(x=x, y=y, z=d, mode='markers', name=model_type,
                       marker=dict(symbol=symbol, color=color, size=10)),
                    # marker=dict(symbol=symbol, color=color, size=10, line=dict(width=2, color='DarkSlateGrey'))),
            row=1, col=1
        )
        
        # Right subplot for abs(c-d) values
        fig.add_trace(
            go.Scatter3d(x=x, y=y, z=d_minus_c, mode='markers', name=model_type,
                       marker=dict(symbol=symbol, color=color, size=10),
                       showlegend=False),
                    # marker=dict(symbol=symbol, color=color, size=10, line=dict(width=2, color='DarkSlateGrey'))),
            row=1, col=2
        )
            
    fig.update_layout(height=600, width=1200, title_text=f"Comparison of Model Performance predicting {label}",
                      scene1=dict(zaxis=dict(title='Test R2', range=[-1, 1]),
                                  xaxis=dict(title='epoch'),
                                  yaxis=dict(title='batch size')),
                      scene2=dict(zaxis=dict(title='Difference in test R2'),
                                  xaxis=dict(title='epoch'),
                                  yaxis=dict(title='batch size')))
    
    save_path = f'/Volumes/Jake_ssd/GANS/hyperparameter_tuning/R2_performances'
    if(not os.path.exists(save_path)): os.makedirs(save_path)
    fig.write_html(save_path + f'/{label}.html')
    return