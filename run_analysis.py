
"""
Run analysis script that initiates the pipeline
"""

# Importing.

import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath('..'))

from src import data, processing, utils, modeling, evaluation

def main():

    """
    Entry point for reproducing the full ML pipeline.

    Returns:
        metrics(dictionary): The evaluation metrics.
    """

    print('\n')

    # Initiation.

    name = 'Diatom_Production_Rate'
    units = '[mmol N / $m^2$ / $s$]'
    category = 'Production rates'

    filename = '/data/ibougoudis/MOAD/files/inputs/jan_mar.nc'

    # Selecting the input features.
    drivers = ['Summation_of_solar_radiation', 'Mean_precipitation', 'Mean_air_temperature', 'Mean_wind_speed']
    spatial = ['Latitude', 'Longitude']
    day_input = ['Day_of_year']
    inputs_names = drivers + spatial + day_input

    # Obtaining the period's features (period, id, months). Id is used for identification of the period during saving.
    period_features = utils.period_identify(filename)

    # Loading the initial dataset.
    ds = data.load_dataset(filename)

    # Obtaining the names, corners, colors and mask for the regions. The mask variable is used for performing individual analysis in each region.
    region_features = data.load_regions(ds)

    # Pre-training.

    # Obtaining the indeces of the input features.
    input_feature_indeces = processing.indeces(drivers, spatial, day_input, inputs_names)

    # Obtaining the train dataset and its features (date, labels).
    ds_train, ds_train_features = data.sel_dataset(ds, '2007', '2020')

    # Input features, target and indeces for train.
    pre_train_data = processing.dataset_preparation(ds_train, name, inputs_names)

    # Correlations between input features and target.
    r_inputs = processing.regr_inputs_targets(pre_train_data['inputs'], pre_train_data['targets'])

    # Printing the correlations as a dataframe.
    print('Metrics between input features and '+name)
    temp = pd.DataFrame(r_inputs, index=inputs_names, columns=[period_features['period']])
    print(temp)
    print('\n')

    print('Pre-training done!')
    print('\n')

    ## Training.

    # Defining the model.
    model = modeling.Diatom_pr_regressor(n_bins=255, drivers_idx=input_feature_indeces['drivers'], spatial_idx=input_feature_indeces['spatial'], day_idx=input_feature_indeces['day'])

    # Train.
    model.train(pre_train_data['inputs'], pre_train_data['targets'])

    # Returning the train predictions.
    predictions = model.test(pre_train_data['inputs'])

    # Returning the targets and predictions in an appropriate form (time-series).
    post_train_data = evaluation.post_processing(ds_train, pre_train_data['targets'], predictions, pre_train_data['indeces'])

    print('Training done!')
    print('\n')

    # Post-Training.
    
    # Obtaining the correlation coefficient, root mean square error and slope of the best fitting line.
    train_metrics = evaluation.metrics(post_train_data['targets'], post_train_data['predictions'])

    # Printing the metrics for the whole domain.
    temp = pd.DataFrame.from_dict(data=train_metrics, orient='index', columns=['value'])
    print('Metrics for the whole domain')
    print(temp.transpose())
    print('\n')

    # Calculating the long-term seasonality and broadcasting it to all train years.
    season_features = utils.seasonality(post_train_data['targets'], ds_train_features['dates'], region_features, ds_train[name])

    # Calculating the train time-series without seasonality.
    post_train_data_season = {'targets': post_train_data['targets']-season_features['season_broadcasted'], 
        'predictions': post_train_data['predictions']-season_features['season_broadcasted']}

    # Obtaining the correlation coefficient, root mean square error and slope of the best fitting line, without seasonality.
    train_metrics_season = evaluation.metrics(post_train_data_season['targets'], post_train_data_season['predictions'])
    # The rmse must be kept the same as before!
    train_metrics_season['rms'] = train_metrics['rms']

    # Printing the metrics for the whole domain without seasonality.
    temp = pd.DataFrame.from_dict(data=train_metrics_season, orient='index', columns=['value'])
    print('Metrics for the whole domain (without seasonality)')
    print(temp.transpose())
    print('\n')

    # Post-Training analysis per region.

    # Calculating targets and predictions per region.
    targets_train_regional = processing.datasets_preparation2(pre_train_data['targets'], season_features['season'], pre_train_data['indeces'], ds_train, name)
    predictions_train_regional = processing.datasets_preparation2(predictions, season_features['season'], pre_train_data['indeces'], ds_train, name)

    train_metrics_region = []
    train_metrics_season_region = []

    # For each region.
    for i in range (len(region_features['names'])):

        # Returning 2 dictionaries, with and without seasonality.
        post_train_data_season_region, post_train_data_region = evaluation.post_processing_region(targets_train_regional, predictions_train_regional, 
            region_features['corners'][i], season_features['season_regional_broadcasted'][i])
        
        # Metrics per region.
        train_metrics_region.append(evaluation.metrics(post_train_data_region['targets'], post_train_data_region['predictions']))

        # Metrics per region without seasonality.
        train_metrics_season_region.append(evaluation.metrics(post_train_data_season_region['targets'], post_train_data_season_region['predictions']))

        # The rmse must be kept the same as before!
        train_metrics_season_region[i]['rms'] = train_metrics_region[i]['rms']

        # Printing the metrics without seasonality per region.
        temp = pd.DataFrame.from_dict(data=train_metrics_season_region[i], orient='index', columns=['value'])
        print('Metrics for ' + region_features['names'][i] + ' (without seasonality)')
        print(temp.transpose())
        print('\n')
    
    print('Post-Training done!')
    print('\n')

    # Pre-Testing.

    # Obtaining the test dataset and its features (date, labels).
    ds_test, ds_test_features = data.sel_dataset(ds, '2021', '2025')

    # Input features, target and indeces for test.
    pre_test_data = processing.dataset_preparation(ds_test, name, inputs_names)

    print('Pre-Testing done!')
    print('\n')

    # Testing.

    # Returning the test predictions.
    predictions_test = model.test(pre_test_data['inputs'])

    # Returning the targets and predictions in an appropriate form (time-series).
    post_test_data = evaluation.post_processing(ds_test, pre_test_data['targets'], predictions_test, pre_test_data['indeces'])

    # Post-Testing.

    # Obtaining the correlation coefficient, root mean square error and slope of the best fitting line.
    test_metrics = evaluation.metrics(post_test_data['targets'], post_test_data['predictions'])

    # Calculating the test time-series without seasonality.
    post_test_data_season = {'targets': post_test_data['targets']-season_features['season_broadcasted'][0:len(post_test_data['targets'])], 
        'predictions': post_test_data['predictions']-season_features['season_broadcasted'][0:len(post_test_data['targets'])]}

    # Obtaining the correlation coefficient, root mean square error and slope of the best fitting line, without seasonality.
    test_metrics_season = evaluation.metrics(post_test_data_season['targets'], post_test_data_season['predictions'])
    # The rmse must be kept the same as before!
    test_metrics_season['rms'] = test_metrics['rms']

    # Printing the metrics for the whole domain without seasonality.
    temp = pd.DataFrame.from_dict(data = test_metrics_season, orient='index', columns=['value'])
    print('Metrics for the whole domain (without seasonality)')
    print(temp.transpose())
    print('\n')

    # Post-Testing analysis per region.

    # Calculating targets and predictions per region.
    targets_test_regional = processing.datasets_preparation2(pre_test_data['targets'], season_features['season'], pre_test_data['indeces'], ds_test, name)
    predictions_test_regional = processing.datasets_preparation2(predictions_test, season_features['season'], pre_test_data['indeces'], ds_test, name)

    test_metrics_region = []
    test_metrics_season_region = []

    # For each region.
    for i in range (len(region_features['names'])):

        # Returning 2 dictionaries, with and without seasonality.
        post_test_data_season_region, post_test_data_region = evaluation.post_processing_region(targets_test_regional, predictions_test_regional, 
            region_features['corners'][i], season_features['season_regional_broadcasted'][i][0:len(post_test_data['targets'])])
        
        # Metrics per region.
        test_metrics_region.append(evaluation.metrics(post_test_data_region['targets'], post_test_data_region['predictions']))

        # Metrics per region without seasonality.
        test_metrics_season_region.append(evaluation.metrics(post_test_data_season_region['targets'], post_test_data_season_region['predictions']))

        # The rmse must be kept the same as before!
        test_metrics_season_region[i]['rms'] = test_metrics_region[i]['rms']

        temp = pd.DataFrame.from_dict(data=test_metrics_season_region[i], orient='index', columns=['value'])
        print('Metrics for ' + region_features['names'][i] + ' (without seasonality)')
        print(temp.transpose())
        print('\n')

    print('Post-Testing done!')
    print('\n')

    # Creating final variables.

    # Targets.
    temp = np.concat([targets_train_regional, targets_test_regional])
    targets_all = data.making_array(temp, ds, name, units)

    # Predictions.
    temp = np.concat([predictions_train_regional, predictions_test_regional])
    predictions_all = data.making_array(temp, ds, name, units)
    
    print('Pipeline completed successfully!')
    print('\n')

    return (targets_all, predictions_all)

if __name__ == "__main__":
    main()