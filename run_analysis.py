
"""
Run analysis script that initiates the pipeline
"""

# Importing.

import numpy as np
import pandas as pd

from src import data, processing, utils, modeling, evaluation, config

configs = config.load_config('configs/diat_pr.yaml')

def main():

    """
    Entry point for reproducing the full ML pipeline.

    Returns:
        metrics(dictionary): The evaluation metrics.
    """

    print('\n')
    print('Pipeline started!')
    print('\n')

    # Initiation.

    name = configs['notebook']['name']
    filename = configs['notebook']['filename']
    units = configs['notebook']['units']

    # Selecting the input features.
    drivers = configs['notebook']['drivers']
    spatial = configs['notebook']['spatial']
    day_input = configs['notebook']['day_input']
    inputs_names = drivers + spatial + day_input

    # Loading the initial dataset.
    ds = data.load_dataset(filename)

    # Obtaining the names, corners, colors and mask for the regions. The mask variable is used for performing individual analysis in each region.
    region_features = data.load_regions(ds, configs)

    # Pre-training.

    # Obtaining the indeces of the input features.
    input_feature_indeces = processing.indeces(drivers, spatial, day_input, inputs_names)

    # Obtaining the train dataset and its features (date, labels).
    ds_train, ds_train_features = data.sel_dataset(ds, configs['notebook']['train_starting_year'], configs['notebook']['train_ending_year'])

    # Input features, target and indeces for train.
    pre_train_data = processing.dataset_preparation(ds_train, name, inputs_names)

    print('Pre-training done!')
    print('\n')

    ## Training.

    # Defining the model.
    model = modeling.Diatom_pr_regressor(n_bins = configs['notebook']['n_bins'], drivers_idx=input_feature_indeces['drivers'], 
        spatial_idx=input_feature_indeces['spatial'], day_idx=input_feature_indeces['day'])
    
    # Train.
    model.train(pre_train_data['inputs'], pre_train_data['targets'])

    # Returning the train predictions.
    predictions = model.test(pre_train_data['inputs'])

    # Returning the targets and predictions in an appropriate form (time-series).
    post_train_data = evaluation.post_processing(ds_train, pre_train_data['targets'], predictions, pre_train_data['indeces'])

    print('Training done!')
    print('\n')

    # Post-Training.

    # Returning the targets and predictions in an appropriate form (time-series).
    post_train_data = evaluation.post_processing(ds_train, pre_train_data['targets'], predictions, pre_train_data['indeces'])
    
    # Obtaining the correlation coefficient, root mean square error and slope of the best fitting line.
    train_metrics = evaluation.metrics(post_train_data['targets'], post_train_data['predictions'])

    # Printing the metrics for the whole domain.
    temp = pd.DataFrame.from_dict(data=train_metrics, orient='index', columns=['value'])
    print('Metrics for the whole domain (Training)')
    print(temp.transpose())
    print('\n')

    # Calculating the long-term seasonality and broadcasting it to all train years.
    season_features = utils.seasonality(post_train_data['targets'], ds_train_features['dates'], region_features, ds_train[name])

    # Calculating targets and predictions per region.
    targets_train_regional = processing.datasets_preparation2(pre_train_data['targets'], season_features['season'], pre_train_data['indeces'], ds_train, name)
    predictions_train_regional = processing.datasets_preparation2(predictions, season_features['season'], pre_train_data['indeces'], ds_train, name)
    
    print('Post-Training done!')
    print('\n')

    # Pre-Testing.

    # Obtaining the test dataset and its features (date, labels).
    ds_test, ds_test_features = data.sel_dataset(ds, configs['notebook']['test_starting_year'], configs['notebook']['test_ending_year'])

    # Input features, target and indeces for test.
    pre_test_data = processing.dataset_preparation(ds_test, name, inputs_names)

    print('Pre-Testing done!')
    print('\n')

    # Testing.

    # Returning the test predictions.
    predictions_test = model.test(pre_test_data['inputs'])

    # Post-Testing.

    # Returning the targets and predictions in an appropriate form (time-series).
    post_test_data = evaluation.post_processing(ds_test, pre_test_data['targets'], predictions_test, pre_test_data['indeces'])

    # Obtaining the correlation coefficient, root mean square error and slope of the best fitting line.
    test_metrics = evaluation.metrics(post_test_data['targets'], post_test_data['predictions'])

    # Printing the metrics for the whole domain.
    temp = pd.DataFrame.from_dict(data = test_metrics, orient='index', columns=['value'])
    print('Metrics for the whole domain (Testing)')
    print(temp.transpose())
    print('\n')

    # Calculating targets and predictions per region.
    targets_test_regional = processing.datasets_preparation2(pre_test_data['targets'], season_features['season'], pre_test_data['indeces'], ds_test, name)
    predictions_test_regional = processing.datasets_preparation2(predictions_test, season_features['season'], pre_test_data['indeces'], ds_test, name)

    # Creating final variables.

    # Targets.
    temp = np.concat([targets_train_regional, targets_test_regional])
    targets_all = data.making_array(temp, ds, name, units)

    # Predictions.
    temp = np.concat([predictions_train_regional, predictions_test_regional])
    predictions_all = data.making_array(temp, ds, name, units)
    
    print('Pipeline completed successfully!')
    print('\n')

    return (model.model, targets_all, predictions_all, train_metrics, test_metrics)

if __name__ == "__main__":
    main()