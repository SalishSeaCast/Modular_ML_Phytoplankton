
"""
Run analysis script that initiates the pipeline
"""

# Importing.

import numpy as np
import pandas as pd
import logging
import argparse
import os

from src import data, processing, utils, modeling, evaluation, config, output, visualization

os.makedirs('outputs/logs/', exist_ok=True)

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[logging.FileHandler('outputs/logs/pipeline.log'), logging.StreamHandler()]) 

logger = logging.getLogger(__name__)

def main(configs):

    """
    Entry point for reproducing the full ML pipeline.

    Returns:
        metrics(dictionary): The evaluation metrics.
    """

    logger.info('Pipeline started')

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

    # Obtaining the indices of the input features.
    input_feature_indices = processing.indices(drivers, spatial, day_input, inputs_names)

    # Obtaining the train dataset and its features (date, labels).
    ds_train, ds_train_features = data.sel_dataset(ds, configs['notebook']['train_starting_year'], configs['notebook']['train_ending_year'])

    # Input features, target and indices for train.
    pre_train_data = processing.dataset_preparation(ds_train, name, inputs_names)

    logger.info('Pre-training done!')

    ## Training.

    # Defining the model.
    model = modeling.Diatom_pr_regressor(n_bins = configs['notebook']['n_bins'], drivers_idx=input_feature_indices['drivers'], 
        spatial_idx=input_feature_indices['spatial'], day_idx=input_feature_indices['day'])
    
    # Train.
    model.train(pre_train_data['inputs'], pre_train_data['targets'])

    # Returning the train predictions.
    predictions = model.test(pre_train_data['inputs'])

    # Returning the targets and predictions in an appropriate form (time-series).
    post_train_data = evaluation.post_processing(ds_train, pre_train_data['targets'], predictions, pre_train_data['indices'])

    logger.info('Training done!')

    # Post-Training.

    # Returning the targets and predictions in an appropriate form (time-series).
    post_train_data = evaluation.post_processing(ds_train, pre_train_data['targets'], predictions, pre_train_data['indices'])
    
    # Obtaining the correlation coefficient, root mean square error and slope of the best fitting line.
    train_metrics = evaluation.metrics(post_train_data['targets'], post_train_data['predictions'])

    # Printing the metrics for the whole domain.
    temp = pd.DataFrame.from_dict(data=train_metrics, orient='index', columns=['value'])
    print('Metrics for the whole domain (Training)')
    print(temp.transpose())
    print('\n')

    # Calculating the long-term seasonality and broadcasting it to all train years.
    season_features = utils.seasonality(post_train_data['targets'], ds_train_features['dates'], region_features, ds_train[name])

    fig = visualization.plotting_seasonality(season_features['season'], ds_train_features['labels'])
    output.save_figure(fig, configs['notebook']['path'] + 'outputs/figures/', 'Long-term seasonality (2007-2020)')

    # Calculating targets and predictions per region.
    targets_train_regional = processing.datasets_preparation2(pre_train_data['targets'], season_features['season'], pre_train_data['indices'], ds_train, name)
    predictions_train_regional = processing.datasets_preparation2(predictions, season_features['season'], pre_train_data['indices'], ds_train, name)
    
    logger.info('Post-Training done!')

    # Pre-Testing.

    # Obtaining the test dataset and its features (date, labels).
    ds_test, ds_test_features = data.sel_dataset(ds, configs['notebook']['test_starting_year'], configs['notebook']['test_ending_year'])

    # Input features, target and indices for test.
    pre_test_data = processing.dataset_preparation(ds_test, name, inputs_names)

    logger.info('Pre-Testing done!')

    # Testing.

    # Returning the test predictions.
    predictions_test = model.test(pre_test_data['inputs'])

    # Post-Testing.

    # Returning the targets and predictions in an appropriate form (time-series).
    post_test_data = evaluation.post_processing(ds_test, pre_test_data['targets'], predictions_test, pre_test_data['indices'])

    # Obtaining the correlation coefficient, root mean square error and slope of the best fitting line.
    test_metrics = evaluation.metrics(post_test_data['targets'], post_test_data['predictions'])

    # Printing the metrics for the whole domain.
    temp = pd.DataFrame.from_dict(data = test_metrics, orient='index', columns=['value'])
    print('Metrics for the whole domain (Testing)')
    print(temp.transpose())
    print('\n')

    # Calculating targets and predictions per region.
    targets_test_regional = processing.datasets_preparation2(pre_test_data['targets'], season_features['season'], pre_test_data['indices'], ds_test, name)
    predictions_test_regional = processing.datasets_preparation2(predictions_test, season_features['season'], pre_test_data['indices'], ds_test, name)

    # Creating final variables.

    # Targets.
    temp = np.concat([targets_train_regional, targets_test_regional])
    targets_all = data.making_array(temp, ds, name, units)

    # Predictions.
    temp = np.concat([predictions_train_regional, predictions_test_regional])
    predictions_all = data.making_array(temp, ds, name, units)
    
    logger.info('Pipeline completed successfully!')

    return {'model': model, 'targets': targets_all, 'predictions': predictions_all, 'train_metrics': train_metrics, 'test_metrics': test_metrics}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Run phytoplankton ML pipeline') # Creating the parser.

    parser.add_argument('--config', required = True, help = 'Path to YAML configuration file') # Adding the config argument.
    parser.add_argument('--save_metrics', action='store_true', help = 'Save metrics') # Adding the save metrics argument.
    parser.add_argument('--save_model', action='store_true', help = 'Save the model') # Adding the save model argument.
    parser.add_argument('--verbose', action = 'store_true', help = 'Enable detailed logging') # Adding the verbose argument.

    args = parser.parse_args() # Reading from the command line.
    configs = config.load_config(args.config) # Loading the typed config file.

    logger.info(f'Using configuration: {args.config}')
    results = main(configs)

    if args.save_metrics:
        output.save_metrics('outputs/metrics/train_metrics.csv', results['train_metrics'])

    if args.save_model:

        # Research.
        output.save_model('outputs/model/', configs['notebook']['regressor'], results['model'])

        # Deployment.
        output.save_api_model('outputs/model/', configs['notebook']['regressor_dep'], results['model'])
    
        # Configuration file.
        output.save_config(args.config, 'outputs/model/')

    if args.verbose:
        logger.setLevel(logging.DEBUG)