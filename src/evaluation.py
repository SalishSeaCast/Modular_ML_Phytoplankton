
"""
Post-processing of the targets and predictions.
"""

import numpy as np
from sklearn.metrics import root_mean_squared_error as rmse

def post_processing(dataset, targets, predictions, indx):

    """
    Preparing targets and predictions for plotting.

    Parameters:
        dataset(xr.Dataset): The selected dataset.
        targets(np.array[float]): The target feature.
        predictions(np.array[float]): The predictions.
        indx(np.array[int]): The indices of the target feature.

    Returns:
        inputs(np.array[float]): The input features.
        target(np.array[float]): The target feature.
    """

    # Re-constructing the time-series (number of days x ...).
    targets_mean = np.reshape(targets,(len(dataset.time_counter), len(indx) // len(dataset.time_counter)))
    predictions_mean = np.reshape(predictions,(len(dataset.time_counter), len(indx) // len(dataset.time_counter)))

    # Taking the mean (one value per day).
    targets_mean = np.mean(targets_mean,axis=1)
    predictions_mean = np.mean(predictions_mean,axis=1)

    return {'targets':targets_mean, 'predictions':predictions_mean}

def metrics(targets,predictions):

    """
    Calculating the metrics for the evaluation comparison.

    Parameters:
        targets(np.array[float]): The target feature.
        predictions(np.array[float]): The predictions.

    Returns:
        r(float): The correlation coefficient.
        rms(float): The root mean square error.
        slope(float): The slope of the best fitting line.
    """

    r = np.round(np.corrcoef(predictions,targets)[0][1],3)

    with np.errstate(divide='ignore'): # For the division with zero warning.
        rms = np.round(rmse(predictions,targets)  / np.mean(targets) * 100, 3)

    m,_ = np.polyfit(targets, predictions, deg=1)
    slope = np.round(m,3)

    return {'r':r, 'rms':rms, 'slope':slope}

def post_processing_region(targets, predictions, region, season):

    """
    Preparing targets and predictions for plotting (per region). Runs in a loop.

    Parameters:
        targets(np.array[float]): The target feature of the current region.
        predictions(np.array[float]): The predictions of the current region.
        region(int): The corners of the current region. 
        season(np.array[float]): The seasonality of the current region.

    Returns:
        temp_data_season(dict): Dictionary contatining the processed targets and predictions for the the current region without seasonality.
        temp_data(dict): Dictionary contatining the processed targets and predictions for the the current region.
    """

    # Isolating each region.
    targets_temp = targets[:, region[0]:region[1], region[2]:region[3]]
    predictions_temp = predictions[:, region[0]:region[1], region[2]:region[3]]

    # Taking the mean (one value per day).
    targets_temp = np.nanmean(targets_temp, (1,2))
    predictions_temp = np.nanmean(predictions_temp, (1,2))

    temp_data_season = {'targets': targets_temp - season, 'predictions': predictions_temp - season} 
    temp_data = {'targets': targets_temp, 'predictions': predictions_temp} 

    return temp_data_season, temp_data
