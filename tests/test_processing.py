
'''
Testing for the datasets preparation of the processing file. 
'''

import numpy as np
import pandas as pd
import xarray as xr

from src.processing import dataset_preparation
from src.processing import indices

def test_dataset_preparation_shapes():

    # Creating some dummy data for the tests.
    times = pd.date_range('2020-01-01', periods=2)

    x = np.arange(20)
    y = np.arange(20)

    ds = xr.Dataset({'feature1': (('time_counter', 'y', 'x'), np.random.rand(2, 20, 20)), 
        'feature2': (('time_counter', 'y', 'x'),np.random.rand(2, 20, 20)), 
        'target': (('time_counter', 'y', 'x'), np.random.rand(2, 20, 20))},

        coords={'time_counter': times, 'x': x, 'y': y})

    input_names=['feature1', 'feature2']

    results = dataset_preparation(ds, 'target', input_names)

    assert isinstance(results, dict) # the type we want. 

    assert 'inputs' in results # if it is included.
    assert 'targets' in results # if it is included.
    assert 'indices' in results # if it is included.

    assert results['inputs'].shape[0] == len(results['targets']) # Proper shapes for training.
    assert results["inputs"].shape[1] == len(input_names) # Proper shapes for training.
    
    assert len(results['targets']) == len(results['indices']) # Verifying that NaNs are removed correctly. 

def test_indices():

    # Creating some dummy data for the tests.
    drivers = ['feature1', 'feature2']
    spatial = ['lat', 'lon']
    day_input = ['day']
    inputs_names = drivers + spatial + day_input

    results = indices(drivers, spatial, day_input, inputs_names)

    assert isinstance(results, dict) # the type we want. 

    assert len(results['drivers']) == len(drivers) # if the lengths are the same.
    assert len(results['spatial']) == len(spatial) # if the lengths are the same.
    assert len(results['day']) == len(day_input) # if the lengths are the same.
