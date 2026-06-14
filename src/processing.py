
"""
Processing of the datasets.
"""

import numpy as np
from sklearn.feature_selection import r_regression

def dataset_preparation(dataset, name, input_names):

    """
    Preparing the datasets.

    Parameters:
        dataset(xr.Dataset): The selected dataset.
        name(str): The name of the target feature.
        lon_range (tuple(int, int)): The range of longitudes indices.

    Returns:
        inputs(np.array[float]): The input features.
        target(np.array[float]): The target feature.
        indx(np.array[int]): The indeces of non-nan points.
    """

    # Preparing x and y to be used as input features
    x = np.tile(dataset.x, len(dataset.time_counter)*len(dataset.y))
    y = np.tile(np.repeat(dataset.y, len(dataset.x)), len(dataset.time_counter))

    # Preparing the dayofyear to be used as input feature
    dayofyear = np.tile(np.arange(0,len(dataset.time_counter)//len(np.unique(dataset.time_counter.dt.year))), len(np.unique(dataset.time_counter.dt.year)))

    inputs = []

    if 'Day_of_year' in input_names: # Dayofyear requires different handling
        for i in input_names[0:input_names.index('Day_of_year')]: # Input features before Dayofyear
            inputs.append(dataset[i].to_numpy().flatten())  

        inputs.append(np.repeat(dayofyear, len(dataset.x)*len(dataset.y)))  # Dayofyear

        for i in input_names[input_names.index('Day_of_year')+1:]: # The rest
            inputs.append(dataset[i].to_numpy().flatten()) 

    else: # If no Dayofyear is included, treat all of them the same
        for i in input_names:
            inputs.append(dataset[i].to_numpy().flatten())

    inputs = np.array(inputs) # Converting it to a np.array

    targets = np.ravel(dataset[name]) # Making it flat

    indx = np.where(np.isfinite(targets) & (x>10) & ((x>100) | (y<880)))[0] # Masking based on nan values of the target feature and the edges of the map

    inputs = inputs[:,indx] # Applying the indeces
    targets = targets[indx] # Applying the indeces

    inputs = inputs.transpose() # Final transformation needed (n_samples, n_features)

    return {'inputs':inputs, 'targets':targets, 'indeces':indx}

def indeces(drivers, spatial, day_input, inputs_names):

    """
    Finding the proper indices for each category of input features. Used in the regressor.

    Parameters:
        drivers(list[str]): The selected atmospheric drivers.
        spatial(list[str]): The selected spatial features.
        day_input(list[str]): The day of the year.
        inputs_names(list[str]): All the names of input features combined.

    Returns:
        drivers_idx(np.array[int]): Indeces of drivers.
        spatial_idx(np.array[int]): Indeces of spatial features.
        day_idx(np.array[int]): Index of day of the year.
    """

    driver_idx = np.arange(len(drivers))

    if spatial:
        spatial_start = inputs_names.index(spatial[0])
        spatial_end = inputs_names.index(spatial[-1]) + 1
        spatial_idx = np.arange(spatial_start, spatial_end)
    else:
        spatial_idx = []
    
    if day_input:
        day_idx = [len(inputs_names)-1]
    else:
        day_idx = []

    return{'drivers':driver_idx, 'spatial':spatial_idx, 'day':day_idx}

def regr_inputs_targets(inputs, targets):

    """
    Calculating the correlation between each input feature and the targets.

    Parameters:
        inputs(np.array[float]): The input features.
        targets(np.array[float]): The target feature.

    Returns:
        r_regr(np.array[float]): The correlations between each input feature and the targets.
    """

    r_regr = np.round(r_regression(inputs,targets),2)

    return r_regr

def datasets_preparation2(variable, season, indeces, ds, name):

    """
    Preparing existing data for making xarray arrays (bringing them into the original format). Vice-versa from function datasets_preparation.

    Parameters:
        variable(np.array[float]): The variable we want to process (flat shape).
        season(np.array[float]): The long-term seasonality (only used for its length).
        indeces(np.array[int]): The indeces where we have values (only used for its length).
        ds(xr.DataSet): The original dataset.
        name(str): The name of the variable (only used for creating the default dimensions).

    Returns:
        variable_all(np.array[float]): The final array, with the proper dimensions, ready to be used to create an xarray DataArray.
    """

    # Re-shaping it to a proper format (years * days per year * ...).
    variable_mean = np.reshape(variable, (len(np.unique(ds.time_counter.dt.year)) * len(season), len(indeces) // len(ds.time_counter)))

    # Using this just to get the indx of valid values for one year.
    temp = np.reshape(np.ravel(ds[name]), (len(ds.time_counter), len(ds.y) * len(ds.x)))
    x =  np.tile(ds.x, len(ds.y))
    y =  np.tile(np.repeat(ds.y, len(ds.x)),1)

    indx = np.where((~np.isnan(temp).any(axis=0)) & (x>10) & ((x>100) | (y<880)))

    variable_all = np.full((len(ds.time_counter), len(ds.y) * len(ds.x)),np.nan) # Defining the variable (for all years).

    variable_all[:,indx[0]] = variable_mean # Filling it.
    variable_all = np.reshape(variable_all,(len(ds.time_counter),len(ds.y),len(ds.x))) # Reshaping it again (number of total days, length y, length x).

    return variable_all