
"""
Data loading and selecting for phytoplankton analysis.
"""

import xarray as xr
import numpy as np
import pandas as pd

def load_dataset(path: str) -> xr.Dataset:

    """
    Loads the dataset from a NetCDF file.

    Parameters:
        path(str): Path to dataset.

    Returns:
        xr.Dataset: The initial dataset.
    """
    
    return xr.open_dataset(path)


def load_regions(ds, configs):
        
    """
    Define the regions for analysis.

    Parameters:
        ds(xr.Dataset): The original dataset.
        configs(yaml): Configurations file.

    Returns:
        region_names(list[str]): The names of the regions.
        regions_all(list[str]): List with the regions' coordinates.
        colors(list[str]): List the colors for each region.
        regions(xr.DataArray): Pointers indicating the regions (same shape as the original dataset).
    """

    # These values are pre-defined.
    SoG_north = configs['data']['SoG_north']
    SoG_center = configs['data']['SoG_center']
    Fraser_plume = configs['data']['Fraser_plume']
    SoG_south = configs['data']['SoG_south']
    Haro_Boundary = configs['data']['Haro_Boundary']
    JdF_west = configs['data']['JdF_west']
    JdF_east = configs['data']['JdF_east']
    PS_all = configs['data']['PS_all']
    PS_main = configs['data']['PS_main']

    region_names = configs['data']['region_names'] # used in the legend of the plot.

    regions_all = [SoG_north,SoG_center,Fraser_plume,SoG_south,Haro_Boundary,JdF_west,JdF_east,PS_all,PS_main] # including all regions in one list.
    colors = configs['data']['colors'] # including all colors in one list

    # Creating an xarray dataArray, which has pointers for each region.
    regions = np.full((len(ds.y),len(ds.x)),np.nan)
    for i in range (0, len(regions_all)):
        regions[regions_all[i][0]:regions_all[i][1], regions_all[i][2]:regions_all[i][3]] = i

    regions = xr.DataArray(regions,dims = ['y','x'])

    return {'names':region_names, 'corners':regions_all, 'colors':colors, 'mask':regions}

def sel_dataset(ds, start, end):

    """
    Select data from the original dataset.

    Parameters:
        ds(xr.Dataset): The initial dataset.
        start(str): Starting year.
        end(str): Ending year

    Returns:
        dataset(xr.Dataset): The sliced dataset.
        dates(pd.datetime): The dates of the dataset.
        labels(np.array[str]): Labels for the dates.
    """

    dataset = ds.sel(time_counter = slice(start, end))

    labels = np.unique(dataset.time_counter.dt.strftime('%d %b'))
    indx_labels = np.argsort(pd.to_datetime(labels, format='%d %b')) # Sorting.
    labels = labels[indx_labels]
    dates = pd.DatetimeIndex(dataset['time_counter'].values)

    return dataset, {'dates':dates, 'labels':labels}

def making_array(variable, ds, name, units):

    """
    Converitng an np.array to an xr.DataArray.

    Parameters:
        variable(np.array[float]): The variable we want to convert.
        ds(xr.Dataset): The dataset. Used for its dimensions.
        name(str): The name of the variable.
        units(str): The units of the variable.

    Returns:
        dataset(xr.Dataset): The new dataset.
    """
    
    # Preparation of the dataarray.
    dataset = xr.DataArray(variable, coords = {'time_counter': ds.time_counter,'y': ds.y, 'x': ds.x}, dims = ['time_counter','y','x'], attrs=dict(description=name, units=units))
    
    return dataset

def file_creation(path, variable, var_name, file_name):

    """
    Saving the Dataset.

    Parameters:
        path(str): The path to save the file.
        variable(xr.DataArray[float]): The variable we want to save.
        name(str): The name of the variable.

    Returns: 
        -
    """

    # Preparation of the dataset. 
    temp = variable.to_dataset(name = var_name)
    temp.to_netcdf(path = path + file_name, mode='a', encoding={var_name:{"zlib": True, "complevel": 9}})
    temp.close()    
