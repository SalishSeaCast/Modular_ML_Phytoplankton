
"""
Data saving for phytoplankton analysis.
"""

import os
import dill
import lzma
import pandas as pd

def save_metrics(path, metrics):

    """
    Saves the selected metrics.

    Parameters:
        path(str): The selected path.
        metrics(dict): Dictionary with the metrics.
        category(str): The grouped category of the metrics.

    Returns:
        -
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.DataFrame(metrics.items(), columns=['key', 'value'])
    df.to_csv(path, index=False)

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

def save_model(path, configs, model):

    """
    Saving the regressor.

    Parameters:
        path(str): The path to save the file.
        configs(yaml): Configurations file.
        model(obj): The regressor object.

    Returns: 
        -
    """

    with lzma.open(path + configs['notebook']['regressor'], 'wb') as f:   
        dill.dump(model.model, f)
