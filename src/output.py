
"""
Data saving for phytoplankton analysis.
"""

import os
import lzma
import pandas as pd
import joblib
import shutil

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

    os.makedirs(path, exist_ok=True)

    file_path = (path + file_name)

    # Preparation of the dataset. 
    temp = variable.to_dataset(name = var_name)
    temp.to_netcdf(path = file_path, mode='a', encoding={var_name:{"zlib": True, "complevel": 9}})
    temp.close()  

def save_model(path, name, model):

    """
    Saving the regressor (reasearch).

    Parameters:
        path(str): The path to save the file.
        name(str): The name of the regressor (from the configuration file).
        model(obj): The regressor object (full).

    Returns: 
        -
    """

    os.makedirs(path, exist_ok=True)

    file_path = (path + name)

    with lzma.open(file_path, 'wb') as f:   
        joblib.dump(model, f)
    
def load_model(path, name):

    """
    Loading the regressor (research).

    Parameters:
        path(str): The path to load the file.
        name(str): The name of the regressor (from the configuration file).

    Returns: 
        model(obj): The regressor object (full).
    """

    file_path = (path + name)

    with lzma.open(file_path, 'rb') as f:
        model = joblib.load(f)

    return model

def save_figure(fig, path, name, dpi=300):

    """
    Saving a figure.

    Parameters:
        fig(obj): The figure object.
        path(str): The path to load the file.
        name(str): The name of the figure.
        dpi(int): The quality of the figure.


    Returns: 
        -
    """

    os.makedirs(path, exist_ok=True)

    file_path = (path + name)

    fig.savefig(file_path, dpi=dpi, bbox_inches = 'tight')

def save_api_model(path, name, model):

    """
    Saving the regressor (deployment).

    Parameters:
        path(str): The path to save the file.
        name(str): The name of the regressor (from the configuration file).
        model(obj): The regressor object (only the pipeline).

    Returns: 
        -
    """

    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, name)

    # save ONLY sklearn pipeline
    joblib.dump(model.model, file_path)


def load_api_model(path, name):

    """
    Loading the regressor (deployment).

    Parameters:
        path(str): The path to load the file.
        name(str): The name of the regressor (from the configuration file).

    Returns: 
        model(obj): The regressor object (only the pipeline).
    """

    file_path = os.path.join(path, name)

    return joblib.load(file_path)

def save_config(source_path, destination_path):

    """
    Saving the config file.

    Parameters:
        source_path(str): The original path of the configuration file.
        destination_path(str): The new path for the configuration file.

    Returns: 
        -
    """

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    shutil.copy2(source_path, destination_path)




