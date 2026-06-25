
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

    os.makedirs(path, exist_ok=True)

    file_path = (path + file_name)

    # Preparation of the dataset. 
    temp = variable.to_dataset(name = var_name)
    temp.to_netcdf(path = file_path, mode='a', encoding={var_name:{"zlib": True, "complevel": 9}})
    temp.close()  

def save_model(path, name, model):

    """
    Saving the regressor.

    Parameters:
        path(str): The path to save the file.
        configs(yaml): Configurations file.
        model(obj): The regressor object.

    Returns: 
        -
    """

    os.makedirs(path, exist_ok=True)

    file_path = (path + name)

    with lzma.open(file_path, 'wb') as f:   
        dill.dump(model, f)
    
def load_model(path, name):

    """
    Loading the regressor.

    Parameters:
        path(str): The path to load the file.
        configs(yaml): Configurations file.

    Returns: 
        model(obj): The regressor object.
    """

    file_path = (path + name)

    with lzma.open(file_path, 'rb') as f:
        model = dill.load(f)

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
