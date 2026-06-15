
"""
Helper for the configurations file.
"""

import yaml

def load_config(path):

    """
    Load configuration file.

    Parameters:
        path(str): Path to yaml file.

    Returns:
        config(dict): Configuration dictionary.
    """

    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return config