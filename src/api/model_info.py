
"""
Model metadata helper functions.
"""

import yaml
from pathlib import Path

CONFIG_PATH = (Path(__file__).parents[2] / 'outputs' / 'model' / 'diat_pr.yaml')

def load_model_info():

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    drivers = config['notebook']['drivers']
    spatial = config['notebook']['spatial']
    day_input = config['notebook']['day_input']

    return {'drivers': drivers, 'spatial': spatial, 'day_input': day_input, 'n_features': len(drivers) + len(spatial) + len(day_input)}
