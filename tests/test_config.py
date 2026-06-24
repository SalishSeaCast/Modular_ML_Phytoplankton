
"""
Testing for the proper loading of the configuration file. 
"""

from src.config import load_config

def test_load_config():

    config = load_config('configs/diat_pr.yaml')

    assert isinstance(config, dict) # the type we want. 

    assert 'name' in config['notebook'] # if it is included. 
