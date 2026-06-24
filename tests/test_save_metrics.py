
'''
Testing for the saving of outputs.
'''

import pandas as pd
from src.output import save_metrics

def test_save_metrics(tmp_path):

    # Creating some dummy data for the tests.
    metrics = {'r': 0.9, 'rms': 10, 'slope': 0.8}

    file_path = (tmp_path/'metrics.csv')

    save_metrics(file_path, metrics)

    saved_df = pd.read_csv(file_path)

    assert file_path.exists() # if the file exists.

    assert len(saved_df) == 3 # number of metrics.

    assert "key" in saved_df.columns # # if it is included.
    assert "value" in saved_df.columns # if it is included.
