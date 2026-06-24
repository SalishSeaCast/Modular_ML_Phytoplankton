
'''
Testing for the proper use of the model.
'''

import numpy as np
from src.modeling import Diatom_pr_regressor

def test_model_coef():

    # Creating dummy fake inputs

    X = np.random.rand(10, 5)
    y = np.random.rand(10)

    model = Diatom_pr_regressor(n_bins=5, drivers_idx=[0, 1], spatial_idx=[2, 3], day_idx=[4])

    model.train(X, y)
    predictions = model.model.predict(X)

    assert model.model is not None # if the model exists.
    assert len(predictions) == len(y) # if the produced output has the proper length. 

