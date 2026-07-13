
'''
Testing for the proper use of the model.
'''

import numpy as np
from src.modeling import Diatom_pr_regressor
from src import output

def test_model_coef():

    # Creating dummy fake inputs

    X = np.random.rand(10, 5)
    y = np.random.rand(10)

    model = Diatom_pr_regressor(n_bins=5, drivers_idx=[0, 1], spatial_idx=[2, 3], day_idx=[4])

    model.train(X, y)
    predictions = model.model.predict(X)

    assert model.model is not None # if the model exists.
    assert len(predictions) == len(y) # if the produced output has the proper length. 

def test_save_load_model(tmp_path):

    # Creating dummy fake inputs
    X = np.random.rand(10, 5)
    y = np.random.rand(10)

    # Creating the model and training it.
    model = Diatom_pr_regressor(n_bins=5, drivers_idx=[0, 1], spatial_idx=[2, 3], day_idx=[4])
    model.train(X, y)
    predictions = model.model.predict(X)

    # Saving the model.
    output.save_model(str(tmp_path), 'tmp_model.xz', model)

    # Loading the model.
    loaded_model = output.load_model(str(tmp_path), 'tmp_model.xz')

    loaded_predictions = loaded_model.model.predict(X)

    assert np.array_equal(predictions, loaded_predictions) # if the predictions of the 2 models are the same.

def test_save_load_model_dep(tmp_path):

    # Creating dummy fake inputs
    X = np.random.rand(10, 5)
    y = np.random.rand(10)

    # Creating the model and training it.
    model = Diatom_pr_regressor(n_bins=5, drivers_idx=[0, 1], spatial_idx=[2, 3], day_idx=[4])
    model.train(X, y)
    predictions = model.model.predict(X)

    # Saving the model.
    output.save_api_model(str(tmp_path), 'tmp_model.joblib', model)

    # Loading the model.
    loaded_model = output.load_api_model(str(tmp_path), 'tmp_model.joblib')

    loaded_predictions = loaded_model.predict(X)

    assert np.array_equal(predictions, loaded_predictions) # if the predictions of the 2 models are the same.

    




