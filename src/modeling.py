
"""
Modeling process for diatom production rate. Defining the regressor, training and testing.
"""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import HistGradientBoostingRegressor

class Diatom_pr_regressor:

    """
    The regressor part.
    """

    def __init__(self, n_bins, drivers_idx, spatial_idx, day_idx, model=None):

        """
        The constructor of the class object.

        Parameters:
            self(class instance): The regressor instance.
            categorical_cols(list[char]): Names of categorical columns.
            numerical_cols(list[char]): Names of numerical columns.
            model(scikit-learn object): Chosen model. If None, then a histogram gradient boosting regressor is used.
            random_state(int): Used for reproducible outputs.
        """

        self.n_bins = n_bins
        self.drivers_idx = drivers_idx
        self.spatial_idx = spatial_idx
        self.day_idx = day_idx

        # Obtaining the categorical features.
        if len(spatial_idx) > 0:
            if len(day_idx) > 0:
                categorical_features = np.concatenate([spatial_idx, day_idx])
            else:
                categorical_features = spatial_idx
        
        else:
            if len(day_idx) > 0:
                categorical_features = day_idx
            else:
                categorical_features = []

        # Obtaining the transformers.
        transformers = [('drivers', StandardScaler(), drivers_idx)]

        if len(spatial_idx) > 0:
            transformers.append(('spatial', KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', quantile_method='averaged_inverted_cdf'), spatial_idx))

        # The default model variation.
        if model is None:
            base_model = HistGradientBoostingRegressor(categorical_features=categorical_features)
        
            self.model = TransformedTargetRegressor(regressor=make_pipeline(ColumnTransformer(transformers=transformers, remainder='passthrough'), base_model), 
                transformer=StandardScaler())

        else:
            self.model = model

    def train(self, x_train, y_train):

        """
        Training the model.

        Parameters:
            self(class instance): The regressor instance.
            x_train(pd.DataFrame): Training input features.
            y_train(pd.DataFrame): Training targets.
        """

        self.model.fit(x_train, y_train)

    def test(self, x_test):

        """
        Testing the model.

        Parameters:
            self(class instance): The regressor instance.
            x_test(pd.DataFrame): Testing input features.

        Returns:
            predictions(numpy array): The predicted answer times.
        """

        predictions = self.model.predict(x_test)

        return predictions
