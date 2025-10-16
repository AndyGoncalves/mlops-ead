import pandas as pd
import pytest
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from train import (reset_seeds,
                    read_data,
                    process_data,
                    create_model,
                    config_mlflow,
                    train_model)

#from train import (read_data,
                   #create_model,
                   #train_model)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

@pytest.fixture
def sample_data():
    """
    A fixture function that returns a sample dataset.

    Returns:
        pandas.DataFrame: A DataFrame containing sample data with three columns: 'feature1',
         'feature2', and 'fetal_health'.
    """
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [6, 7, 8, 9, 10],
        'fetal_health': [1, 1, 2, 3, 2]
    })
    return data


def test_read_data():
    """
    This function tests the `read_data` function. It checks whether the returned data is not
     empty for both features (X) and labels (y).

    Parameters:
    None

    Returns:
    None
    """
    X, y = read_data()

    assert not X.empty
    assert not y.empty


def test_create_model():
    """
    Generate the function comment for the given function body in a markdown code block with
    the correct language syntax.
    """
    X, _ = read_data()
    model = create_model(X)

    assert len(model.layers) > 2
    assert model.trainable
    assert isinstance(model, Sequential)


def test_train_model(sample_data):
    """
    Generate a function comment for the given function body in a markdown code block with
    the correct language syntax.

    Parameters:
        sample_data (pandas.DataFrame): The input data containing features and target
        variable.

    Returns:
        None
    """
    X = sample_data.drop(['fetal_health'], axis=1)
    y = sample_data['fetal_health'] - 1
    model = create_model(X)
    train_model(model, X, y, is_train=False)
    assert model.history.history['loss'][-1] > 0
    assert model.history.history['val_loss'][-1] > 0
