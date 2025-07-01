from zenml import step
import logging
import pandas as pd
from sklearn.base import RegressorMixin
from src.model_dev import LinearRegressionModel

@step
def train_model(    
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> RegressorMixin:
    """
    This function will train and return the model
    """
    model = LinearRegressionModel()
    model.train(X_train , y_train)
    return model
    