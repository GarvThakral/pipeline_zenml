from zenml import step
import logging
import pandas as pd
from sklearn.base import RegressorMixin

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
    pass
    