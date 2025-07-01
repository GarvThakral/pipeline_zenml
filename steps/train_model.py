from zenml import step
import logging
import pandas as pd
from sklearn.base import RegressorMixin
from src.model_dev import LinearRegressionModel

@step
def train_model(    
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RegressorMixin:
    """
    This function will train and return the model
    """
    logging.info("Training the model")
    try:
        model = LinearRegressionModel()
        model = model.train(X_train , y_train)
        return model
    except Exception as e:
        logging.error(f"Error while training the mode {e}")
        raise