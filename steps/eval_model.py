from zenml import step
import logging
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error  # Use MSE for regression
from zenml.materializers import BuiltInMaterializer  # Import appropriate materializer
from sklearn.base import BaseEstimator
from typing import Any

@step
def eval_model(model: Any, X_test:pd.DataFrame, y_test:pd.Series) -> float:
    """
    This function tests and evaluates the model based on the provided test set.
    
    Arguments:
        model: Trained LinearRegression model
        data: Tuple containing (X_test: pd.DataFrame, y_test: pd.Series)
    
    Returns:
        float: Mean squared error of the model
    """
    logging.info("Evaluating the model performance")
    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)  # Use MSE instead of accuracy
        logging.info(f"The model mean squared error is {mse}")
        return float(mse)
    except Exception as e:
        logging.error(f"Error while evaluating model: {e}")
        raise

