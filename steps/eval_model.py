from zenml import step
import logging
import pandas as pd
from sklearn.base import RegressorMixin

@step
def eval_model(model:RegressorMixin,data)->float:
    """
    This function tests and evaluates the model based on the provided test set
    Arguments : model , data (X_train,X_test,y_train,y_test)
    returns : accuracy  
    """
    model.predict()
    pass