from zenml import step
import pandas as pd
from src.data_cleaning import DataCleaning , DataDivideStrategy , DataPreProcessing 
from typing import Tuple


@step
def clean_data(data:pd.DataFrame)->Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    """
    This function takes in a pandas DataFrame and cleans it 
    Arguments : data:pd.DataFrame
    returns : data :pd.DataFrame
    """
    clean_data = DataCleaning(data,DataPreProcessing()).handle_data()
    divide_data = DataCleaning(clean_data,DataDivideStrategy()) # type: ignore
    X_train , X_test , y_train , y_test = divide_data.handle_data()
    return X_train , X_test , y_train , y_test # type: ignore