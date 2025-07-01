from zenml import step
import pandas as pd

@step
def clean_data(data:pd.DataFrame)->pd.DataFrame:
    """
    This function takes in a pandas DataFrame and cleans it 
    Arguments : data:pd.DataFrame
    returns : data :pd.DataFrame
    """
    pass