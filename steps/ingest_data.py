from zenml import step
import logging
import pandas as pd

@step
def ingest_data(file_path) -> str:
    """
    This function takes the file path and ingests the data .
    Arguments : file_path
    returns : pd.dataframe
    """
    pass