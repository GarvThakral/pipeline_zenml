from zenml import step
import logging
import pandas as pd

@step
def ingest_data(file_path) -> pd.DataFrame:
    """
    This function takes the file path and ingests the data .
    Arguments : file_path
    returns : pd.dataframe
    """
    data = pd.read_csv(file_path)
    return data