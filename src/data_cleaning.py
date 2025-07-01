from abc import ABC , abstractmethod
import pandas as pd
from typing_extensions import Union
from typing import Any
import logging
import numpy as np
from sklearn.model_selection import train_test_split
class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Any:
        pass

class DataPreProcessing(DataStrategy):
    """
    Class extending data strategy and preprocessing/cleaning the data
    """
    def handle_data(self, data: pd.DataFrame)->pd.DataFrame:
        """
        Preprocess data
        """
        try:
            logging.info("Cleaning data")
            data = data.drop(
                    [
                        "order_approved_at",
                        "order_delivered_carrier_date",
                        "order_delivered_customer_date",
                        "order_estimated_delivery_date",
                        "order_purchase_timestamp",
                    ],
                    axis=1
                )
            data["product_weight_g"].fillna(data["product_weight_g"].median(),inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(),inplace=True)   
            data["product_height_cm"].fillna(data["product_height_cm"].median(),inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(),inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = [
                "order_item_id",
                "customer_zip_code_prefix",
            ]
            data.drop(columns=cols_to_drop, inplace=True)
            return data
        except Exception as e:
            logging.error(f"Error cleaning data - {e}")
            raise
            
    

class DataDivideStrategy(DataStrategy):
    """
    Class extending data stratedy and dividing the data into trai/test sets
    """
    from typing import Tuple

    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] :
        """
        Divide the data into train/test
        """
        try:
            logging.info(f"Splitting the dataset into train and test")
            self.data = data
            X = data.iloc[:,:-1]
            y = data.iloc[:,-1] 
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )
            return X_train, X_test, y_train, y_test 
        except Exception as e:
            logging.error(f"Error splitting the dataset {e}")
            raise

class DataCleaning:
    def __init__(self, data:pd.DataFrame , strategy: DataStrategy):
        """
        Initialize DataCleaning with a specific strategy
        """
        self.data = data
        self.strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data using the specified strategy
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise   
