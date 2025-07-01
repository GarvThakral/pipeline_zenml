from abc import ABC , abstractmethod
import pandas as pd

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self,data:pd.DataFrame):
        pass

class DataPreProcessing(DataStrategy):
    """
    Class extending the data strategy and implementing the functions
    """
    def handle_data(self, data: pd.DataFrame):
        """
        Preprocess data
        """
        return 
    pass