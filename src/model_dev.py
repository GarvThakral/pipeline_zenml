from abc import ABC,abstractmethod
from sklearn.base import RegressorMixin
import pandas as pd
import logging

class ModelDevelopment(ABC):
    """
    This method defines the abstract class which will be implemented for 
    every model going forward
    """
    @abstractmethod
    def train(self ,model:str , data:pd.DataFrame)->RegressorMixin:
        pass

class LinearRegressionModel(ModelDevelopment):
    """
    This class implements the ModelDevelopment class for a linear regression model
    """
    def train(self ,model:str , data:pd.DataFrame)->RegressorMixin:
        pass
    

