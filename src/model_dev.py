from abc import ABC,abstractmethod
from sklearn.base import RegressorMixin
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
from sklearn.base import ClassifierMixin
from typing import Union
from sklearn.svm import SVC

class ModelDevelopment(ABC):
    """
    This method defines the abstract class which will be implemented for 
    every model going forward
    """
    @abstractmethod
    def train(self, X_train:pd.DataFrame , y_train:pd.Series)->Union[RegressorMixin , ClassifierMixin]:
        pass

class LinearRegressionModel(ModelDevelopment):
    """
    This class implements the ModelDevelopment class for a linear regression model
    """
    def train(self, X_train:pd.DataFrame , y_train:pd.Series)->RegressorMixin:
        """Train the linear regression model."""
        logging.info("Training Linear Regression model...")
        try:
            model = LinearRegression()
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise
class SVCClassifier(ModelDevelopment):
    def train(self , X_train:pd.DataFrame , y_train:pd.Series)->ClassifierMixin:
        """Train the SVC model."""
        logging.info("Training SVC model...")
        try:
            model = SVC()
            model = model.fit(X = X_train , y = y_train)
            return model
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise
    



