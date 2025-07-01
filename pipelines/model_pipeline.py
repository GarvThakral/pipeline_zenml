from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.eval_model import eval_model
from typing import Tuple
@pipeline
def model_pipeline(file_path):
    data = ingest_data(file_path)
    X_train , X_test , y_train , y_test  = clean_data(data)
    model = train_model(X_train , y_train)
    accuracy = eval_model(model , X_test , y_test)
