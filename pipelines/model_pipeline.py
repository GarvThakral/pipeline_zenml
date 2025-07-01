from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.eval_model import eval_model

@pipeline
def model_pipeline(file_path):
    data = ingest_data(file_path)
    cleaned_data = clean_data(data)
    model = train_model(cleaned_data)
    accuracy = eval_model(model)