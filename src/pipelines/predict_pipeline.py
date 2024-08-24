import os
import sys
import pandas as pd
import joblib
from src.utils import load_object
from src.exception import CustomException
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

model_path = os.path.join("artifacts", "model.pkl")
preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

@dataclass
class CustomData:
    gender: str
    marital_status: str
    occupation: str
    educational_qualifications: str
    feedback: str
    age: float
    monthly_income: float
    family_size: float
    latitude: float
    longitude: float
    pin_code: float

    def get_data_as_dataframe(self):
        data = {
            'Gender': [self.gender],
            'Marital Status': [self.marital_status],
            'Occupation': [self.occupation],
            'Educational Qualifications': [self.educational_qualifications],
            'Feedback': [self.feedback],
            'Age': [self.age],
            'Monthly Income': [self.monthly_income],
            'Family size': [self.family_size],
            'latitude': [self.latitude],
            'longitude': [self.longitude],
            'Pin code': [self.pin_code]
        }
        return pd.DataFrame(data)

class PredictionPipeline:
    def __init__(self):
        self.model = self.load_model()
        self.preprocessor = self.load_preprocessor()

    def load_model(self):
        try:
            model = joblib.load(model_path)
            if not isinstance(model, CatBoostClassifier):
                raise ValueError("Loaded object is not a CatBoostClassifie.")
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def load_preprocessor(self):
        try:
            preprocessor = joblib.load(preprocessor_path)
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)


    def predict(self, input_df):
        try:


            # Preprocess the data
            input_features = self.preprocessor.transform(input_df)

            # Make predictions
            predictions = self.model.predict(input_features)
            return predictions

        except Exception as e:
            raise CustomException(e, sys)

