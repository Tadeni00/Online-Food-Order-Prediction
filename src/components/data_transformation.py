from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
import joblib
import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.datatransformationconfig = DataTransformationConfig()

    def get_data_transformer_object(self, df):
        try:
            # Define numerical columns as before
            numerical_columns = ['Age', 'Monthly Income', 'Family size', 'latitude', 'longitude', 'Pin code']
            
            # Define categorical columns by dynamically finding one-hot encoded columns
            categorical_columns_prefixes = ['Gender', 'Marital Status', 'Occupation', 'Educational Qualifications', 'Feedback']
            categorical_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in categorical_columns_prefixes)]

            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', RobustScaler())  # Use RobustScaler for numerical data
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            logging.info(f'Numerical columns: {numerical_columns}')
            logging.info(f'Categorical columns: {categorical_columns}')

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', numerical_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessing object')
            preprocessor_obj = self.get_data_transformer_object(train_df)  # Pass train_df to get_data_transformer_object()

            target_column_name = 'Output'

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Encode target variable
            label_encoder = LabelEncoder()
            target_feature_train_df = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_df = label_encoder.transform(target_feature_test_df)

            logging.info("Applying preprocessing object on training and testing dataframes.")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Handling class imbalance using SMOTETomek.")
            smotetomek = SMOTETomek(random_state=42)
            input_feature_train_arr, target_feature_train_arr = smotetomek.fit_resample(input_feature_train_arr, target_feature_train_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_arr)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            self.save_object(
                file_path=self.datatransformationconfig.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.datatransformationconfig.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


    def save_object(self, file_path, obj):
        try:
            print(f"Saving object of type: {type(obj)}")
            with open(file_path, 'wb') as file_obj:
                joblib.dump(obj, file_obj)
        except Exception as e:
            raise CustomException(e, sys)
