import pandas as pd
import os
from src.exception import CustomException
from src.logger import logging
import sys

class DataValidation:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def validate_file_existence(self):
        """
        Ensure that the files exist.
        """
        try:
            if not os.path.exists(self.train_path):
                raise FileNotFoundError(f"Training file not found at path: {self.train_path}")
            if not os.path.exists(self.test_path):
                raise FileNotFoundError(f"Test file not found at path: {self.test_path}")
            logging.info(f"Files found. Train: {self.train_path}, Test: {self.test_path}")
        except Exception as e:
            raise CustomException(e, sys)

    def validate_data_format(self, df):
        """
        Ensure that the dataframe is in the correct format.
        This includes checks for missing columns, data types, and invalid values.
        """
        try:
            # Expected columns and their data types
            expected_columns = {
                'Age': 'int64',
                'Gender': 'object',
                'Marital Status': 'object',
                'Occupation': 'object',
                'Monthly Income': 'int64',
                'Educational Qualifications': 'object',
                'Family size': 'int64',
                'latitude': 'float64',
                'longitude': 'float64',
                'Pin code': 'int64',
                'Feedback': 'object'
            }
            
            # Check for missing columns
            missing_columns = [col for col in expected_columns.keys() if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")
            
            # Check for data type consistency
            for col, dtype in expected_columns.items():
                if df[col].dtype != dtype:
                    raise ValueError(f"Column {col} has incorrect data type. Expected {dtype}, got {df[col].dtype}")

            logging.info("Data format validation passed.")
        except Exception as e:
            raise CustomException(e, sys)

    def validate_missing_values(self, df):
        """
        Check for missing values in critical columns.
        """
        try:
            # Critical columns that must not have missing values
            critical_columns = ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income',
                                'Family size', 'latitude', 'longitude', 'Pin code']
            missing_data = df[critical_columns].isnull().sum()

            if missing_data.any():
                raise ValueError(f"Missing values found in critical columns: {missing_data[missing_data > 0]}")
            
            logging.info("No missing values in critical columns.")
        except Exception as e:
            raise CustomException(e, sys)

    def validate_business_logic(self, df):
        """
        Validate data against business logic rules.
        """
        try:
            # Example business rule: Age should be between 18 and 35
            if not df['Age'].between(18, 35).all():
                raise ValueError("Some entries in 'Age' column are outside the expected range (18-35).")
            
            # Example business rule: Monthly Income should not be negative
            if (df['Monthly Income'] < 0).any():
                raise ValueError("Some entries in 'Monthly Income' column have negative values.")
            
            logging.info("Business logic validation passed.")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self):
        """
        Perform the full data validation process on both training and test datasets.
        """
        try:
            self.validate_file_existence()
            
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            
            logging.info("Starting validation for the training dataset.")
            self.validate_data_format(train_df)
            self.validate_missing_values(train_df)
            self.validate_business_logic(train_df)
            
            logging.info("Starting validation for the testing dataset.")
            self.validate_data_format(test_df)
            self.validate_missing_values(test_df)
            self.validate_business_logic(test_df)
            
            logging.info("Data validation completed successfully for both training and testing datasets.")
        except Exception as e:
            raise CustomException(e, sys)
