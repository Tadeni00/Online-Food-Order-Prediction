import os
import sys
import dill
from src.exception import CustomException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def save_object(file_path, obj):
    '''
    Save an object to a file using dill.
    
    Parameters:
    - file_path (str): Path where the object will be saved.
    - obj: The object to be saved.
    
    Raises:
    - CustomException: If there's an error during the saving process.
    '''
    try:
        dir_path = os.path.dirname(file_path)
        
        # Ensure the directory exists
        os.makedirs(dir_path, exist_ok=True)
        
        # Save the object to a file
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    '''
    Evaluate multiple models and return their performance scores.
    
    Parameters:
    - X_train (array-like): Training feature data.
    - y_train (array-like): Training target data.
    - X_test (array-like): Testing feature data.
    - y_test (array-like): Testing target data.
    - models (dict): Dictionary of model names and model instances.
    
    Returns:
    - report (dict): Dictionary of model names and their performance scores on the test data.
    
    Raises:
    - CustomException: If there's an error during model evaluation.
    '''
    try:
        report = {}
        
        # Iterate over each model in the dictionary
        for name, model in models.items():
            model.fit(X_train, y_train)  # Train the model
            y_pred = model.predict(X_test)  # Predict on testing data
            
            # Calculate performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            
            # Store the performance metrics in the report dictionary
            report[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return report
    
    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)

def load_object(file_path):
    '''
    Load an object from a file using dill.
    
    Parameters:
    - file_path (str): Path to the file where the object is saved.
    
    Returns:
    - The object loaded from the file.
    
    Raises:
    - CustomException: If there's an error during loading.
    '''
    try:
        # Load the object from the pickle file
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
    
    except Exception as e:
        # Raise a custom exception if an error occurs
        raise CustomException(e, sys)

    print(f"Type of loaded object: {type(obj)}")
    return obj
