import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            
            # Unpacking the training and testing arrays into features and target variables
            X_train = train_array[:, :-1]  # Features for training
            y_train = train_array[:, -1]   # Target variable for training
            X_test = test_array[:, :-1]    # Features for testing
            y_test = test_array[:, -1]     # Target variable for testing

            logging.info("Defining models and preprocessing steps")
            
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Bagging": BaggingClassifier(),
                "SVC": SVC(probability=True),
                "K-Neighbors": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                # "XGBoost": XGBClassifier(),
                "CatBoost": CatBoostClassifier(verbose=False),
                "LightGBM": LGBMClassifier(),
                "LDA": LinearDiscriminantAnalysis(),
                "MLP": MLPClassifier(max_iter=500, learning_rate_init=0.001)
            }

            best_models = {}
            best_f1_scores = {}
            best_accuracies = {}
            best_precisions = {}
            best_recalls = {}

            logging.info("Training and evaluating each model")

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary')  # Adjust if necessary
                recall = recall_score(y_test, y_pred, average='binary')        # Adjust if necessary
                f1 = f1_score(y_test, y_pred, average='binary')                # Adjust if necessary

                logging.info(f"Model performance for {model_name}")
                logging.info(f"- Accuracy: {accuracy:.4f}")
                logging.info(f"- Precision: {precision:.4f}")
                logging.info(f"- Recall: {recall:.4f}")
                logging.info(f"- F1 Score: {f1:.4f}")
                logging.info(f"- Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

                best_models[model_name] = model
                best_f1_scores[model_name] = f1
                best_accuracies[model_name] = accuracy
                best_precisions[model_name] = precision
                best_recalls[model_name] = recall

            best_model_score = max(best_f1_scores.values())
            best_model_name = list(best_f1_scores.keys())[list(best_f1_scores.values()).index(best_model_score)]
            best_model = best_models[best_model_name]
            best_model_accuracy = max(best_accuracies.values())
            best_model_precision = max(best_precisions.values())
            best_model_recall = max(best_recalls.values())

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient score")

            # logging.info(f"Best model found: {best_model_name} with F1 score {best_model_score:.4f}")
            logging.info(f"Best model: {best_model_name}")
            logging.info(f"Accuracy: {best_model_accuracy:.4f}")
            logging.info(f"Precision: {best_model_precision:.4f}")
            logging.info(f"Recall: {best_model_recall:.4f}")
            logging.info(f"F1 score: {best_model_score:.4f}")

            logging.info(f"Saving model to {self.model_trainer_config.trained_model_file_path}")
            try:
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )
                logging.info("Model saved successfully.")
            except Exception as e:
                logging.error("Failed to save the model.")
                raise CustomException(e, sys)

            return best_model, best_model_score, best_model_accuracy, best_model_precision, best_model_recall

        except ValueError as e:
            logging.error("ValueError encountered while training the model")
            raise CustomException(e, sys)
        except Exception as e:
            logging.error("Exception encountered while training the model")
            raise CustomException(e, sys)

