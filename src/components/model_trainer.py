import os 
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evalute_models
import numpy as np
import pandas as pd
from src.config.configuration import ModelTrainingConfig
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, 
                              AdaBoostClassifier,
                              GradientBoostingClassifier)
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, 
                             precision_score, 
                             recall_score, 
                             f1_score)

class ModelTraining:
    def __init__(self):
        self.config = ModelTrainingConfig()
    
    def initiate_model_training(self, train_arr, validation_arr, test_arr):
        try:
            logging.info("split trianing and test input data")
            x_train,y_train,x_valid, y_valid,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                validation_arr[:,:-1],
                validation_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models ={ 
            # "Decision Tree": DecisionTreeClassifier(),
            "Random Forest Classifier": RandomForestClassifier(),
            # "Gradient Boosting": GradientBoostingClassifier(),
            # "XGBClassifier": XGBClassifier(), 
            # "CatBoostClassifier": CatBoostClassifier(verbose=False),
            # "AdaBoostClassifier": AdaBoostClassifier(),
            # "KNN": KNeighborsClassifier(),
            # "SVM": SVC(),
            }
            param_grids = {
                # "Decision Tree": {
                #     'criterion': ['entropy'],
                #     'max_depth': [10],
                #     # 'min_samples_split': [2, 5, 10],
                # },
                "Random Forest Classifier": {
                    'n_estimators': [70],
                    'max_depth': [11],
                    # 'min_samples_split': [2, 5, 10],
                },
                # "Gradient Boosting": {
                #     'n_estimators': [100, 150, 200],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'max_depth': [3, 6, 9],
                # },
                # "XGBClassifier": {
                #     'n_estimators': [100, 150, 200],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'max_depth': [3, 6, 9],
                # },
                # "CatBoostClassifier": {
                #     'iterations': [100, 150, 200],
                #     'depth': [6, 8, 10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                # },
                # "AdaBoostClassifier": {
                #     'n_estimators': [50, 100, 200],
                #     'learning_rate': [0.01, 0.1, 1.0],
                # # },
                # "KNN": {
                #     'n_neighbors': [3],
                #     'weights': ['distance'],
                # },
                # "SVM": {
                #     'C': [10],
                #     'kernel': ['linear'],
                # },
            }
            
            train_model_report, valid_model_report = evalute_models(x_train=x_train,y_train=y_train,
                                             x_valid=x_valid,y_valid=y_valid,models=models,params=param_grids)
            
            ## best model score
            print(train_model_report)
            print(valid_model_report)
            best_model_score = max(sorted(valid_model_report.values()))

            ## best model name
            
            best_model_name = list(valid_model_report.keys())[
                list(valid_model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            
            # # if best_model_score<0.6:
            # #     raise CustomException("No best model found")
            # # logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.config.model_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            accuracy=accuracy_score(y_test,predicted)
            print(f"Accuracy of model is {accuracy}")
            return (accuracy, best_model_name)
        except Exception as e:
            raise CustomException(e, sys)        