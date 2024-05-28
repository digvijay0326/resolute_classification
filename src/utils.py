import os
import sys
import pandas as ps
import numpy as np
import dill
from src.exception import CustomException
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)

    
def evalute_models(x_train,y_train,x_valid,y_valid,models,params):
    try:
        report_valid = {}
        report_train = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            print(f"Training model:{model} started")
            gs = GridSearchCV(model,param,cv=2)
            gs.fit(x_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            print(f"Training model:{model} completed")
            y_train_pred = model.predict(x_train)

            y_valid_pred = model.predict(x_valid)

            train_model_score = accuracy_score(y_train, y_train_pred)
            valid_model_score = accuracy_score(y_valid, y_valid_pred)
            report_train[list(models.keys())[i]] = train_model_score
            report_valid[list(models.keys())[i]] = valid_model_score
        
        return report_train, report_valid
    except Exception as e:
        raise CustomException(e, sys)     
    