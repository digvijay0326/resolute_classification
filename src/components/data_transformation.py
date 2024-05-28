import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import DataTransformationConfig
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  
from scipy.stats import zscore
from src.utils import save_object

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
    
    def get_transform_object(self):
        pipeline = Pipeline(
            steps=[
                ('scaler', StandardScaler(with_mean=False)),
                ('pca', PCA(n_components=10))
            ]
        )
        
        return pipeline
    def initiate_data_transformation(self, train_data_path:str,
                                     validation_data_path:str, test_data_path:str):
        try:
            logging.info("Transforming data")
            train_data = pd.read_csv(train_data_path)
            validation_data = pd.read_csv(validation_data_path)
            test_data = pd.read_csv(test_data_path)
            
            target_feature_name = 'target'
            input_train_df = train_data.drop(columns=[target_feature_name], axis=1)
            target_feature_train_df = train_data[target_feature_name]
            input_validation_df = validation_data.drop(columns=[target_feature_name], axis=1)
            target_feature_validation_df = validation_data[target_feature_name]
            input_test_df = test_data.drop(columns=[target_feature_name], axis=1)
            target_feature_test_df = test_data[target_feature_name]
            
            preprocessing_object = self.get_transform_object()
            input_feature_train_arr = preprocessing_object.fit_transform(input_train_df)
            input_feature_validation_arr = preprocessing_object.fit_transform(input_validation_df)
            input_feature_test_arr = preprocessing_object.fit_transform(input_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            validation_arr = np.c_[input_feature_validation_arr, np.array(target_feature_validation_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(self.config.preprocessor_path, preprocessing_object)
            
            return(train_arr, validation_arr, test_arr)
        
            
        except Exception as e:
            raise CustomException(e, sys)