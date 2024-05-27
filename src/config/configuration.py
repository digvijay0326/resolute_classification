from dataclasses import dataclass
import os
import sys

@dataclass
class DataIngestionConfig:
    data_path:str = os.path.join('artifacts','data','data.csv')
    train_data_path:str = os.path.join('artifacts','data','train.csv')
    test_data_path:str = os.path.join('artifacts','data','test.csv')
    evaluation_data_path:str = os.path.join('artifacts','data','evaluation.csv')
    
@dataclass
class DataTransformationConfig:
    preprocessor_path:str = os.path.join('artifacts','models','preprocessor.pkl')
