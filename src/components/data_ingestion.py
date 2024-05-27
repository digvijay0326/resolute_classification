import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.config.configuration import DataIngestionConfig

class DataIngestion:
    def __init__(self):
        self.config=DataIngestionConfig()

    def load_data(self):
        try:
            logging.info("Loading train data")
            data_df = pd.read_excel("raw_data/train.xlsx")
            evaluation_df = pd.read_excel("raw_data/test.xlsx")
            os.makedirs(os.path.dirname(self.config.train_data_path),exist_ok=True)
            data_df.to_csv(self.config.data_path, index=False, header=True)
            train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)
            train_df.to_csv(self.config.train_data_path, index=False, header=True)
            test_df.to_csv(self.config.test_data_path, index=False, header=True)
            evaluation_df.to_csv(self.config.evaluation_data_path, index=False, header=True)
            logging.info("Data loaded successfully")
            
            return(
                self.config.train_data_path,
                self.config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

