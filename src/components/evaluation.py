import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, save_object
from src.config.configuration import EvaluationConfig
class Evaluation:
    def __init__(self):
        self.config = EvaluationConfig()
    
    def evaluate(self):
        try:
            model_path = os.path.join('artifacts', 'models', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'models', 'preprocessor.pkl')
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            evaluation_df = pd.read_csv(self.config.evaluation_data_path)
            processed_df = preprocessor.transform(evaluation_df)
            # print(evaluation_df.columns())
            predictions = model.predict(processed_df)
            # print(predictions.shape)
            evaluation_df['predicted_class'] =  predictions
            evaluation_df.to_csv(self.config.prediction_path, index=False, header = True)
        except Exception as e:
            raise CustomException(e, sys)