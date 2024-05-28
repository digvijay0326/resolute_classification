from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTraining
from src.components.evaluation import Evaluation

if __name__ == "__main__":
    data_ingestion=DataIngestion()
    train_path, validation_path, test_path = data_ingestion.load_data()
    data_transformation = DataTransformation()
    train_arr, validation_arr, test_arr = data_transformation.initiate_data_transformation(train_path,
                                                                               validation_path,
                                                                                           test_path)
    model_training = ModelTraining()
    # model_training.initiate_model_training(train_arr, validation_arr, test_arr)
    evaluation = Evaluation()
    evaluation.evaluate()
    # accuracy, best_model = model_training.initiate_model_training(train_arr, validation_arr, test_arr)
    
    