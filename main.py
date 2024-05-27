from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__ == "__main__":
    data_ingestion=DataIngestion()
    train_path, test_path = data_ingestion.load_data()
    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_path, test_path)
    