import sys
import os
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


# from src.utils import save_object

@dataclass 
class DataIngestionConfig:
    train_data_path: str = os.path.join("Arifact", "train.csv")
    test_data_path: str = os.path.join("Arifact", "test.csv")
    raw_data_path: str = os.path.join("Arifact", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(os.path.join("Notebook", "data", "stud.csv"))
            logging.info("Read the dataset as a DataFrame")

            # Create directories if they do not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

             # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train-Test Split is initiated")

            # Split data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

             # Save the datasets to their respective paths
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info("Ingestion of data is completed!")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )



        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformaton = DataTransformation()
    
    train_arr, test_arr,_ = data_transformaton.initiate_data_transformation(train_data, test_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_modl_trainer(train_arr, test_arr))