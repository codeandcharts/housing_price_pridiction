import os
import sys

import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:
    """Paths for raw train/test CSVs."""

    train_data_path: str = os.path.join("data", "raw", "train.csv")
    test_data_path: str = os.path.join("data", "raw", "test.csv")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion process.")
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)
            logging.info("Data ingestion completed successfully.")
            return train_df, test_df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # 1) Build config & ingest
    config = DataIngestionConfig()
    ingestion = DataIngestion(config)
    train_df, test_df = ingestion.initiate_data_ingestion()

    # 2) Transform in‐memory
    transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
        train_df=train_df, test_df=test_df
    )

    logging.info(f"Transformation complete. Preprocessor saved to {preprocessor_path}")
    print("Shapes:", train_arr.shape, test_arr.shape)
