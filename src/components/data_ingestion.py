import os
import sys

from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    """Data Ingestion Configuration class to store the paths for data ingestion."""

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
