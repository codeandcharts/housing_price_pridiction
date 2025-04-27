# src/components/data_ingestion.py

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

# import downstream components
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from sklearn.metrics import mean_squared_error
from src.components.visualize_results import plot_feature_importances, plot_shap_summary


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("data", "raw", "train.csv")
    processed_train_path: str = os.path.join("data", "processed", "train.csv")
    processed_test_path: str = os.path.join("data", "processed", "test.csv")


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        try:
            logging.info(f"Reading raw data from {self.config.raw_data_path}")
            df = pd.read_csv(self.config.raw_data_path)

            logging.info("Splitting data into train/validation (80/20)")
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            # Ensure output dirs exist
            os.makedirs(
                os.path.dirname(self.config.processed_train_path), exist_ok=True
            )
            os.makedirs(os.path.dirname(self.config.processed_test_path), exist_ok=True)

            # Save splits
            train_df.to_csv(self.config.processed_train_path, index=False)
            test_df.to_csv(self.config.processed_test_path, index=False)

            logging.info(
                f"Saved processed train to {self.config.processed_train_path} (shape {train_df.shape})"
            )
            logging.info(
                f"Saved processed test  to {self.config.processed_test_path}  (shape {test_df.shape})"
            )

            return train_df, test_df

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # 1) Ingest & split
        cfg = DataIngestionConfig()
        ingestion = DataIngestion(cfg)
        train_df, test_df = ingestion.initiate_data_ingestion()
        print("✅ Train shape:", train_df.shape)
        print("✅ Test  shape:", test_df.shape)

        # 2) Transform
        transformer = DataTransformation()
        train_arr, test_arr, preprocessor_path = (
            transformer.initiate_data_transformation(train_df, test_df)
        )
        print("✅ Transformed train_arr shape:", train_arr.shape)
        print("✅ Transformed test_arr  shape:", test_arr.shape)

        # 3) Train & evaluate
        trainer = ModelTrainer()
        best_model_name, best_r2 = trainer.initiate_model_trainer(
            train_arr, test_arr, preprocessor_path
        )
        print(f"✅ Best model: {best_model_name}, R² on validation: {best_r2:.4f}")

    except Exception:
        logging.exception("Pipeline failed")
        sys.exit(1)
