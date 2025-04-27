import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.visualize_results import (
    plot_feature_importances,
    plot_shap_summary,
    get_feature_names_from_preprocessor,
)


@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = os.path.join("artifacts")
    plots_dir: str = os.path.join("artifacts", "plots")
    feature_importance_path: str = os.path.join(
        "artifacts", "plots", "feature_importance.png"
    )
    shap_summary_path: str = os.path.join("artifacts", "plots", "shap_summary.png")


class TrainingPipeline:
    def __init__(self):
        self.config = TrainingPipelineConfig()
        # Create necessary directories
        os.makedirs(self.config.artifacts_dir, exist_ok=True)
        os.makedirs(self.config.plots_dir, exist_ok=True)

    def start_training(self):
        try:
            logging.info("Starting the training pipeline")

            # 1) Data Ingestion
            logging.info("Step 1: Data Ingestion")
            ingestion_config = DataIngestionConfig()
            data_ingestion = DataIngestion(ingestion_config)
            train_df, test_df = data_ingestion.initiate_data_ingestion()

            # 2) Data Transformation
            logging.info("Step 2: Data Transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = (
                data_transformation.initiate_data_transformation(train_df, test_df)
            )

            # 3) Model Training
            logging.info("Step 3: Model Training")
            model_trainer = ModelTrainer()
            best_model_name, best_r2_score = model_trainer.initiate_model_trainer(
                train_arr, test_arr, preprocessor_path
            )

            # 4) Generate visualizations
            logging.info("Step 4: Generating Visualizations")
            self._generate_visualizations(train_arr, best_model_name, preprocessor_path)

            logging.info(
                f"Training pipeline completed successfully. Best model: {best_model_name} with R² score: {best_r2_score:.4f}"
            )

            return best_model_name, best_r2_score

        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)

    def _generate_visualizations(self, train_arr, best_model_name, preprocessor_path):
        """Generate feature importance and SHAP visualizations"""
        try:
            # Load the trained model
            from src.utils import load_object

            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            # Get preprocessor and extract feature names
            preprocessor = load_object(preprocessor_path)

            # Get feature names
            try:
                feature_names = get_feature_names_from_preprocessor(preprocessor)
                logging.info(
                    f"Extracted {len(feature_names)} feature names from preprocessor"
                )
            except Exception as e:
                logging.warning(f"Could not extract feature names: {str(e)}")
                # Fallback: create generic feature names
                feature_names = [f"feature_{i}" for i in range(train_arr.shape[1] - 1)]

            # Generate feature importance plot
            X_train = train_arr[:, :-1]  # All columns except the last (target)

            # 1) Feature Importance Plot
            try:
                plot_feature_importances(
                    model=model,
                    X=X_train,
                    feature_names=feature_names,
                    top_n=15,
                    save_path=self.config.feature_importance_path,
                )
                logging.info(
                    f"Feature importance plot saved to {self.config.feature_importance_path}"
                )
            except Exception as e:
                logging.warning(f"Failed to generate feature importance plot: {str(e)}")

            # 2) SHAP Summary Plot (for tree-based models)
            if hasattr(model, "feature_importances_"):
                try:
                    plot_shap_summary(
                        model=model,
                        X=X_train,
                        feature_names=feature_names,
                        sample_size=min(200, X_train.shape[0]),
                        save_path=self.config.shap_summary_path,
                    )
                    logging.info(
                        f"SHAP summary plot saved to {self.config.shap_summary_path}"
                    )
                except Exception as e:
                    logging.warning(f"Failed to generate SHAP summary plot: {str(e)}")
            else:
                logging.info(
                    f"Model {best_model_name} does not support SHAP visualization (not tree-based)"
                )

        except Exception as e:
            logging.warning(f"Error generating visualizations: {str(e)}")
            # Don't fail the pipeline due to visualization errors


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    best_model, r2_score = pipeline.start_training()
    print(f"✅ Training completed successfully!")
    print(f"✅ Best model: {best_model}")
    print(f"✅ R² score: {r2_score:.4f}")
