import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

import warnings

warnings.filterwarnings("ignore", message=".*valid feature names.*")


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # 1) Define your models
            models = {
                "LinearReg": LinearRegression(),
                "RandomForest": RandomForestRegressor(),
                "XGBoost": XGBRegressor(),
                "Ridge": Ridge(),
                "GBR": GradientBoostingRegressor(),
                "LGBM": lgb.LGBMRegressor(verbose=-1),
                "CatBoost": CatBoostRegressor(verbose=False),
                "DecisionTree": DecisionTreeRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor(),
            }

            # 2) Define hyperparameter grids for each
            param_grid = {
                "LinearReg": {"fit_intercept": [True, False]},
                "RandomForest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                },
                "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
                "Ridge": {"alpha": [0.1, 1.0, 10.0]},
                "GBR": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
                "LGBM": {"n_estimators": [100, 200], "num_leaves": [31, 50]},
                "CatBoost": {"iterations": [100, 200], "depth": [4, 6]},
                "DecisionTree": {"max_depth": [None, 5, 10]},
                "AdaBoost": {"n_estimators": [50, 100]},
                "KNN": {"n_neighbors": [3, 5, 7]},
            }

            # 3) Evaluate them
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param_grid=param_grid,
            )

            # 4) Select the best
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(
                    f"No model met the performance threshold; best R² was {best_model_score:.3f}"
                )

            logging.info(f"Best model: {best_model_name} (R² = {best_model_score:.3f})")

            # 5) Save the preprocessor and the model
            #    (assuming save_object handles both .pkl paths appropriately)
            save_object(file_path=preprocessor_path, obj="YOUR_PREPROCESSOR_OBJECT")
            save_object(file_path=self.config.trained_model_path, obj=best_model)
            logging.info(
                f"Saved preprocessor to {preprocessor_path} and model to {self.config.trained_model_path}"
            )

            # 6) Final R² on test set
            preds = best_model.predict(X_test)
            r2 = r2_score(y_test, preds)

            return best_model_name, r2

        except Exception as e:
            raise CustomException(e, sys)
