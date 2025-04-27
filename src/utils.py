import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param_grid):
    """
    Trains each model in `models` using GridSearchCV over the corresponding
    hyperparameters in `param_grid`, then returns a dict of test-set R² scores.
    """
    try:
        report = {}

        for model_name, model in models.items():
            # grab the right hyper‐parameter dict
            grid = param_grid[model_name]

            # 1) find best params
            gs = GridSearchCV(model, grid, cv=3)
            gs.fit(X_train, y_train)

            # 2) set model to the best and retrain
            best_params = gs.best_params_
            model.set_params(**best_params)
            model.fit(X_train, y_train)

            # 3) evaluate
            y_test_pred = model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
