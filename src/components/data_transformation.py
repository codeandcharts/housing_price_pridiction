import sys
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "data", "preprocessed", "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = [
                "Id",
                "MSSubClass",
                "LotFrontage",
                "LotArea",
                "OverallQual",
                "OverallCond",
                "YearBuilt",
                "YearRemodAdd",
                "MasVnrArea",
                "BsmtFinSF1",
                "BsmtFinSF2",
                "BsmtUnfSF",
                "TotalBsmtSF",
                "1stFlrSF",
                "2ndFlrSF",
                "LowQualFinSF",
                "GrLivArea",
                "BsmtFullBath",
                "BsmtHalfBath",
                "FullBath",
                "HalfBath",
                "BedroomAbvGr",
                "KitchenAbvGr",
                "TotRmsAbvGrd",
                "Fireplaces",
                "GarageYrBlt",
                "GarageCars",
                "GarageArea",
                "WoodDeckSF",
                "OpenPorchSF",
                "EnclosedPorch",
                "3SsnPorch",
                "ScreenPorch",
                "PoolArea",
                "MiscVal",
                "MoSold",
                "YrSold",
            ]

            ode_cols = [
                "LotShape",
                "LandContour",
                "Utilities",
                "LandSlope",
                "BsmtQual",
                "BsmtFinType1",
                "CentralAir",
                "Functional",
                "FireplaceQu",
                "GarageFinish",
                "GarageQual",
                "PavedDrive",
                "ExterCond",
                "KitchenQual",
                "BsmtExposure",
                "HeatingQC",
                "ExterQual",
                "BsmtCond",
            ]

            ohe_cols = [
                "Street",
                "LotConfig",
                "Neighborhood",
                "Condition1",
                "Condition2",
                "BldgType",
                "HouseStyle",
                "RoofStyle",
                "Exterior1st",
                "Exterior2nd",
                "MasVnrType",
                "Foundation",
                "Electrical",
                "SaleType",
                "MSZoning",
                "SaleCondition",
                "Heating",
                "GarageType",
                "RoofMatl",
            ]

            num_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )
            ode_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "ordinal_encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                ]
            )
            ohe_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot_encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )

            col_trans = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("ode_pipeline", ode_pipeline, ode_cols),
                    ("ohe_pipeline", ohe_pipeline, ohe_cols),
                ],
                remainder="passthrough",
                n_jobs=-1,
            )

            return Pipeline([("preprocessing", col_trans)])

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """
        Takes train/test DataFrames, fits the preprocessor on train,
        transforms both, and returns numpy arrays + preprocessor path.
        """
        try:
            logging.info("Starting data transformation.")

            preprocessor = self.get_data_transformer_object()
            target_col = "SalePrice"

            # split out X/y
            X_train = train_df.drop(columns=[target_col], axis=1)
            y_train = train_df[target_col].values

            # drop target if present, ignore if not
            X_test = test_df.drop(columns=[target_col], axis=1, errors="ignore")
            y_test = (
                test_df[target_col].values if target_col in test_df.columns else None
            )

            logging.info("Fitting preprocessor on training data.")
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # reassemble arrays
            train_arr = np.c_[X_train_arr, y_train]
            test_arr = np.c_[X_test_arr, y_test] if y_test is not None else X_test_arr

            # save preprocessor object
            save_object(
                file_path=self.config.preprocessor_obj_file_path, obj=preprocessor
            )
            logging.info(
                f"Preprocessor saved at {self.config.preprocessor_obj_file_path}"
            )

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
