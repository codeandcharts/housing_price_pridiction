import sys
import os
import numpy as np
import pandas as pd
import pickle

from src.exception import CustomException
from src.logger import logging


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join(
            "data", "preprocessed", "preprocessor.pkl"
        )

    def predict(self, features):
        try:
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            logging.info("Preprocessing input features")
            data_scaled = preprocessor.transform(features)

            logging.info("Making prediction")
            prediction = model.predict(data_scaled)
            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        MSSubClass,
        LotFrontage,
        LotArea,
        OverallQual,
        OverallCond,
        YearBuilt,
        YearRemodAdd,
        MasVnrArea,
        BsmtFinSF1,
        BsmtFinSF2,
        BsmtUnfSF,
        TotalBsmtSF,
        FirstFlrSF,
        SecondFlrSF,
        LowQualFinSF,
        GrLivArea,
        BsmtFullBath,
        BsmtHalfBath,
        FullBath,
        HalfBath,
        BedroomAbvGr,
        KitchenAbvGr,
        TotRmsAbvGrd,
        Fireplaces,
        GarageYrBlt,
        GarageCars,
        GarageArea,
        WoodDeckSF,
        OpenPorchSF,
        EnclosedPorch,
        ThreeSsnPorch,
        ScreenPorch,
        PoolArea,
        MiscVal,
        MoSold,
        YrSold,
        LotShape,
        LandContour,
        Utilities,
        LandSlope,
        BsmtQual,
        BsmtExposure,
        BsmtFinType1,
        BsmtFinType2,
        BsmtCond,
        CentralAir,
        Electrical,
        ExterQual,
        ExterCond,
        FireplaceQu,
        Functional,
        GarageFinish,
        GarageQual,
        HeatingQC,
        KitchenQual,
        PavedDrive,
        Street,
        LotConfig,
        Neighborhood,
        Condition1,
        Condition2,
        BldgType,
        HouseStyle,
        RoofStyle,
        RoofMatl,
        Exterior1st,
        Exterior2nd,
        MasVnrType,
        Foundation,
        SaleType,
        SaleCondition,
        Heating,
        GarageType,
    ):
        self.MSSubClass = MSSubClass
        self.LotFrontage = LotFrontage
        self.LotArea = LotArea
        self.OverallQual = OverallQual
        self.OverallCond = OverallCond
        self.YearBuilt = YearBuilt
        self.YearRemodAdd = YearRemodAdd
        self.MasVnrArea = MasVnrArea
        self.BsmtFinSF1 = BsmtFinSF1
        self.BsmtFinSF2 = BsmtFinSF2
        self.BsmtUnfSF = BsmtUnfSF
        self.TotalBsmtSF = TotalBsmtSF
        self.FirstFlrSF = FirstFlrSF  # Matching column name in DF
        self.SecondFlrSF = SecondFlrSF  # Matching column name in DF
        self.LowQualFinSF = LowQualFinSF
        self.GrLivArea = GrLivArea
        self.BsmtFullBath = BsmtFullBath
        self.BsmtHalfBath = BsmtHalfBath
        self.FullBath = FullBath
        self.HalfBath = HalfBath
        self.BedroomAbvGr = BedroomAbvGr
        self.KitchenAbvGr = KitchenAbvGr
        self.TotRmsAbvGrd = TotRmsAbvGrd
        self.Fireplaces = Fireplaces
        self.GarageYrBlt = GarageYrBlt
        self.GarageCars = GarageCars
        self.GarageArea = GarageArea
        self.WoodDeckSF = WoodDeckSF
        self.OpenPorchSF = OpenPorchSF
        self.EnclosedPorch = EnclosedPorch
        self.ThreeSsnPorch = ThreeSsnPorch  # Matching column name in DF
        self.ScreenPorch = ScreenPorch
        self.PoolArea = PoolArea
        self.MiscVal = MiscVal
        self.MoSold = MoSold
        self.YrSold = YrSold
        self.LotShape = LotShape
        self.LandContour = LandContour
        self.Utilities = Utilities
        self.LandSlope = LandSlope
        self.BsmtQual = BsmtQual
        self.BsmtExposure = BsmtExposure
        self.BsmtFinType1 = BsmtFinType1
        self.BsmtFinType2 = BsmtFinType2
        self.BsmtCond = BsmtCond
        self.CentralAir = CentralAir
        self.Electrical = Electrical
        self.ExterQual = ExterQual
        self.ExterCond = ExterCond
        self.FireplaceQu = FireplaceQu
        self.Functional = Functional
        self.GarageFinish = GarageFinish
        self.GarageQual = GarageQual
        self.HeatingQC = HeatingQC
        self.KitchenQual = KitchenQual
        self.PavedDrive = PavedDrive
        self.Street = Street
        self.LotConfig = LotConfig
        self.Neighborhood = Neighborhood
        self.Condition1 = Condition1
        self.Condition2 = Condition2
        self.BldgType = BldgType
        self.HouseStyle = HouseStyle
        self.RoofStyle = RoofStyle
        self.RoofMatl = RoofMatl
        self.Exterior1st = Exterior1st
        self.Exterior2nd = Exterior2nd
        self.MasVnrType = MasVnrType
        self.Foundation = Foundation
        self.SaleType = SaleType
        self.SaleCondition = SaleCondition
        self.Heating = Heating
        self.GarageType = GarageType

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "MSSubClass": [self.MSSubClass],
                "LotFrontage": [self.LotFrontage],
                "LotArea": [self.LotArea],
                "OverallQual": [self.OverallQual],
                "OverallCond": [self.OverallCond],
                "YearBuilt": [self.YearBuilt],
                "YearRemodAdd": [self.YearRemodAdd],
                "MasVnrArea": [self.MasVnrArea],
                "BsmtFinSF1": [self.BsmtFinSF1],
                "BsmtFinSF2": [self.BsmtFinSF2],
                "BsmtUnfSF": [self.BsmtUnfSF],
                "TotalBsmtSF": [self.TotalBsmtSF],
                "1stFlrSF": [self.FirstFlrSF],
                "2ndFlrSF": [self.SecondFlrSF],
                "LowQualFinSF": [self.LowQualFinSF],
                "GrLivArea": [self.GrLivArea],
                "BsmtFullBath": [self.BsmtFullBath],
                "BsmtHalfBath": [self.BsmtHalfBath],
                "FullBath": [self.FullBath],
                "HalfBath": [self.HalfBath],
                "BedroomAbvGr": [self.BedroomAbvGr],
                "KitchenAbvGr": [self.KitchenAbvGr],
                "TotRmsAbvGrd": [self.TotRmsAbvGrd],
                "Fireplaces": [self.Fireplaces],
                "GarageYrBlt": [self.GarageYrBlt],
                "GarageCars": [self.GarageCars],
                "GarageArea": [self.GarageArea],
                "WoodDeckSF": [self.WoodDeckSF],
                "OpenPorchSF": [self.OpenPorchSF],
                "EnclosedPorch": [self.EnclosedPorch],
                "3SsnPorch": [self.ThreeSsnPorch],
                "ScreenPorch": [self.ScreenPorch],
                "PoolArea": [self.PoolArea],
                "MiscVal": [self.MiscVal],
                "MoSold": [self.MoSold],
                "YrSold": [self.YrSold],
                "LotShape": [self.LotShape],
                "LandContour": [self.LandContour],
                "Utilities": [self.Utilities],
                "LandSlope": [self.LandSlope],
                "BsmtQual": [self.BsmtQual],
                "BsmtExposure": [self.BsmtExposure],
                "BsmtFinType1": [self.BsmtFinType1],
                "BsmtFinType2": [self.BsmtFinType2],
                "BsmtCond": [self.BsmtCond],
                "CentralAir": [self.CentralAir],
                "Electrical": [self.Electrical],
                "ExterQual": [self.ExterQual],
                "ExterCond": [self.ExterCond],
                "FireplaceQu": [self.FireplaceQu],
                "Functional": [self.Functional],
                "GarageFinish": [self.GarageFinish],
                "GarageQual": [self.GarageQual],
                "HeatingQC": [self.HeatingQC],
                "KitchenQual": [self.KitchenQual],
                "PavedDrive": [self.PavedDrive],
                "Street": [self.Street],
                "LotConfig": [self.LotConfig],
                "Neighborhood": [self.Neighborhood],
                "Condition1": [self.Condition1],
                "Condition2": [self.Condition2],
                "BldgType": [self.BldgType],
                "HouseStyle": [self.HouseStyle],
                "RoofStyle": [self.RoofStyle],
                "RoofMatl": [self.RoofMatl],
                "Exterior1st": [self.Exterior1st],
                "Exterior2nd": [self.Exterior2nd],
                "MasVnrType": [self.MasVnrType],
                "Foundation": [self.Foundation],
                "SaleType": [self.SaleType],
                "SaleCondition": [self.SaleCondition],
                "Heating": [self.Heating],
                "GarageType": [self.GarageType],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame created from custom data")
            return df

        except Exception as e:
            raise CustomException(e, sys)
