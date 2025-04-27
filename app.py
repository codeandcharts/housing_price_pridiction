import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle
from PIL import Image

# Add the root directory to the path for imports
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.utils import load_object
from src.logger import logging
from src.exception import CustomException

# Style constants
BACKGROUND_COLOR_LIGHT = "#ffffff"
TEXT_COLOR_LIGHT = "#111111"
BACKGROUND_COLOR_DARK = "#111111"
TEXT_COLOR_DARK = "#ffffff"

# Set page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load model options for display
@st.cache_resource
def get_model_info():
    try:
        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join("data", "preprocessed", "preprocessor.pkl")

        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)

        model_name = type(model).__name__
        return model_name, model, preprocessor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return "Unknown Model", None, None


# Feature input groups - ONLY TOP 5 FEATURES based on importance
def get_feature_options():
    """
    Returns the possible values for categorical features
    and default/min/max for numerical features
    """
    options = {
        # Top categorical features - only one included
        "Neighborhood": [
            "Blmngtn",
            "Blueste",
            "BrDale",
            "BrkSide",
            "ClearCr",
            "CollgCr",
            "Crawfor",
            "Edwards",
            "Gilbert",
            "IDOTRR",
            "MeadowV",
            "Mitchel",
            "NoRidge",
            "NPkVill",
            "NridgHt",
            "NWAmes",
            "OldTown",
            "SWISU",
            "Sawyer",
            "SawyerW",
            "Somerst",
            "StoneBr",
            "Timber",
            "Veenker",
        ],
    }

    # Only include the top 4 numerical features based on importance
    numerical_defaults = {
        "OverallQual": 5,
        "GrLivArea": 1500,
        "YearBuilt": 1980,
        "TotalBsmtSF": 1000,
    }

    # Default values for all other features (not exposed in UI)
    default_values = {
        "MSSubClass": 20,
        "LotFrontage": 60,
        "LotArea": 8000,
        "OverallCond": 5,
        "YearRemodAdd": 1980,
        "MasVnrArea": 0,
        "BsmtFinSF1": 500,
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 500,
        "1stFlrSF": 1000,
        "2ndFlrSF": 500,
        "LowQualFinSF": 0,
        "BsmtFullBath": 0,
        "BsmtHalfBath": 0,
        "FullBath": 2,
        "HalfBath": 0,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "TotRmsAbvGrd": 6,
        "Fireplaces": 0,
        "GarageYrBlt": 1980,
        "GarageCars": 2,
        "GarageArea": 400,
        "WoodDeckSF": 0,
        "OpenPorchSF": 0,
        "EnclosedPorch": 0,
        "ThreeSsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "MiscVal": 0,
        "MoSold": 6,
        "YrSold": 2025,
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LandSlope": "Gtl",
        "BsmtQual": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "Unf",
        "BsmtFinType2": "Unf",
        "BsmtCond": "TA",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "ExterQual": "TA",
        "ExterCond": "TA",
        "FireplaceQu": "NA",
        "Functional": "Typ",
        "GarageFinish": "Unf",
        "GarageQual": "TA",
        "HeatingQC": "TA",
        "KitchenQual": "TA",
        "PavedDrive": "Y",
        "Street": "Pave",
        "LotConfig": "Inside",
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrType": "None",
        "Foundation": "PConc",
        "SaleType": "WD",
        "SaleCondition": "Normal",
        "Heating": "GasA",
        "GarageType": "Attchd",
    }

    return options, numerical_defaults, default_values


model_name, model, preprocessor = get_model_info()


def format_price(price):
    """Format the price with dollar sign and commas"""
    return f"${price:,.2f}"


# Style functions
def set_page_style(is_dark_mode):
    if is_dark_mode:
        bg_color = BACKGROUND_COLOR_DARK
        text_color = TEXT_COLOR_DARK
    else:
        bg_color = BACKGROUND_COLOR_LIGHT
        text_color = TEXT_COLOR_LIGHT

    # Set page background and text colors using CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stSidebar .sidebar-content {{
            background-color: {bg_color};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {text_color};
        }}
        .stButton>button {{
            background-color: #4CAF50;
            color: white;
            border-radius: 12px;
            border: none;
            padding: 10px 24px;
            transition-duration: 0.4s;
        }}
        .stButton>button:hover {{
            background-color: #45a049;
        }}
        .info-box {{
            background-color: {"#1e3a49" if is_dark_mode else "#e1f5fe"};
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }}
        .prediction-box {{
            background-color: {"#253d2f" if is_dark_mode else "#edf7ed"};
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }}
        .greeting-box {{
            background-color: {"#2d3748" if is_dark_mode else "#f8f9fa"};
            border-radius: 10px;
            padding: 25px;
            margin: 15px 0;
            border-left: 5px solid #4CAF50;
        }}
        .important-text {{
            color: {"#4CAF50" if is_dark_mode else "#1e8e3e"};
            font-weight: 600;
        }}
        .header-with-line {{
            border-bottom: 2px solid {"#555" if is_dark_mode else "#ddd"};
            padding-bottom: 8px;
            margin-bottom: 16px;
        }}
        .model-info {{
            font-style: italic;
            opacity: 0.8;
        }}
        .big-price {{
            font-size: 48px;
            font-weight: bold;
            color: {"#4CAF50" if is_dark_mode else "#1e8e3e"};
            text-align: center;
            padding: 20px;
            margin: 20px 0;
        }}
        .price-range {{
            font-size: 24px;
            color: {"#4CAF50" if is_dark_mode else "#1e8e3e"};
            text-align: center;
            margin: 10px 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Main App UI
def main():
    # Get options
    feature_options, numerical_defaults, default_values = get_feature_options()

    # State for theme toggling
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    # State for prediction
    if "prediction_made" not in st.session_state:
        st.session_state.prediction_made = False
        st.session_state.predicted_price = 0

    # Apply the style based on the current mode
    set_page_style(st.session_state.dark_mode)

    # Sidebar
    with st.sidebar:
        st.title("🏠 House Price Predictor")

        # Theme toggle
        theme_toggle = st.toggle("Dark Mode", value=st.session_state.dark_mode)
        if theme_toggle != st.session_state.dark_mode:
            st.session_state.dark_mode = theme_toggle
            st.rerun()

        st.markdown("---")
        st.subheader("About")
        st.markdown(
            """
        This app predicts house price ranges based on 5 key factors that most influence house values.
        
        <div class="model-info">
        Using: <b>{}</b> model
        </div>
        """.format(model_name),
            unsafe_allow_html=True,
        )

        # # Feature importance visualization
        # if os.path.exists("artifacts/plots/feature_importance.png"):
        #     st.markdown("---")
        #     st.subheader("Feature Importance")
        #     st.image(
        #         "artifacts/plots/feature_importance.png",
        #         caption="Top Feature Importances",
        #     )

    # Show prediction at the top if available
    if st.session_state.prediction_made:
        base_price = st.session_state.predicted_price
        low_range = base_price * 0.9
        high_range = base_price * 1.1

        st.markdown(
            f"""
            <div class="big-price">
                Estimated House Price Range
            </div>
            <div class="price-range">
                {format_price(low_range)} - {format_price(high_range)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Welcome greeting
    st.markdown(
        """
        <div class="greeting-box">
            <h2>👋 Welcome to the House Price Predictor!</h2>
            <p>This simple tool helps you estimate the selling price range of a house based on just 5 key factors. Our analysis of thousands of home sales has identified these as the most influential factors that impact house values.</p>
            <p>Simply input the details below, and our machine learning model will provide a realistic price range estimate.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main content area
    st.subheader("Enter Key House Details")
    st.markdown(
        """
        <div class="info-box">
            Please fill in the 5 key features below that have the biggest impact on house prices.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Create columns for feature inputs (just 5 most important features)
    col1, col2 = st.columns(2)

    with col1:
        # Overall Quality - #1 most important feature
        overall_qual = st.slider(
            "Overall Quality (Materials & Finish) ⭐",
            1,
            10,
            numerical_defaults["OverallQual"],
            help="Rating of the overall quality of materials and finish (1=Very Poor, 10=Excellent) - THE #1 factor affecting house price",
        )

        # Total Living Area - #2 most important feature
        gr_liv_area = st.number_input(
            "Total Living Area (Square Feet) 🏠",
            min_value=500,
            max_value=8000,
            value=numerical_defaults["GrLivArea"],
            help="Total above-ground living area in square feet - The #2 most important factor",
        )

        # Neighborhood - important categorical feature
        neighborhood = st.selectbox(
            "Neighborhood Location 🏘️",
            options=feature_options["Neighborhood"],
            help="The neighborhood location has a significant impact on house value",
        )

    with col2:
        # Year Built - #3 most important feature
        year_built = st.number_input(
            "Year of Construction 📅",
            min_value=1900,
            max_value=2025,
            value=numerical_defaults["YearBuilt"],
            help="The year the house was originally built - Newer houses typically command higher prices",
        )

        # Basement Size - #4 most important feature
        total_bsmt_sf = st.number_input(
            "Total Basement Area (Square Feet) 🏠",
            min_value=0,
            max_value=6000,
            value=numerical_defaults["TotalBsmtSF"],
            help="Total square footage of basement area - Houses with large, finished basements are more valuable",
        )

    # Prediction button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("Calculate Price Range 💲", use_container_width=True)

    if predict_button:
        try:
            # Create data object with all inputs
            data = CustomData(
                # Top 5 features from the form
                OverallQual=overall_qual,
                GrLivArea=gr_liv_area,
                YearBuilt=year_built,
                TotalBsmtSF=total_bsmt_sf,
                Neighborhood=neighborhood,
                # All other features set to default values
                MSSubClass=default_values["MSSubClass"],
                LotFrontage=default_values["LotFrontage"],
                LotArea=default_values["LotArea"],
                OverallCond=default_values["OverallCond"],
                YearRemodAdd=default_values["YearRemodAdd"],
                MasVnrArea=default_values["MasVnrArea"],
                BsmtFinSF1=default_values["BsmtFinSF1"],
                BsmtFinSF2=default_values["BsmtFinSF2"],
                BsmtUnfSF=default_values["BsmtUnfSF"],
                FirstFlrSF=default_values["1stFlrSF"],
                SecondFlrSF=default_values["2ndFlrSF"],
                LowQualFinSF=default_values["LowQualFinSF"],
                BsmtFullBath=default_values["BsmtFullBath"],
                BsmtHalfBath=default_values["BsmtHalfBath"],
                FullBath=default_values["FullBath"],
                HalfBath=default_values["HalfBath"],
                BedroomAbvGr=default_values["BedroomAbvGr"],
                KitchenAbvGr=default_values["KitchenAbvGr"],
                TotRmsAbvGrd=default_values["TotRmsAbvGrd"],
                Fireplaces=default_values["Fireplaces"],
                GarageYrBlt=default_values["GarageYrBlt"],
                GarageCars=default_values["GarageCars"],
                GarageArea=default_values["GarageArea"],
                WoodDeckSF=default_values["WoodDeckSF"],
                OpenPorchSF=default_values["OpenPorchSF"],
                EnclosedPorch=default_values["EnclosedPorch"],
                ThreeSsnPorch=default_values["ThreeSsnPorch"],
                ScreenPorch=default_values["ScreenPorch"],
                PoolArea=default_values["PoolArea"],
                MiscVal=default_values["MiscVal"],
                MoSold=default_values["MoSold"],
                YrSold=default_values["YrSold"],
                LotShape=default_values["LotShape"],
                LandContour=default_values["LandContour"],
                Utilities=default_values["Utilities"],
                LandSlope=default_values["LandSlope"],
                BsmtQual=default_values["BsmtQual"],
                BsmtExposure=default_values["BsmtExposure"],
                BsmtFinType1=default_values["BsmtFinType1"],
                BsmtFinType2=default_values["BsmtFinType2"],
                BsmtCond=default_values["BsmtCond"],
                CentralAir=default_values["CentralAir"],
                Electrical=default_values["Electrical"],
                ExterQual=default_values["ExterQual"],
                ExterCond=default_values["ExterCond"],
                FireplaceQu=default_values["FireplaceQu"],
                Functional=default_values["Functional"],
                GarageFinish=default_values["GarageFinish"],
                GarageQual=default_values["GarageQual"],
                HeatingQC=default_values["HeatingQC"],
                KitchenQual=default_values["KitchenQual"],
                PavedDrive=default_values["PavedDrive"],
                Street=default_values["Street"],
                LotConfig=default_values["LotConfig"],
                Condition1=default_values["Condition1"],
                Condition2=default_values["Condition2"],
                BldgType=default_values["BldgType"],
                HouseStyle=default_values["HouseStyle"],
                RoofStyle=default_values["RoofStyle"],
                RoofMatl=default_values["RoofMatl"],
                Exterior1st=default_values["Exterior1st"],
                Exterior2nd=default_values["Exterior2nd"],
                MasVnrType=default_values["MasVnrType"],
                Foundation=default_values["Foundation"],
                SaleType=default_values["SaleType"],
                SaleCondition=default_values["SaleCondition"],
                Heating=default_values["Heating"],
                GarageType=default_values["GarageType"],
            )

            # Convert to DataFrame
            df = data.get_data_as_dataframe()

            # Create pipeline and predict
            with st.spinner("Calculating house price range..."):
                # Slight delay to show loading effect
                import time

                time.sleep(0.5)

                # Get prediction
                pipeline = PredictPipeline()
                prediction = pipeline.predict(df)[0]

                # Store in session state
                st.session_state.prediction_made = True
                st.session_state.predicted_price = prediction

                # Calculate price range (±10% for uncertainty)
                low_price = prediction * 0.9
                high_price = prediction * 1.1

                # Format and display prediction with range
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <h2>Estimated House Price Range</h2>
                        <h1 class="important-text">{format_price(low_price)} - {format_price(high_price)}</h1>
                        <p>Most likely price: {format_price(prediction)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Add visualization based on key features
                st.markdown("### Understanding This Price Range")

                # Create visualization of the most impactful features
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Show a comparison chart
                    st.markdown("#### Key Factors Affecting This Estimate")

                    # Create a radar chart of key features
                    key_features = {
                        "Overall Quality": overall_qual / 10,
                        "Living Area": gr_liv_area / 3000,
                        "Year Built": (year_built - 1900) / 120,
                        "Basement Size": total_bsmt_sf / 2000,
                        "Location Value": 0.7,  # Simplified metric for neighborhood
                    }

                    # Generate radar chart
                    categories = list(key_features.keys())
                    values = list(key_features.values())

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill="toself",
                            name="House Features",
                            line_color="#4CAF50",
                        )
                    )

                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("#### What This Range Means")

                    # Create a simple breakdown of what contributes to price
                    st.markdown("""
                    The price range reflects market uncertainty and takes into account:
                    
                    - **Overall Quality**: The #1 price factor
                    - **Living Area**: Size significantly impacts value
                    - **House Age**: Newer houses typically cost more
                    - **Basement**: Size adds valuable space
                    - **Location**: Neighborhood desirability
                    """)

                    st.markdown("**Market Context**")
                    st.markdown(f"""
                    This estimate is based on historical data and comparable houses in the {neighborhood} neighborhood.
                    
                    Local market conditions, seasonal trends, and unique property features can cause actual prices to vary within this range.
                    """)

                # Show model info
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(
                    f"This price range was calculated using the {model_name} model, focusing on the 5 most important factors that influence house prices. The range accounts for market variability and model uncertainty."
                )

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            logging.error(f"Prediction error: {str(e)}")


if __name__ == "__main__":
    main()
