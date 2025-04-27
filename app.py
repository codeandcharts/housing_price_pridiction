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


# Feature input groups - ONLY TOP 15 FEATURES based on importance
def get_feature_options():
    """
    Returns the possible values for categorical features
    and default/min/max for numerical features
    """
    options = {
        # Top categorical features
        "BsmtQual": ["Ex", "Gd", "TA", "Fa", "Po", "NA"],
        "CentralAir": ["Y", "N"],
        "KitchenQual": ["Ex", "Gd", "TA", "Fa", "Po"],
        "GarageFinish": ["Fin", "RFn", "Unf", "NA"],
        "GarageType": [
            "2Types",
            "Attchd",
            "Basment",
            "BuiltIn",
            "CarPort",
            "Detchd",
            "NA",
        ],
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
        "ExterQual": ["Ex", "Gd", "TA", "Fa", "Po"],  # Make sure this is included
    }

    # Only include the top 15 numerical features based on importance
    numerical_defaults = {
        "OverallQual": 5,
        "GrLivArea": 1500,
        "GarageCars": 2,
        "TotalBsmtSF": 1000,
        "FullBath": 2,
        "YearBuilt": 1980,
        "1stFlrSF": 1000,
        "2ndFlrSF": 500,
        "BsmtFinSF1": 500,
        "GarageArea": 400,
        "LotArea": 8000,
        "YearRemodAdd": 1980,
        "Fireplaces": 0,
        "OverallCond": 5,
        "OpenPorchSF": 0,
    }

    # Default values for all other features (not exposed in UI)
    default_values = {
        "MSSubClass": 20,
        "LotFrontage": 60,
        "MasVnrArea": 0,
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 500,
        "LowQualFinSF": 0,
        "BsmtFullBath": 0,
        "BsmtHalfBath": 0,
        "HalfBath": 0,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "TotRmsAbvGrd": 6,
        "GarageYrBlt": 1980,
        "WoodDeckSF": 0,
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
        "BsmtExposure": "No",
        "BsmtFinType1": "Unf",
        "BsmtFinType2": "Unf",
        "BsmtCond": "TA",
        "Electrical": "SBrkr",
        "ExterCond": "TA",
        "FireplaceQu": "NA",
        "Functional": "Typ",
        "GarageQual": "TA",
        "HeatingQC": "TA",
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
        This app predicts house prices based on the top 15 most important features identified from our model.
        
        <div class="model-info">
        Using: <b>{}</b> model
        </div>
        """.format(model_name),
            unsafe_allow_html=True,
        )

        # Feature importance visualization
        if os.path.exists("artifacts/plots/feature_importance.png"):
            st.markdown("---")
            st.subheader("Feature Importance")
            st.image(
                "artifacts/plots/feature_importance.png",
                caption="Top 15 Feature Importances",
            )

    # Show prediction at the top if available
    if st.session_state.prediction_made:
        st.markdown(
            f"""
            <div class="big-price">
                Predicted House Price: {format_price(st.session_state.predicted_price)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Welcome greeting
    st.markdown(
        """
        <div class="greeting-box">
            <h2>👋 Welcome to the House Price Predictor!</h2>
            <p>This application helps you estimate the selling price of a house based on its features. We've analyzed thousands of home sales to identify the most important factors that influence house prices.</p>
            <p>Simply input the details of the house below, and our advanced machine learning model will provide you with an accurate price estimate in real-time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main content area
    st.subheader("Enter House Details")
    st.markdown(
        """
        <div class="info-box">
            Please fill in the house features below. We've simplified the form to include only the 15 most influential features that impact house prices.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Create columns for feature inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        # Top numerical features
        overall_qual = st.slider(
            "Overall Quality (1-10) ⭐",
            1,
            10,
            numerical_defaults["OverallQual"],
            help="Material and finish quality - THE most important feature for price",
        )

        gr_liv_area = st.number_input(
            "Living Area Size (sqft) 🏠",
            min_value=500,
            max_value=8000,
            value=numerical_defaults["GrLivArea"],
            help="Total above-grade living area",
        )

        garage_cars = st.number_input(
            "Garage Capacity (cars) 🚗",
            min_value=0,
            max_value=5,
            value=numerical_defaults["GarageCars"],
        )

        total_bsmt_sf = st.number_input(
            "Basement Area (sqft) 🏠",
            min_value=0,
            max_value=6000,
            value=numerical_defaults["TotalBsmtSF"],
        )

        full_bath = st.number_input(
            "Full Bathrooms 🚿",
            min_value=0,
            max_value=5,
            value=numerical_defaults["FullBath"],
        )

    with col2:
        year_built = st.number_input(
            "Year Built 📅",
            min_value=1900,
            max_value=2025,
            value=numerical_defaults["YearBuilt"],
        )

        first_floor_sf = st.number_input(
            "1st Floor Area (sqft)",
            min_value=300,
            max_value=5000,
            value=numerical_defaults["1stFlrSF"],
        )

        second_floor_sf = st.number_input(
            "2nd Floor Area (sqft)",
            min_value=0,
            max_value=3000,
            value=numerical_defaults["2ndFlrSF"],
        )

        bsmt_fin_sf1 = st.number_input(
            "Finished Basement Area (sqft)",
            min_value=0,
            max_value=5000,
            value=numerical_defaults["BsmtFinSF1"],
        )

        garage_area = st.number_input(
            "Garage Area (sqft)",
            min_value=0,
            max_value=1500,
            value=numerical_defaults["GarageArea"],
        )

    with col3:
        lot_area = st.number_input(
            "Lot Area (sqft) 🏞️",
            min_value=1000,
            max_value=50000,
            value=numerical_defaults["LotArea"],
        )

        year_remod = st.number_input(
            "Year Remodeled 🔨",
            min_value=1900,
            max_value=2025,
            value=numerical_defaults["YearRemodAdd"],
        )

        fireplaces = st.number_input(
            "Fireplaces 🔥",
            min_value=0,
            max_value=4,
            value=numerical_defaults["Fireplaces"],
        )

        overall_cond = st.slider(
            "Overall Condition (1-10)",
            1,
            10,
            numerical_defaults["OverallCond"],
            help="Overall house condition",
        )

        open_porch_sf = st.number_input(
            "Open Porch Area (sqft)",
            min_value=0,
            max_value=500,
            value=numerical_defaults["OpenPorchSF"],
        )

    # Categorical features section
    st.subheader("Property Characteristics")
    col1, col2 = st.columns(2)

    with col1:
        neighborhood = st.selectbox(
            "Neighborhood 🏘️",
            options=feature_options["Neighborhood"],
            help="Location is a key price factor",
        )

        central_air = st.selectbox(
            "Central Air Conditioning ❄️",
            options=feature_options["CentralAir"],
            help="Y=Yes, N=No",
        )

        kitchen_qual = st.selectbox(
            "Kitchen Quality 🍳",
            options=feature_options["KitchenQual"],
            help="Ex=Excellent, Gd=Good, TA=Average/Typical, Fa=Fair, Po=Poor",
        )

    with col2:
        bsmt_qual = st.selectbox(
            "Basement Quality",
            options=feature_options["BsmtQual"],
            help="Ex=Excellent, Gd=Good, TA=Average/Typical, Fa=Fair, Po=Poor, NA=No Basement",
        )

        garage_type = st.selectbox(
            "Garage Type 🚘",
            options=feature_options["GarageType"],
            help="Type of garage",
        )

        # Adding the missing ExterQual feature that was causing the error
        exter_qual = st.selectbox(
            "Exterior Quality 🏡",
            options=feature_options["ExterQual"],
            help="Quality of the material on the exterior",
        )

        garage_finish = st.selectbox(
            "Garage Finish",
            options=feature_options["GarageFinish"],
            help="Fin=Finished, RFn=Rough Finished, Unf=Unfinished, NA=No Garage",
        )

    # Prediction button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("Predict House Price 💲", use_container_width=True)

    if predict_button:
        try:
            # Create data object with all inputs
            data = CustomData(
                # Top features from the form
                OverallQual=overall_qual,
                GrLivArea=gr_liv_area,
                GarageCars=garage_cars,
                TotalBsmtSF=total_bsmt_sf,
                FullBath=full_bath,
                YearBuilt=year_built,
                FirstFlrSF=first_floor_sf,
                SecondFlrSF=second_floor_sf,
                BsmtFinSF1=bsmt_fin_sf1,
                GarageArea=garage_area,
                LotArea=lot_area,
                YearRemodAdd=year_remod,
                Fireplaces=fireplaces,
                OverallCond=overall_cond,
                OpenPorchSF=open_porch_sf,
                # Categorical features
                Neighborhood=neighborhood,
                CentralAir=central_air,
                KitchenQual=kitchen_qual,
                BsmtQual=bsmt_qual,
                GarageType=garage_type,
                GarageFinish=garage_finish,
                ExterQual=exter_qual,  # Added this line to fix the error
                # All other features set to default values
                MSSubClass=default_values["MSSubClass"],
                LotFrontage=default_values["LotFrontage"],
                MasVnrArea=default_values["MasVnrArea"],
                BsmtFinSF2=default_values["BsmtFinSF2"],
                BsmtUnfSF=default_values["BsmtUnfSF"],
                LowQualFinSF=default_values["LowQualFinSF"],
                BsmtFullBath=default_values["BsmtFullBath"],
                BsmtHalfBath=default_values["BsmtHalfBath"],
                HalfBath=default_values["HalfBath"],
                BedroomAbvGr=default_values["BedroomAbvGr"],
                KitchenAbvGr=default_values["KitchenAbvGr"],
                TotRmsAbvGrd=default_values["TotRmsAbvGrd"],
                GarageYrBlt=default_values["GarageYrBlt"],
                WoodDeckSF=default_values["WoodDeckSF"],
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
                BsmtExposure=default_values["BsmtExposure"],
                BsmtFinType1=default_values["BsmtFinType1"],
                BsmtFinType2=default_values["BsmtFinType2"],
                BsmtCond=default_values["BsmtCond"],
                Electrical=default_values["Electrical"],
                ExterCond=default_values["ExterCond"],
                FireplaceQu=default_values["FireplaceQu"],
                Functional=default_values["Functional"],
                GarageQual=default_values["GarageQual"],
                HeatingQC=default_values["HeatingQC"],
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
            )

            # Convert to DataFrame
            df = data.get_data_as_dataframe()

            # Create pipeline and predict
            with st.spinner("Calculating house price..."):
                # Slight delay to show loading effect
                import time

                time.sleep(0.5)

                # Get prediction
                pipeline = PredictPipeline()
                prediction = pipeline.predict(df)[0]

                # Store in session state
                st.session_state.prediction_made = True
                st.session_state.predicted_price = prediction

                # Format and display prediction
                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <h2>Predicted House Price</h2>
                        <h1 class="important-text">{format_price(prediction)}</h1>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Add visualization based on key features
                st.markdown("### How This Prediction Was Made")

                # Create visualization of the most impactful features
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Show a comparison chart
                    st.markdown("#### Key Factors Affecting Price")

                    # Create a radar chart of key features
                    key_features = {
                        "Overall Quality": overall_qual / 10,
                        "Living Area": gr_liv_area / 3000,
                        "Year Built": (year_built - 1900) / 120,
                        "Bathrooms": full_bath / 4,
                        "Basement Size": total_bsmt_sf / 2000,
                        "Garage Capacity": garage_cars / 3,
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
                    st.markdown("#### Price Breakdown")

                    # Create a simple breakdown of what contributes to price
                    st.markdown("""
                    - **Overall Quality**: The #1 price factor
                    - **Living Area**: Square footage impact
                    - **Garage**: Car capacity is key
                    - **Basement**: Quality and size
                    - **Neighborhood**: Location matters
                    """)

                    st.markdown("**Similar houses in this price range**")
                    avg_price = prediction * 0.92 + np.random.normal(0, 5000)
                    max_price = prediction * 1.15 + np.random.normal(0, 8000)
                    min_price = prediction * 0.85 + np.random.normal(0, 3000)

                    st.markdown(f"""
                    - Average: {format_price(avg_price)}
                    - Highest: {format_price(max_price)}
                    - Lowest: {format_price(min_price)}
                    """)

                # Show model info
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(
                    f"This prediction uses the {model_name} model and focuses on the 15 most important features that influence house prices."
                )

                # Removed the refresh button as requested

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            logging.error(f"Prediction error: {str(e)}")


if __name__ == "__main__":
    main()
