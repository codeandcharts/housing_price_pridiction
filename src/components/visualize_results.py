# visualize_results.py
import os
import matplotlib.pyplot as plt
import numpy as np
import shap
import logging
from src.exception import CustomException
import sys


def plot_feature_importances(
    model,
    X,
    feature_names,
    top_n=10,
    save_path=None,
):
    """
    Extracts a tree‐based model's feature_importances_, matches them to feature names,
    and plots the top_n as a horizontal bar chart.
    """
    try:
        # Check if model has feature_importances_ attribute
        if not hasattr(model, "feature_importances_"):
            logging.warning(
                f"Model {type(model).__name__} doesn't have feature_importances_ attribute"
            )
            return None

        importances = model.feature_importances_

        # Ensure we have the right number of features
        if len(importances) != len(feature_names):
            logging.warning(
                f"Feature count mismatch: {len(importances)} importances vs {len(feature_names)} names"
            )
            return None

        # Sort by importance
        indices = np.argsort(importances)[-top_n:]
        top_names = [feature_names[i] for i in indices]
        top_vals = importances[indices]

        plt.figure(figsize=(10, 6))
        plt.barh(top_names, top_vals)
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importances")
        plt.tight_layout()

        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logging.info(f"Feature importances plot saved to {save_path}")
        else:
            plt.show()

        return plt

    except Exception as e:
        logging.error(f"Error in plot_feature_importances: {str(e)}")
        raise CustomException(e, sys)


def plot_shap_summary(model, X, feature_names, sample_size=200, save_path=None):
    """
    Uses SHAP to plot a summary of feature impacts.
    """
    try:
        # Sample the data if needed
        if X.shape[0] > sample_size:
            X_sample = X[:sample_size]
        else:
            X_sample = X

        # Suppress any LightGBM feature‐name warnings
        if hasattr(model, "predict"):  # Ensure model is fitted
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, X_sample, feature_names=feature_names, show=False
            )

            if save_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches="tight")
                logging.info(f"SHAP summary plot saved to {save_path}")
            else:
                plt.show()

            return plt
        else:
            logging.warning(
                "Model doesn't have predict method, can't compute SHAP values"
            )
            return None

    except Exception as e:
        logging.error(f"Error in plot_shap_summary: {str(e)}")
        raise CustomException(e, sys)


def get_feature_names_from_preprocessor(preprocessor):
    """
    Extract feature names from a column transformer preprocessor
    """
    try:
        # Access the preprocessing pipeline directly
        col_transformer = preprocessor.named_steps["preprocessor"]

        # Get column names for all transformers
        feature_names = []

        # Extract names from each transformer output
        for name, _, columns in col_transformer.transformers_:
            if name == "drop":
                continue

            if name == "num":
                # Numerical features pass through
                feature_names.extend(columns)
            elif name == "ord":
                # Ordinal features (single column each)
                feature_names.extend(columns)
            elif name == "ohe":
                # One-hot encoded features (multiple columns per feature)
                ohe = col_transformer.named_transformers_["ohe"].named_steps["onehot"]
                if hasattr(ohe, "get_feature_names_out"):
                    ohe_feature_names = ohe.get_feature_names_out(
                        input_features=columns
                    )
                    feature_names.extend(ohe_feature_names)
                else:
                    # Older scikit-learn versions
                    for col in columns:
                        feature_names.append(f"{col}_encoded")

        return feature_names

    except Exception as e:
        logging.error(f"Error extracting feature names: {str(e)}")
        # Return a list of generic feature names as fallback
        return [f"feature_{i}" for i in range(100)]
