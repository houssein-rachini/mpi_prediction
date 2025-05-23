import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
import xgboost as xgb
import io
import ee
import os
from google.oauth2 import service_account

service_account_info = dict(st.secrets["google_ee"])  # No need for .to_json()

credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=["https://www.googleapis.com/auth/earthengine"]
)

ee.Initialize(credentials)

# Define model paths
MODEL_PATHS = {
    "DNN": "trained_dnn_model.h5",
    "ML": "trained_ml_model.pkl",
    "DNN+RF": "trained_ensemble_rf_dnn_model.h5",
    "DNN+XGBoost": "trained_ensemble_xgb_dnn_model.h5",
}

SCALER_PATHS = {
    "DNN": "dnn_scaler.pkl",
    "ML": "ml_scaler.pkl",
    "Ensemble": "ensemble_scaler.pkl",
}

PRETRAINED_MODELS_PATHS = {
    "DNN": "models/global/trained_dnn_model.h5",  # USED FOR STANDALONE
    "ML": "models/global/trained_ml_model.pkl",  # USED FOR STANDALONE
    "DNN+RF": "models/global/trained_ensemble_rf_dnn_model.h5",  # USED FOR ENSEMBLE
    "DNN+XGBoost": "models/global/trained_ensemble_xgb_dnn_model.h5",  # USED FOR ENSEMBLE
    "XGBoost": "models/global/trained_ensemble_xgb_model.json",  # USED FOR ENSEMBLE
    "RF": "models/global/trained_ensemble_rf_model.pkl",  # USED FOR ENSEMBLE
}

PRETRAINED_SCALERS_PATHS = {
    "DNN": "models/global/dnn_scaler.pkl",  # USED FOR STANDALONE
    "ML": "models/global/ml_scaler.pkl",  # USED FOR STANDALONE
    "Ensemble": "models/global/ensemble_scaler.pkl",  # USED FOR ENSEMBLE
}


def load_scaler(model_type):
    """Load scaler based on model type."""
    return joblib.load(SCALER_PATHS[model_type])


def load_pretrained_scaler(model_type):
    """Load pretrained scaler based on model type."""
    return joblib.load(PRETRAINED_SCALERS_PATHS[model_type])


def preprocess_data(test_data, scaler):
    """Ensure test data matches trained model features."""
    feature_names = scaler.feature_names_in_
    missing_columns = [col for col in feature_names if col not in test_data.columns]
    for col in missing_columns:
        test_data[col] = 0  # Fill missing columns with 0
    test_data_selected = test_data[feature_names]
    return scaler.transform(test_data_selected)


def predict_dnn(test_data, USE_PRETRAINED_MODELS):
    """Predict using the standalone DNN model."""
    if USE_PRETRAINED_MODELS:
        dnn_model = load_model(
            PRETRAINED_MODELS_PATHS["DNN"],
            custom_objects={
                "mse": MeanSquaredError(),
                "mae": MeanAbsoluteError(),
                "rmse": tf.keras.metrics.RootMeanSquaredError(),
            },
        )
        scaler = load_pretrained_scaler("DNN")
        test_data_scaled = preprocess_data(test_data, scaler)
        predictions = dnn_model.predict(test_data_scaled).flatten()
    else:
        if not os.path.exists(MODEL_PATHS["DNN"]):
            st.error("❌ DNN model file not found. Please train the model first.")
            return None
        if not os.path.exists(SCALER_PATHS["DNN"]):
            st.error("❌ DNN scaler file not found. Please train the model first.")
            return None

        dnn_model = load_model(
            MODEL_PATHS["DNN"],
            custom_objects={
                "mse": MeanSquaredError(),
                "mae": MeanAbsoluteError(),
                "rmse": tf.keras.metrics.RootMeanSquaredError(),
            },
        )
        scaler = load_scaler("DNN")
        test_data_scaled = preprocess_data(test_data, scaler)
        predictions = dnn_model.predict(test_data_scaled).flatten()
    # return np.maximum(predictions, 0)
    return np.clip(predictions, 0, 1)


def predict_ml(test_data, USE_PRETRAINED_MODELS):
    """Predict using an ML model."""

    model_path = (
        PRETRAINED_MODELS_PATHS["ML"] if USE_PRETRAINED_MODELS else MODEL_PATHS["ML"]
    )
    scaler_path = (
        PRETRAINED_SCALERS_PATHS["ML"] if USE_PRETRAINED_MODELS else SCALER_PATHS["ML"]
    )

    if not os.path.exists(model_path):
        st.error("❌ ML model file not found. Please train or upload it first.")
        return None

    if not os.path.exists(scaler_path):
        st.error("❌ ML scaler file not found. Please train or upload it first.")
        return None

    ml_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    test_data_scaled = preprocess_data(test_data, scaler)
    predictions = ml_model.predict(test_data_scaled)

    return np.clip(predictions, 0, 1)


def predict_ensemble(test_data, model_type, alpha, USE_PRETRAINED_MODELS):
    """Predict using an ensemble model (DNN + XGBoost or DNN + RF)."""

    model_path = (
        PRETRAINED_MODELS_PATHS[model_type]
        if USE_PRETRAINED_MODELS
        else MODEL_PATHS[model_type]
    )
    scaler_path = (
        PRETRAINED_SCALERS_PATHS["Ensemble"]
        if USE_PRETRAINED_MODELS
        else SCALER_PATHS["Ensemble"]
    )

    if not os.path.exists(model_path):
        st.error(f"❌ DNN model file for '{model_type}' not found.")
        return None
    if not os.path.exists(scaler_path):
        st.error("❌ Ensemble scaler file not found.")
        return None

    dnn_model = load_model(
        model_path,
        custom_objects={
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "rmse": tf.keras.metrics.RootMeanSquaredError(),
        },
    )
    scaler = joblib.load(scaler_path)
    test_data_scaled = preprocess_data(test_data, scaler)

    # Load base model
    base_model_path = None
    base_model = None

    if model_type == "DNN+XGBoost":
        base_model_path = (
            PRETRAINED_MODELS_PATHS["XGBoost"]
            if USE_PRETRAINED_MODELS
            else "trained_ensemble_xgb_model.json"
        )
        if not os.path.exists(base_model_path):
            st.error("❌ XGBoost model file not found.")
            return None
        base_model = xgb.XGBRegressor()
        base_model.load_model(base_model_path)

    elif model_type == "DNN+RF":
        base_model_path = (
            PRETRAINED_MODELS_PATHS["RF"]
            if USE_PRETRAINED_MODELS
            else "trained_ensemble_rf_model.pkl"
        )
        if not os.path.exists(base_model_path):
            st.error("❌ Random Forest model file not found.")
            return None
        base_model = joblib.load(base_model_path)

    y_pred_dnn = dnn_model.predict(test_data_scaled).flatten()
    y_pred_base = base_model.predict(test_data_scaled)
    y_pred_ensemble = alpha * y_pred_dnn + (1 - alpha) * y_pred_base

    return np.clip(y_pred_ensemble, 0, 1)


def plot_results(test_data):
    """Generate visualizations for predictions."""
    st.subheader("📈 Predictions vs Actual MPI")
    if "MPI" in test_data.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=test_data["MPI"], y=test_data["Predicted_MPI"], alpha=0.6, ax=ax
        )
        ax.set_xlabel("Actual MPI")
        ax.set_ylabel("Predicted MPI")
        ax.set_title("Actual vs Predicted MPI")
        ax.axline((0, 0), slope=1, color="red", linestyle="--")
        st.pyplot(fig)
    else:
        st.warning(
            "No 'MPI' column found in test data. Skipping Actual vs Predicted plot."
        )

    st.subheader("📊 Distribution of Predicted MPI")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(test_data["Predicted_MPI"], bins=30, kde=True, color="blue", ax=ax)
    ax.set_xlabel("Predicted MPI")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Predicted MPI")
    st.pyplot(fig)


def show_predictions_tab():
    st.title("🔮 MPI Prediction")

    # Model Selection
    model_choice = st.selectbox(
        "Select a model for prediction:", ["DNN", "ML", "DNN+RF", "DNN+XGBoost"]
    )
    if model_choice in ["DNN+RF", "DNN+XGBoost"]:
        # allow the user to select alpha
        alpha = st.slider(
            "Ensemble Weight (DNN Contribution)", 0.0, 1.0, 0.15, key="testing_alpha"
        )

    # File Upload
    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")

    if uploaded_file:
        test_data = pd.read_csv(
            uploaded_file, encoding="utf-8", encoding_errors="replace"
        )
        st.write("### Test Data Preview:")
        st.dataframe(test_data.head())

        # Perform Prediction
        if st.button("Predict MPI for All Available Years"):
            with st.spinner("Generating predictions..."):
                if model_choice == "DNN":
                    predictions = predict_dnn(test_data)
                    output_file = "test_results_dnn.csv"
                elif model_choice == "ML":
                    predictions = predict_ml(test_data)
                    output_file = "test_results_ml.csv"
                elif model_choice == "DNN+RF":
                    predictions = predict_ensemble(test_data, "DNN+RF", alpha)
                    output_file = "test_results_ensemble_rf.csv"
                elif model_choice == "DNN+XGBoost":
                    predictions = predict_ensemble(test_data, "DNN+XGBoost", alpha)
                    output_file = "test_results_ensemble_xgb.csv"

                # Save predictions
                test_data["Predicted_MPI"] = predictions
                test_data.to_csv(output_file, index=False)
                if predictions is not None:
                    st.success(f"✅ Predictions saved to {output_file}")

                    st.download_button(
                        label="Download Predictions CSV",
                        data=test_data.to_csv(index=False),
                        file_name=output_file,
                        mime="text/csv",
                    )

                    plot_results(test_data)
                else:
                    st.error("❌ Prediction failed. Please check the error messages.")
