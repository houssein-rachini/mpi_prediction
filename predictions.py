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
    "DNN+KNN": "trained_ensemble_knn_dnn_model.h5",
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
    "KNN": "models/global/trained_ensemble_knn_model.pkl",  # USED FOR ENSEMBLE
}

PRETRAINED_SCALERS_PATHS = {
    "DNN": "models/global/dnn_scaler.pkl",  # USED FOR STANDALONE
    "ML": "models/global/ml_scaler.pkl",  # USED FOR STANDALONE
    "Ensemble": "models/global/ensemble_scaler.pkl",  # USED FOR ENSEMBLE
}


# ========== Preprocessing ==========
def preprocess_data(test_data, scaler):
    feature_names = scaler.feature_names_in_
    missing_columns = [col for col in feature_names if col not in test_data.columns]
    for col in missing_columns:
        test_data[col] = 0  # fill missing
    test_data_selected = test_data[feature_names]
    return scaler.transform(test_data_selected)


# ========== Caching Models/Scalers ==========
@st.cache_resource
def load_dnn_model(USE_PRETRAINED):
    path = PRETRAINED_MODELS_PATHS["DNN"] if USE_PRETRAINED else MODEL_PATHS["DNN"]
    return load_model(
        path,
        custom_objects={
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "rmse": tf.keras.metrics.RootMeanSquaredError(),
        },
    )


@st.cache_resource
def load_dnn_scaler(USE_PRETRAINED):
    path = PRETRAINED_SCALERS_PATHS["DNN"] if USE_PRETRAINED else SCALER_PATHS["DNN"]
    return joblib.load(path)


@st.cache_resource
def load_ml_model(USE_PRETRAINED):
    path = PRETRAINED_MODELS_PATHS["ML"] if USE_PRETRAINED else MODEL_PATHS["ML"]
    return joblib.load(path)


@st.cache_resource
def load_ml_scaler(USE_PRETRAINED):
    path = PRETRAINED_SCALERS_PATHS["ML"] if USE_PRETRAINED else SCALER_PATHS["ML"]
    return joblib.load(path)


@st.cache_resource
def load_ensemble_scaler(USE_PRETRAINED):
    path = (
        PRETRAINED_SCALERS_PATHS["Ensemble"]
        if USE_PRETRAINED
        else SCALER_PATHS["Ensemble"]
    )
    return joblib.load(path)


@st.cache_resource
def load_ensemble_models(model_type, USE_PRETRAINED):
    dnn_path = (
        PRETRAINED_MODELS_PATHS[model_type]
        if USE_PRETRAINED
        else MODEL_PATHS[model_type]
    )
    dnn_model = load_model(
        dnn_path,
        custom_objects={
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "rmse": tf.keras.metrics.RootMeanSquaredError(),
        },
    )

    if model_type == "DNN+XGBoost":
        base_path = (
            PRETRAINED_MODELS_PATHS["XGBoost"]
            if USE_PRETRAINED
            else "trained_ensemble_xgb_model.json"
        )
        base_model = xgb.XGBRegressor()
        base_model.load_model(base_path)
    elif model_type == "DNN+RF":
        base_path = (
            PRETRAINED_MODELS_PATHS["RF"]
            if USE_PRETRAINED
            else "trained_ensemble_rf_model.pkl"
        )
        base_model = joblib.load(base_path)
    elif model_type == "DNN+KNN":
        base_path = (
            PRETRAINED_MODELS_PATHS["KNN"]
            if USE_PRETRAINED
            else "trained_ensemble_knn_model.pkl"
        )
        base_model = joblib.load(base_path)
    else:
        raise ValueError(f"Invalid ensemble model type: {model_type}")

    return dnn_model, base_model


# ========== Fast Predict Functions ==========


def predict_dnn_fast(test_data, dnn_model, scaler):
    test_data_scaled = preprocess_data(test_data, scaler)
    return np.clip(dnn_model.predict(test_data_scaled).flatten(), 0, 1)


def predict_ml_fast(test_data, ml_model, scaler):
    test_data_scaled = preprocess_data(test_data, scaler)
    return np.clip(ml_model.predict(test_data_scaled), 0, 1)


def predict_ensemble_fast(test_data, dnn_model, base_model, scaler, alpha):
    test_data_scaled = preprocess_data(test_data, scaler)
    y_pred_dnn = dnn_model.predict(test_data_scaled).flatten()
    y_pred_base = base_model.predict(test_data_scaled)
    y_pred = alpha * y_pred_dnn + (1 - alpha) * y_pred_base
    return np.clip(y_pred, 0, 1)


# ========== Visualization  ==========


def plot_results(test_data):
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

    model_choice = st.selectbox(
        "Select a model for prediction:",
        ["DNN", "ML", "DNN+RF", "DNN+XGBoost", "DNN+KNN"],
    )

    alpha = None
    if model_choice in ["DNN+RF", "DNN+XGBoost", "DNN+KNN"]:
        alpha = st.slider(
            "Ensemble Weight (DNN Contribution)", 0.0, 1.0, 0.15, key="testing_alpha"
        )

    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")

    if uploaded_file:
        test_data = pd.read_csv(
            uploaded_file, encoding="utf-8", encoding_errors="replace"
        )
        st.write("### Test Data Preview:")
        st.dataframe(test_data.head())

        if st.button("Predict MPI for All Available Rows"):
            with st.spinner("Generating predictions..."):

                predictions = None
                output_file = None

                if model_choice == "DNN":
                    dnn_model = load_dnn_model(USE_PRETRAINED=True)
                    scaler = load_dnn_scaler(USE_PRETRAINED=True)
                    predictions = predict_dnn_fast(test_data, dnn_model, scaler)
                    output_file = "test_results_dnn.csv"

                elif model_choice == "ML":
                    ml_model = load_ml_model(USE_PRETRAINED=True)
                    scaler = load_ml_scaler(USE_PRETRAINED=True)
                    predictions = predict_ml_fast(test_data, ml_model, scaler)
                    output_file = "test_results_ml.csv"

                else:
                    dnn_model, base_model = load_ensemble_models(
                        model_choice, USE_PRETRAINED=True
                    )
                    scaler = load_ensemble_scaler(USE_PRETRAINED=True)
                    predictions = predict_ensemble_fast(
                        test_data, dnn_model, base_model, scaler, alpha
                    )
                    if model_choice == "DNN+RF":
                        output_file = "test_results_ensemble_rf.csv"
                    elif model_choice == "DNN+XGBoost":
                        output_file = "test_results_ensemble_xgb.csv"
                    elif model_choice == "DNN+KNN":
                        output_file = "test_results_ensemble_knn.csv"

                if predictions is not None:
                    test_data["Predicted_MPI"] = predictions
                    test_data.to_csv(output_file, index=False)
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
