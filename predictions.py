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


def load_scaler(model_type):
    """Load scaler based on model type."""
    return joblib.load(SCALER_PATHS[model_type])


def preprocess_data(test_data, scaler):
    """Ensure test data matches trained model features."""
    feature_names = scaler.feature_names_in_
    missing_columns = [col for col in feature_names if col not in test_data.columns]
    for col in missing_columns:
        test_data[col] = 0  # Fill missing columns with 0
    test_data_selected = test_data[feature_names]
    return scaler.transform(test_data_selected)


def predict_dnn(test_data):
    """Predict using the standalone DNN model."""
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
    return np.maximum(predictions, 0)


def predict_ml(test_data):
    """Predict using an ML model."""
    ml_model = joblib.load(MODEL_PATHS["ML"])
    scaler = load_scaler("ML")
    test_data_scaled = preprocess_data(test_data, scaler)
    predictions = ml_model.predict(test_data_scaled)
    return np.maximum(predictions, 0)


def predict_ensemble(test_data, model_type, alpha):
    """Predict using an ensemble model (DNN + XGBoost or DNN + RF)."""
    dnn_model = load_model(
        MODEL_PATHS[model_type],
        custom_objects={
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "rmse": tf.keras.metrics.RootMeanSquaredError(),
        },
    )

    if model_type == "DNN+XGBoost":
        base_model = xgb.XGBRegressor()
        base_model.load_model("trained_ensemble_xgb_model.json")
    elif model_type == "DNN+RF":
        base_model = joblib.load("trained_ensemble_rf_model.pkl")

    scaler = load_scaler("Ensemble")
    test_data_scaled = preprocess_data(test_data, scaler)

    y_pred_dnn = dnn_model.predict(test_data_scaled).flatten()
    y_pred_base = base_model.predict(test_data_scaled)
    y_pred_ensemble = alpha * y_pred_dnn + (1 - alpha) * y_pred_base
    return np.maximum(y_pred_ensemble, 0)


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
        if st.button("Predict MPI"):
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

                st.success(f"✅ Predictions saved to {output_file}")

                st.download_button(
                    label="Download Predictions CSV",
                    data=test_data.to_csv(index=False),
                    file_name=output_file,
                    mime="text/csv",
                )

                plot_results(test_data)
