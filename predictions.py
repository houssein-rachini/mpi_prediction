import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ee_auth import initialize_earth_engine

initialize_earth_engine()

# Model Paths
MODEL_PATHS = {
    "DNN": "trained_dnn_model.h5",
    "ML": "trained_ml_model.pkl",
    "DNN+RF": "trained_ensemble_rf_dnn_model.h5",
    "DNN+XGBoost": "trained_ensemble_xgb_dnn_model.h5",
    "DNN+LightGBM": "trained_ensemble_lgbm_dnn_model.h5",
    "DNN+KNN": "trained_ensemble_knn_dnn_model.h5",
    "XGBQ05": "trained_xgb_quantile_q05.json",
    "XGBQ50": "trained_xgb_quantile_q50.json",
    "XGBQ95": "trained_xgb_quantile_q95.json",
}

SCALER_PATHS = {
    "DNN": "dnn_scaler.pkl",
    "ML": "ml_scaler.pkl",
    "Ensemble": "ensemble_scaler.pkl",
    "Quantile": "quantile_scaler.pkl",
}

PRETRAINED_MODELS_PATHS = {
    "DNN": "models/global/trained_dnn_model.h5",
    "ML": "models/global/trained_ml_model.pkl",
    "DNN+RF": "models/global/trained_ensemble_rf_dnn_model.h5",
    "DNN+XGBoost": "models/global/trained_ensemble_xgb_dnn_model.h5",
    "DNN+LightGBM": "models/global/trained_ensemble_lgbm_dnn_model.h5",
    "DNN+KNN": "models/global/trained_ensemble_knn_dnn_model.h5",
    "XGBoost": "models/global/trained_ensemble_xgb_model.json",
    "LightGBM": "models/global/trained_ensemble_lgbm_model.pkl",
    "RF": "models/global/trained_ensemble_rf_model.pkl",
    "KNN": "models/global/trained_ensemble_knn_model.pkl",
    "XGBQ05": "models/global/trained_xgb_quantile_q05.json",
    "XGBQ50": "models/global/trained_xgb_quantile_q50.json",
    "XGBQ95": "models/global/trained_xgb_quantile_q95.json",
}

PRETRAINED_SCALERS_PATHS = {
    "DNN": "models/global/dnn_scaler.pkl",
    "ML": "models/global/ml_scaler.pkl",
    "Ensemble": "models/global/ensemble_scaler.pkl",
    "Quantile": "models/global/quantile_scaler.pkl",
}
STACKED_LOCAL_DIR = "models/stacked"
STACKED_PRETRAINED_DIR = "models/global/stacked"

# ========== Preprocessing ==========


def preprocess_data(test_data, scaler):
    test_data = test_data.copy()
    if "Mean_LST" in test_data.columns and "Median_NDVI" in test_data.columns:
        lst = test_data["Mean_LST"].replace(0, np.nan)
        test_data["ndvi_lst_ratio"] = test_data["Median_NDVI"] / lst
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
def load_quantile_scaler(USE_PRETRAINED):
    path = (
        PRETRAINED_SCALERS_PATHS["Quantile"]
        if USE_PRETRAINED
        else SCALER_PATHS["Quantile"]
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
    elif model_type == "DNN+LightGBM":
        base_path = (
            PRETRAINED_MODELS_PATHS["LightGBM"]
            if USE_PRETRAINED
            else "trained_ensemble_lgbm_model.pkl"
        )
        base_model = joblib.load(base_path)
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


@st.cache_resource
def load_quantile_models(USE_PRETRAINED):
    if USE_PRETRAINED:
        q05_path = PRETRAINED_MODELS_PATHS["XGBQ05"]
        q50_path = PRETRAINED_MODELS_PATHS["XGBQ50"]
        q95_path = PRETRAINED_MODELS_PATHS["XGBQ95"]
    else:
        q05_path = MODEL_PATHS["XGBQ05"]
        q50_path = MODEL_PATHS["XGBQ50"]
        q95_path = MODEL_PATHS["XGBQ95"]

    q05 = xgb.XGBRegressor()
    q50 = xgb.XGBRegressor()
    q95 = xgb.XGBRegressor()
    q05.load_model(q05_path)
    q50.load_model(q50_path)
    q95.load_model(q95_path)
    return {"q05": q05, "q50": q50, "q95": q95}


@st.cache_resource
def load_stacked_artifacts(USE_PRETRAINED):
    preferred_dir = STACKED_PRETRAINED_DIR if USE_PRETRAINED else STACKED_LOCAL_DIR
    fallback_dir = STACKED_LOCAL_DIR if USE_PRETRAINED else STACKED_PRETRAINED_DIR

    model_dir = preferred_dir if os.path.isdir(preferred_dir) else fallback_dir
    metadata_path = os.path.join(model_dir, "metadata.json")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    meta_model_path = os.path.join(model_dir, "meta_model.pkl")

    if not (
        os.path.exists(metadata_path)
        and os.path.exists(scaler_path)
        and os.path.exists(meta_model_path)
    ):
        raise FileNotFoundError(
            f"Stacked model artifacts missing in '{model_dir}'. Expected metadata.json, scaler.pkl, meta_model.pkl."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    scaler = joblib.load(scaler_path)
    meta_model = joblib.load(meta_model_path)
    meta_scaler_path = os.path.join(model_dir, "meta_scaler.pkl")
    meta_scaler = joblib.load(meta_scaler_path) if os.path.exists(meta_scaler_path) else None

    include_dnn = bool(metadata.get("include_dnn", True))
    dnn_model = None
    if include_dnn:
        dnn_path = os.path.join(model_dir, "dnn_model.h5")
        if not os.path.exists(dnn_path):
            raise FileNotFoundError(
                f"DNN is enabled in stacked metadata but '{dnn_path}' was not found."
            )
        dnn_model = load_model(
            dnn_path,
            custom_objects={
                "mse": MeanSquaredError(),
                "mae": MeanAbsoluteError(),
                "rmse": tf.keras.metrics.RootMeanSquaredError(),
            },
        )

    saved_files = metadata.get("saved_base_model_files", {})
    base_model_order = metadata.get("base_model_order", [])
    base_models = {}
    for model_name in base_model_order:
        filename = saved_files.get(model_name)
        if not filename:
            raise FileNotFoundError(
                f"Missing saved file mapping for base model '{model_name}' in stacked metadata."
            )
        model_path = os.path.join(model_dir, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Base model file not found: {model_path}")
        base_models[model_name] = joblib.load(model_path)

    return {
        "model_dir": model_dir,
        "metadata": metadata,
        "scaler": scaler,
        "meta_model": meta_model,
        "meta_scaler": meta_scaler,
        "dnn_model": dnn_model,
        "base_models": base_models,
        "base_model_order": base_model_order,
    }


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


def predict_quantile_fast(test_data, quantile_models, scaler):
    test_data_scaled = preprocess_data(test_data, scaler)
    lower = quantile_models["q05"].predict(test_data_scaled)
    median = quantile_models["q50"].predict(test_data_scaled)
    upper = quantile_models["q95"].predict(test_data_scaled)

    # Guard against quantile crossing.
    lo = np.minimum(lower, upper)
    hi = np.maximum(lower, upper)

    lo = np.clip(lo, 0, 1)
    median = np.clip(median, 0, 1)
    hi = np.clip(hi, 0, 1)
    width = np.maximum(hi - lo, 0)
    return {"lower": lo, "median": median, "upper": hi, "width": width}


def _stacked_meta_input(pred_map, metadata):
    meta_feature_order = metadata.get("meta_feature_order")
    include_dnn = bool(metadata.get("include_dnn", True))
    use_abs_diff = bool(metadata.get("use_abs_diff", False))
    ensemble_feature_alphas = metadata.get("ensemble_feature_alphas", {})

    if not meta_feature_order:
        meta_feature_order = []
        for model_name in metadata.get("meta_source_order", metadata.get("base_model_order", [])):
            meta_feature_order.append(f"oof_{model_name.lower().replace(' ', '_')}")
        if use_abs_diff and include_dnn and "XGBoost" in metadata.get("base_model_order", []):
            meta_feature_order.append("abs_diff_dnn_xgboost")

    model_lookup = {
        f"oof_{name.lower().replace(' ', '_')}": name
        for name in metadata.get("base_model_order", [])
    }

    cols = []
    for feat_name in meta_feature_order:
        if feat_name == "oof_dnn":
            if "DNN" not in pred_map:
                raise ValueError("Stacked metadata expects DNN predictions, but DNN is unavailable.")
            cols.append(pred_map["DNN"])
        elif feat_name == "abs_diff_dnn_xgboost":
            if "DNN" not in pred_map or "XGBoost" not in pred_map:
                raise ValueError(
                    "Stacked metadata expects abs(DNN-XGBoost), but required predictions are unavailable."
                )
            cols.append(np.abs(pred_map["DNN"] - pred_map["XGBoost"]))
        elif feat_name == "oof_dnn_xgboost_ensemble":
            if "DNN" not in pred_map or "XGBoost" not in pred_map:
                raise ValueError(
                    "Stacked metadata expects DNN+XGBoost ensemble feature, but required predictions are unavailable."
                )
            alpha = float(ensemble_feature_alphas.get("DNN+XGBoost Ensemble", 0.4))
            cols.append(alpha * pred_map["DNN"] + (1.0 - alpha) * pred_map["XGBoost"])
        elif feat_name == "oof_dnn_random_forest_ensemble":
            if "DNN" not in pred_map or "Random Forest" not in pred_map:
                raise ValueError(
                    "Stacked metadata expects DNN+Random Forest ensemble feature, but required predictions are unavailable."
                )
            alpha = float(
                ensemble_feature_alphas.get("DNN+Random Forest Ensemble", 0.4)
            )
            cols.append(
                alpha * pred_map["DNN"] + (1.0 - alpha) * pred_map["Random Forest"]
            )
        elif feat_name == "oof_dnn_knn_ensemble":
            if "DNN" not in pred_map or "KNN Regressor" not in pred_map:
                raise ValueError(
                    "Stacked metadata expects DNN+KNN ensemble feature, but required predictions are unavailable."
                )
            alpha = float(ensemble_feature_alphas.get("DNN+KNN Ensemble", 0.4))
            cols.append(
                alpha * pred_map["DNN"] + (1.0 - alpha) * pred_map["KNN Regressor"]
            )
        elif feat_name.startswith("oof_") and feat_name in model_lookup:
            base_name = model_lookup[feat_name]
            cols.append(pred_map[base_name])
        else:
            raise ValueError(f"Unsupported stacked meta-feature in metadata: {feat_name}")

    return np.column_stack(cols)


def predict_stacked_fast(test_data, stacked_artifacts):
    scaler = stacked_artifacts["scaler"]
    metadata = stacked_artifacts["metadata"]
    dnn_model = stacked_artifacts["dnn_model"]
    base_models = stacked_artifacts["base_models"]
    meta_model = stacked_artifacts["meta_model"]
    meta_scaler = stacked_artifacts.get("meta_scaler")

    test_data_scaled = preprocess_data(test_data, scaler)
    pred_map = {}

    if bool(metadata.get("include_dnn", True)):
        y_pred_dnn = dnn_model.predict(test_data_scaled, verbose=0).reshape(-1)
        pred_map["DNN"] = y_pred_dnn

    for model_name in stacked_artifacts["base_model_order"]:
        pred_map[model_name] = base_models[model_name].predict(test_data_scaled)

    meta_input = _stacked_meta_input(pred_map, metadata)
    if meta_scaler is not None:
        meta_input = meta_scaler.transform(meta_input)
    return np.clip(meta_model.predict(meta_input), 0, 1)


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
        [
            "DNN",
            "ML",
            "DNN+RF",
            "DNN+XGBoost",
            "DNN+LightGBM",
            "DNN+KNN",
            "XGB-Quantile",
            "Stacked",
        ],
    )

    alpha = None
    if model_choice in ["DNN+RF", "DNN+XGBoost", "DNN+LightGBM", "DNN+KNN"]:
        alpha = st.slider(
            "Ensemble Weight (DNN Contribution)", 0.0, 1.0, 0.15, key="testing_alpha"
        )

    use_pretrained_model = st.checkbox(
        "Use Pre-trained Model", value=True, key="predictions_tab_use_pretrained"
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
                    dnn_model = load_dnn_model(USE_PRETRAINED=use_pretrained_model)
                    scaler = load_dnn_scaler(USE_PRETRAINED=use_pretrained_model)
                    predictions = predict_dnn_fast(test_data, dnn_model, scaler)
                    output_file = "test_results_dnn.csv"

                elif model_choice == "ML":
                    ml_model = load_ml_model(USE_PRETRAINED=use_pretrained_model)
                    scaler = load_ml_scaler(USE_PRETRAINED=use_pretrained_model)
                    predictions = predict_ml_fast(test_data, ml_model, scaler)
                    output_file = "test_results_ml.csv"

                elif model_choice == "Stacked":
                    stacked_artifacts = load_stacked_artifacts(USE_PRETRAINED=use_pretrained_model)
                    predictions = predict_stacked_fast(test_data, stacked_artifacts)
                    output_file = "test_results_stacked.csv"

                elif model_choice == "XGB-Quantile":
                    quantile_models = load_quantile_models(USE_PRETRAINED=use_pretrained_model)
                    scaler = load_quantile_scaler(USE_PRETRAINED=use_pretrained_model)
                    quant_pred = predict_quantile_fast(test_data, quantile_models, scaler)
                    predictions = quant_pred["median"]
                    test_data["Predicted_MPI_Lower_90"] = quant_pred["lower"]
                    test_data["Predicted_MPI_Upper_90"] = quant_pred["upper"]
                    test_data["Predicted_MPI_Interval_Width"] = quant_pred["width"]
                    output_file = "test_results_xgb_quantile.csv"

                else:
                    dnn_model, base_model = load_ensemble_models(
                        model_choice, USE_PRETRAINED=use_pretrained_model
                    )
                    scaler = load_ensemble_scaler(USE_PRETRAINED=use_pretrained_model)
                    predictions = predict_ensemble_fast(
                        test_data, dnn_model, base_model, scaler, alpha
                    )
                    if model_choice == "DNN+RF":
                        output_file = "test_results_ensemble_rf.csv"
                    elif model_choice == "DNN+XGBoost":
                        output_file = "test_results_ensemble_xgb.csv"
                    elif model_choice == "DNN+LightGBM":
                        output_file = "test_results_ensemble_lgbm.csv"
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
