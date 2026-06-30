"""
Run a trained model on the precomputed Turkey feature cache (tr_features_*.csv)
WITHOUT Streamlit or Earth Engine. Replicates the app's preprocessing
(predictions.preprocess_data) and ensemble math (predict_ensemble_fast) exactly.

Usage:
    python predict_tr_features.py
    python predict_tr_features.py --model DNN+XGBoost --alpha 0.15
    python predict_tr_features.py --features tr_features_2012_2024.csv --out tr_predictions.csv

Output: tr_predictions.csv with region_code, year, Predicted_MPI, Predicted_Severe_Poverty_%.
"""
import argparse
import os
import numpy as np
import pandas as pd
import joblib

# Quiet TF before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
import xgboost as xgb

# ── Model registry (local files in this directory) ─────────────────────────────
# Each ensemble entry: (dnn_h5, base_path, base_kind). base_kind: "joblib" | "xgb".
ENSEMBLES = {
    "DNN+LightGBM": ("trained_ensemble_lgbm_dnn_model.h5", "trained_ensemble_lgbm_model.pkl", "joblib"),
    "DNN+XGBoost":  ("trained_ensemble_xgb_dnn_model.h5",  "trained_ensemble_xgb_model.json", "xgb"),
    "DNN+KNN":      ("trained_ensemble_knn_dnn_model.h5",  "trained_ensemble_knn_model.pkl",  "joblib"),
}
ENSEMBLE_SCALER = "ensemble_scaler.pkl"

_KERAS_CUSTOM = {
    "mse": MeanSquaredError(),
    "mae": MeanAbsoluteError(),
    "rmse": tf.keras.metrics.RootMeanSquaredError(),
}


def compute_sev_pov(mpi):
    """Severe poverty % from MPI (same polynomial as the app)."""
    pov = 0.04133 + 34.58 * mpi + 263 * (mpi ** 2) - 180.8 * (mpi ** 3)
    return np.clip(pov, 0, 100)


def preprocess(test_data, scaler):
    """Identical to predictions.preprocess_data: select/order the scaler's
    feature columns, fill any missing with 0, then transform."""
    test_data = test_data.copy()
    # The app recomputes ndvi_lst_ratio only when Median_NDVI is present.
    if "Mean_LST" in test_data.columns and "Median_NDVI" in test_data.columns:
        lst = test_data["Mean_LST"].replace(0, np.nan)
        test_data["ndvi_lst_ratio"] = test_data["Median_NDVI"] / lst
    feature_names = list(scaler.feature_names_in_)
    for col in feature_names:
        if col not in test_data.columns:
            test_data[col] = 0
    return scaler.transform(test_data[feature_names])


def predict(df, model_name, alpha):
    scaler = joblib.load(ENSEMBLE_SCALER)
    X = preprocess(df, scaler)

    dnn_h5, base_path, base_kind = ENSEMBLES[model_name]
    dnn_model = load_model(dnn_h5, custom_objects=_KERAS_CUSTOM)
    if base_kind == "xgb":
        base_model = xgb.XGBRegressor()
        base_model.load_model(base_path)
    else:
        base_model = joblib.load(base_path)

    y_dnn = dnn_model.predict(X, verbose=0).flatten()
    y_base = base_model.predict(X)
    y = alpha * y_dnn + (1 - alpha) * y_base
    return np.clip(y, 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="tr_features_2012_2024.csv")
    p.add_argument("--model", default="DNN+LightGBM", choices=list(ENSEMBLES))
    p.add_argument("--alpha", type=float, default=0.15,
                   help="DNN contribution in the ensemble (app default 0.15)")
    p.add_argument("--out", default="tr_predictions.csv")
    args = p.parse_args()

    df = pd.read_csv(args.features)
    print(f"Loaded {len(df)} rows from {args.features}")
    print(f"Model: {args.model}  |  alpha (DNN weight): {args.alpha}")

    preds = predict(df, args.model, args.alpha)

    out = df[["region_code", "year"]].copy()
    out["Predicted_MPI"] = preds
    out["Predicted_Severe_Poverty_%"] = compute_sev_pov(preds)
    out = out.sort_values(["region_code", "year"]).reset_index(drop=True)
    out.to_csv(args.out, index=False)

    print(f"\nSaved {len(out)} predictions to {args.out}")
    print(f"Predicted_MPI: min={preds.min():.4f}  max={preds.max():.4f}  mean={preds.mean():.4f}")
    print("\nSample:")
    print(out.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
