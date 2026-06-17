"""
Build adm2_predictions_training_countries.csv

Loads GEE exports from gee_export_adm2_9countries.py (Drive folder
gaul9_adm2_vars_500m), merges features, runs locally trained models.

Countries : Bosnia and Herzegovina, Egypt, Jordan, Kyrgyzstan, Montenegro,
            Morocco, Tajikistan, Tunisia, Turkey
Years     : 2015-2024

Input CSV: adm2_vars_9countries/adm2_all_vars.csv

Output columns:
  country, adm1_code, adm2_code, adm2_name, year,
  predicted_MPI, population, pred_DNN_only, pred_XGB_only
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.models import load_model
import xgboost as xgb

BASE_DIR  = Path(__file__).resolve().parent
CSV_DIR   = BASE_DIR / "adm2_vars_9countries"
OUT_FILE  = BASE_DIR / "adm2_predictions_training_countries.csv"

ALL_VARS_CSV = CSV_DIR / "adm2_all_vars.csv"

FEATURES = [
    "Mean_GPP", "StdDev_GPP", "Median_Pop", "StdDev_Pop",
    "Mean_LST", "StdDev_LST", "Mean_NTL", "StdDev_NTL", "Sum_NTL",
    "Median_NDVI", "StdDev_NDVI",
]


# ─────────────────────── data loading ─────────────────────────────────────────

def load_features() -> pd.DataFrame:
    if not ALL_VARS_CSV.exists():
        raise FileNotFoundError(
            f"{ALL_VARS_CSV.name} not found.\n"
            f"Run gee_export_adm2_9countries.py --start-tasks, download the 5 CSVs "
            f"from Drive folder 'gaul9_adm2_vars_500m' to {CSV_DIR}, "
            f"then merge them into adm2_all_vars.csv."
        )

    df = pd.read_csv(ALL_VARS_CSV, encoding="utf-8")
    df["ADM2_CODE"] = pd.to_numeric(df["ADM2_CODE"], errors="coerce").astype("Int64")
    df["Year"]      = pd.to_numeric(df["Year"],      errors="coerce").astype("Int64")

    missing_feats = [f for f in FEATURES if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns: {missing_feats}")

    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"  Loaded: {len(df)} rows | "
          f"{df['Country'].nunique()} countries | "
          f"years {df['Year'].min()}–{df['Year'].max()}")
    return df


# ─────────────────────── model loading ────────────────────────────────────────

def load_models() -> dict:
    custom = {
        "mse":  MeanSquaredError(),
        "mae":  MeanAbsoluteError(),
        "rmse": tf.keras.metrics.RootMeanSquaredError(),
    }
    dnn_model     = load_model(BASE_DIR / "trained_dnn_model.h5",              custom_objects=custom)
    dnn_scaler    = joblib.load(BASE_DIR / "dnn_scaler.pkl")
    ml_model      = joblib.load(BASE_DIR / "trained_ml_model.pkl")
    ml_scaler     = joblib.load(BASE_DIR / "ml_scaler.pkl")
    ens_dnn_model = load_model(BASE_DIR / "trained_ensemble_xgb_dnn_model.h5", custom_objects=custom)
    ens_xgb_model = xgb.XGBRegressor()
    ens_xgb_model.load_model(BASE_DIR / "trained_ensemble_xgb_model.json")
    ens_scaler    = joblib.load(BASE_DIR / "ensemble_scaler.pkl")
    return {
        "dnn_model": dnn_model, "dnn_scaler": dnn_scaler,
        "ml_model":  ml_model,  "ml_scaler":  ml_scaler,
        "ens_dnn":   ens_dnn_model, "ens_xgb": ens_xgb_model, "ens_scaler": ens_scaler,
    }


def _scale_and_fill(df_input: pd.DataFrame, scaler) -> np.ndarray:
    feat_names = scaler.feature_names_in_
    df = df_input.copy()
    for col in feat_names:
        if col not in df.columns:
            df[col] = 0.0
    return scaler.transform(df[feat_names])


# ─────────────────────── main ─────────────────────────────────────────────────

def main():
    print("Loading GEE export CSVs ...")
    df = load_features()

    print("Loading locally trained models ...")
    models = load_models()
    print("  -> Models loaded.")

    # ── Vectorised batch predictions (no row-by-row loop) ─────────────────────
    print("Predicting (batch) ...")

    X_dnn = _scale_and_fill(df, models["dnn_scaler"])
    X_ml  = _scale_and_fill(df, models["ml_scaler"])
    X_ens = _scale_and_fill(df, models["ens_scaler"])

    pred_dnn = np.clip(models["dnn_model"].predict(X_dnn, verbose=0).flatten(), 0, 1)
    pred_xgb = np.clip(models["ml_model"].predict(X_ml), 0, 1)
    p_ens_dnn = models["ens_dnn"].predict(X_ens, verbose=0).flatten()
    p_ens_xgb = models["ens_xgb"].predict(X_ens)
    pred_ens  = np.clip(0.4 * p_ens_dnn + 0.6 * p_ens_xgb, 0, 1)

    df_out = pd.DataFrame({
        "country":       df["Country"].values,
        "adm1_code":     df["ADM1_CODE"].values,
        "adm2_code":     df["ADM2_CODE"].values,
        "adm2_name":     df["ADM2_NAME"].values,
        "year":          df["Year"].values,
        "predicted_MPI": pred_ens,
        "population":    df["Total_Pop"].values,
        "pred_DNN_only": pred_dnn,
        "pred_XGB_only": pred_xgb,
    })

    df_out.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\nDone. Saved {OUT_FILE.name}")
    print(f"  {len(df_out)} rows | {df_out['country'].nunique()} countries | "
          f"years {df_out['year'].min()}–{df_out['year'].max()}")
    print(f"  {df_out['country'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
