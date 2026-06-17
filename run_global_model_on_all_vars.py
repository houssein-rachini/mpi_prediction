"""
run_global_model_on_all_vars.py

Runs the global DNN+XGBoost ensemble (models/global/) on
adm2_vars_9countries/adm2_all_vars.csv for all years (2015-2024).

Saves two prediction columns per row:
  predicted_MPI_no_ratio   -- ndvi_lst_ratio forced to 0.0
  predicted_MPI_with_ratio -- ndvi_lst_ratio computed as Median_NDVI / Mean_LST

Output: adm2_predictions_global_model.csv
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.models import load_model
from pathlib import Path
import xgboost as xgb

BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / "global"
IN_CSV     = BASE_DIR / "adm2_vars_9countries" / "adm2_all_vars.csv"
OUT_CSV    = BASE_DIR / "adm2_predictions_global_model.csv"

ALPHA = 0.4


def load_models() -> dict:
    custom = {"mse": MeanSquaredError(), "mae": MeanAbsoluteError(),
               "rmse": tf.keras.metrics.RootMeanSquaredError()}
    dnn   = load_model(MODELS_DIR / "trained_ensemble_xgb_dnn_model.h5",
                       custom_objects=custom)
    xgb_m = xgb.XGBRegressor()
    xgb_m.load_model(MODELS_DIR / "trained_ensemble_xgb_model.json")
    scaler = joblib.load(MODELS_DIR / "ensemble_scaler.pkl")
    return {"dnn": dnn, "xgb": xgb_m, "scaler": scaler}


def _scale(df_input: pd.DataFrame, scaler, use_ratio: bool) -> np.ndarray:
    feat_names = scaler.feature_names_in_
    df = df_input.copy()
    for col in feat_names:
        if col not in df.columns:
            df[col] = 0.0
    if not use_ratio:
        df["ndvi_lst_ratio"] = 0.0
    return scaler.transform(df[feat_names])


def _predict(df_feat: pd.DataFrame, models: dict, use_ratio: bool) -> np.ndarray:
    X     = _scale(df_feat, models["scaler"], use_ratio)
    p_dnn = models["dnn"].predict(X, verbose=0).flatten()
    p_xgb = models["xgb"].predict(X)
    return np.clip(ALPHA * p_dnn + (1 - ALPHA) * p_xgb, 0, 1)


def main() -> None:
    print(f"Loading {IN_CSV.name} ...")
    df = pd.read_csv(IN_CSV, encoding="utf-8")
    print(f"  {len(df)} rows | {df['Country'].nunique()} countries | years {df['Year'].min()}–{df['Year'].max()}")

    # Compute actual ndvi_lst_ratio (used for with_ratio pass)
    den = df["Mean_LST"].replace(0, np.nan)
    df["ndvi_lst_ratio"] = (df["Median_NDVI"] / den).fillna(0.0)

    print("\nLoading global models ...")
    models = load_models()
    print("  -> Models loaded.")

    feat_names = list(models["scaler"].feature_names_in_)
    base_feats = [f for f in feat_names if f != "ndvi_lst_ratio"]
    df_feat = df.dropna(subset=base_feats)
    n_drop = len(df) - len(df_feat)
    if n_drop:
        print(f"  {n_drop} rows dropped (null features)")

    print(f"\nPredicting {len(df_feat)} rows ...")

    out = df_feat[["Country", "ADM1_CODE", "ADM2_CODE", "ADM2_NAME", "Year"]].copy()

    print("  pass 1/2: no_ratio (ndvi_lst_ratio = 0.0) ...")
    out["predicted_MPI_no_ratio"]   = _predict(df_feat, models, use_ratio=False)

    print("  pass 2/2: with_ratio (actual Median_NDVI / Mean_LST) ...")
    out["predicted_MPI_with_ratio"] = _predict(df_feat, models, use_ratio=True)

    out = out.sort_values(["Country", "ADM2_CODE", "Year"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"\nSaved {OUT_CSV.name}  ({len(out)} rows)")
    for col in ["predicted_MPI_no_ratio", "predicted_MPI_with_ratio"]:
        label = col.replace("predicted_MPI_", "")
        s = out[col]
        print(f"\n  [{label}]  mean={s.mean():.4f}  std={s.std():.4f}  min={s.min():.4f}  max={s.max():.4f}")
        print(out.groupby("Country")[col].mean().sort_values().round(4).rename("mean_pred").to_string())


if __name__ == "__main__":
    main()
