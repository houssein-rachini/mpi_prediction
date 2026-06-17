"""
append_training_predictions.py

Appends four prediction columns to mpi_training_panel.csv:

  predicted_MPI_masked          -- masked model,   ndvi_lst_ratio = 0.0
  predicted_MPI_unmasked        -- unmasked model, ndvi_lst_ratio = 0.0
  predicted_MPI_masked_ratio    -- masked model,   actual NDVI_LSTn_ratio used
  predicted_MPI_unmasked_ratio  -- unmasked model, actual NDVI_LSTn_ratio used

Run:
    python append_training_predictions.py
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

BASE_DIR = Path(__file__).resolve().parent
TRAINING_FILE = BASE_DIR / "mpi_training_panel.csv"

ALPHA = 0.4

COL_RENAME = {
    "Mean_LSTn":      "Mean_LST",
    "StdDev_LSTn":    "StdDev_LST",
    "NDVI_LSTn_ratio":"ndvi_lst_ratio",
    "observed_MPI":   "MPI",
    "country":        "Country",
    "adm1_code":      "ADM1_CODE",
    "adm1_name":      "ADM1_NAME",
}

FEATURES_BASE = [
    "Mean_GPP", "StdDev_GPP", "Median_Pop", "StdDev_Pop",
    "Mean_LST", "StdDev_LST", "Mean_NTL", "StdDev_NTL", "Sum_NTL",
    "Median_NDVI", "StdDev_NDVI",
]


def _load_masked() -> dict:
    custom = {"mse": MeanSquaredError(), "mae": MeanAbsoluteError(),
               "rmse": tf.keras.metrics.RootMeanSquaredError()}
    dnn   = load_model(BASE_DIR / "trained_ensemble_xgb_dnn_model.h5", custom_objects=custom)
    xgb_m = xgb.XGBRegressor()
    xgb_m.load_model(BASE_DIR / "trained_ensemble_xgb_model.json")
    scaler = joblib.load(BASE_DIR / "ensemble_scaler.pkl")
    return {"dnn": dnn, "xgb": xgb_m, "scaler": scaler}


def _load_unmasked() -> dict:
    custom = {"mse": MeanSquaredError(), "mae": MeanAbsoluteError(),
               "rmse": tf.keras.metrics.RootMeanSquaredError()}
    dnn   = load_model(BASE_DIR / "unmasked_ensemble_xgb_dnn_model.h5", custom_objects=custom)
    xgb_m = xgb.XGBRegressor()
    xgb_m.load_model(BASE_DIR / "unmasked_ensemble_xgb_model.json")
    scaler = joblib.load(BASE_DIR / "unmasked_ensemble_scaler.pkl")
    return {"dnn": dnn, "xgb": xgb_m, "scaler": scaler}


def _scale(df_input: pd.DataFrame, scaler, use_ratio: bool) -> np.ndarray:
    """Scale features. If use_ratio=False, ndvi_lst_ratio is forced to 0.0."""
    feat_names = scaler.feature_names_in_
    df = df_input.copy()
    for col in feat_names:
        if col not in df.columns:
            df[col] = 0.0
    if not use_ratio and "ndvi_lst_ratio" in feat_names:
        df["ndvi_lst_ratio"] = 0.0
    return scaler.transform(df[feat_names])


def _predict(df_feat: pd.DataFrame, models: dict, use_ratio: bool) -> np.ndarray:
    X     = _scale(df_feat, models["scaler"], use_ratio)
    p_dnn = models["dnn"].predict(X, verbose=0).flatten()
    p_xgb = models["xgb"].predict(X)
    return np.clip(ALPHA * p_dnn + (1 - ALPHA) * p_xgb, 0, 1)


def _print_stats(label: str, actual: pd.Series, preds: np.ndarray) -> None:
    resid = actual.values - preds
    print(f"\n  [{label}]")
    print(f"  Actual MPI   — mean={actual.mean():.4f}  std={actual.std():.4f}")
    print(f"  Predicted    — mean={preds.mean():.4f}  std={preds.std():.4f}")
    print(f"  Residual     — mean={resid.mean():.4f}  std={resid.std():.4f}")
    print(f"  (positive residual = model underpredicts)")


def main() -> None:
    print(f"Loading {TRAINING_FILE.name} ...")
    df = pd.read_csv(TRAINING_FILE, encoding="utf-8")
    df = df.rename(columns={k: v for k, v in COL_RENAME.items() if k in df.columns})
    print(f"  {len(df)} rows | {df['Country'].nunique()} countries")

    valid = df.dropna(subset=FEATURES_BASE)
    n_drop = len(df) - len(valid)
    if n_drop:
        print(f"  {n_drop} rows dropped (null features)")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading masked models ...")
    masked_models = _load_masked()
    print("  -> Masked models loaded.")

    print("Loading unmasked models ...")
    unmasked_models = _load_unmasked()
    print("  -> Unmasked models loaded.")

    actual = df.loc[valid.index, "MPI"]

    # ── Four prediction passes ────────────────────────────────────────────────
    runs = [
        ("predicted_MPI_masked",         masked_models,   False),
        ("predicted_MPI_unmasked",        unmasked_models, False),
        ("predicted_MPI_masked_ratio",    masked_models,   True),
        ("predicted_MPI_unmasked_ratio",  unmasked_models, True),
    ]

    for col, models, use_ratio in runs:
        label = col.replace("predicted_MPI_", "")
        print(f"\nPredicting ({label}) ...")
        preds = _predict(valid, models, use_ratio)
        df.loc[valid.index, col] = preds
        _print_stats(label, actual, preds)

    # ── Per-country residuals ─────────────────────────────────────────────────
    for col, _, _ in runs:
        label = col.replace("predicted_MPI_", "")
        df2 = df.loc[valid.index].copy()
        df2["residual"] = df2["MPI"].values - df2[col].values
        print(f"\n  Mean residual by country [{label}] (actual - predicted):")
        print(df2.groupby("Country")["residual"].mean().sort_values().round(4).to_string())

    # ── Save (rename back to original column names) ───────────────────────────
    RENAME_BACK = {v: k for k, v in COL_RENAME.items()}
    df = df.rename(columns={k: v for k, v in RENAME_BACK.items() if k in df.columns})
    df.to_csv(TRAINING_FILE, index=False, encoding="utf-8")
    print(f"\nSaved {TRAINING_FILE.name}")
    pred_cols = [c for c in df.columns if c.startswith("predicted_MPI")]
    for c in pred_cols:
        print(f"  {c} → {df[c].notna().sum()} rows")


if __name__ == "__main__":
    main()
