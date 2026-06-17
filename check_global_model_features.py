"""
check_global_model_features.py

Prints all feature information for the global DNN+XGBoost ensemble:
  - Scaler feature names, means, and scales
  - XGBoost feature names and importance
  - DNN architecture summary

Run:
    python check_global_model_features.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.models import load_model
from pathlib import Path
import xgboost as xgb

MODELS_DIR = Path(__file__).resolve().parent / "models" / "global"


def main() -> None:
    # ── Scaler ────────────────────────────────────────────────────────────────
    scaler = joblib.load(MODELS_DIR / "ensemble_scaler.pkl")
    print("=" * 60)
    print("SCALER")
    print("=" * 60)
    print(f"  n_features : {scaler.n_features_in_}")
    print(f"  Features   : {list(scaler.feature_names_in_)}")
    print()
    print(f"  {'Feature':<20} {'Mean':>12} {'Std (scale)':>14}")
    print(f"  {'-'*20} {'-'*12} {'-'*14}")
    for feat, mean, scale in zip(scaler.feature_names_in_, scaler.mean_, scaler.scale_):
        print(f"  {feat:<20} {mean:>12.4f} {scale:>14.4f}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_m = xgb.XGBRegressor()
    xgb_m.load_model(MODELS_DIR / "trained_ensemble_xgb_model.json")
    booster = xgb_m.get_booster()
    importance = booster.get_score(importance_type="gain")

    print()
    print("=" * 60)
    print("XGBOOST")
    print("=" * 60)
    print(f"  n_features    : {xgb_m.n_features_in_}")
    print(f"  Feature names : {booster.feature_names}")
    print()
    print(f"  {'Feature':<20} {'Gain':>12}")
    print(f"  {'-'*20} {'-'*12}")
    for feat, gain in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat:<20} {gain:>12.2f}")

    # ── DNN ───────────────────────────────────────────────────────────────────
    custom = {
        "mse":  MeanSquaredError(),
        "mae":  MeanAbsoluteError(),
        "rmse": tf.keras.metrics.RootMeanSquaredError(),
    }
    dnn = load_model(MODELS_DIR / "trained_ensemble_xgb_dnn_model.h5",
                     custom_objects=custom)

    print()
    print("=" * 60)
    print("DNN ARCHITECTURE")
    print("=" * 60)
    dnn.summary()
    print()
    print(f"  Input shape  : {dnn.input_shape}")
    print(f"  Output shape : {dnn.output_shape}")
    print(f"  Total params : {dnn.count_params():,}")


if __name__ == "__main__":
    main()
