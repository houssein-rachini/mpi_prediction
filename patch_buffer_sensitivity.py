"""
Patch buffer_sensitivity_folds.csv:
 1. Add mask=yes to the existing 30 masked rows.
 2. Run DNN+XGB 5-fold CV on the unmasked dataset.
 3. Append 5 unmasked rows (mask=no, buffer_m=None).
"""

from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

BASE_DIR      = Path(__file__).resolve().parent
OUT_FILE      = BASE_DIR / "buffer_sensitivity_folds.csv"
UNMASKED_FILE = BASE_DIR / "unmasked_Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv"

FEATURES = [
    "Mean_GPP", "StdDev_GPP", "Median_Pop", "StdDev_Pop",
    "Mean_LST", "StdDev_LST", "Mean_NTL", "StdDev_NTL", "Sum_NTL",
    "Median_NDVI", "StdDev_NDVI", "ndvi_lst_ratio",
]

N_SPLITS     = 5
RANDOM_STATE = 42
LR           = 0.001
WEIGHT_DECAY = 1e-5
BATCH_SIZE   = 128
PATIENCE     = 10
EPOCHS       = 200
ALPHA_ENS    = 0.4

DEFAULT_LAYERS = [
    {"type": "Dense",              "units": 256, "activation": "relu"},
    {"type": "BatchNormalization"},
    {"type": "Dropout",            "rate": 0.15},
    {"type": "Dense",              "units": 128, "activation": "relu"},
    {"type": "BatchNormalization"},
    {"type": "Dropout",            "rate": 0.10},
    {"type": "Dense",              "units": 64,  "activation": "relu"},
    {"type": "BatchNormalization"},
    {"type": "Dense",              "units": 32,  "activation": "relu"},
    {"type": "Dense",              "units": 1,   "activation": "relu"},
]


def _build_dnn(input_dim: int) -> Sequential:
    lr_schedule = CosineDecay(initial_learning_rate=LR, decay_steps=10000, alpha=0.0005)
    model = Sequential()
    for i, layer in enumerate(DEFAULT_LAYERS):
        if layer["type"] == "Dense":
            model.add(Dense(layer["units"], activation=layer["activation"],
                            input_shape=(input_dim,) if i == 0 else ()))
        elif layer["type"] == "BatchNormalization":
            model.add(BatchNormalization())
        elif layer["type"] == "Dropout":
            model.add(Dropout(layer["rate"]))
    model.compile(
        optimizer=AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY),
        loss=Huber(delta=1.0),
        metrics=["mae"],
    )
    return model


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, encoding="utf-8")
    if "ndvi_lst_ratio" not in df.columns:
        lst_col  = next((c for c in df.columns if "LST" in c and "Mean" in c), None)
        ndvi_col = next((c for c in df.columns if "NDVI" in c and "Median" in c), None)
        if lst_col and ndvi_col:
            den = df[lst_col]
            df["ndvi_lst_ratio"] = (df[ndvi_col] / den).where(den != 0)
    target_col = next((c for c in ["MPI", "observed_MPI"] if c in df.columns), None)
    df = df.rename(columns={target_col: "MPI"})
    df = df.dropna(subset=FEATURES + ["MPI"]).reset_index(drop=True)
    return df[FEATURES].values, df["MPI"].values


def run_fold(X_tr_s, X_te_s, y_tr, y_te):
    dnn = _build_dnn(X_tr_s.shape[1])
    cb  = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    dnn.fit(X_tr_s, y_tr, validation_data=(X_te_s, y_te),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[cb], verbose=0)
    p_dnn = dnn.predict(X_te_s, verbose=0).flatten()
    tf.keras.backend.clear_session()

    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                  max_depth=5, min_child_weight=1,
                                  random_state=RANDOM_STATE, verbosity=0)
    xgb_model.fit(X_tr_s, y_tr)
    p_xgb = xgb_model.predict(X_te_s)

    pred = np.clip(ALPHA_ENS * p_dnn + (1 - ALPHA_ENS) * p_xgb, 0, 1)
    return (r2_score(y_te, pred),
            float(np.sqrt(mean_squared_error(y_te, pred))),
            float(mean_absolute_error(y_te, pred)))


def main():
    # Step 1: load existing 30-row masked CSV and add mask=yes
    existing = pd.read_csv(OUT_FILE, encoding="utf-8")
    if "mask" not in existing.columns:
        existing.insert(0, "mask", "yes")
        print(f"Added mask=yes to {len(existing)} existing rows.")
    else:
        print("mask column already present.")

    # Step 2: run unmasked folds
    print(f"\nmask=no  buffer=N/A  ({UNMASKED_FILE.name})")
    X, y = _load_dataset(UNMASKED_FILE)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    new_rows = []
    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X)):
        print(f"  fold {fold_idx+1}/{N_SPLITS} ...", end=" ", flush=True)
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        r2, rmse, mae = run_fold(X_tr_s, X_te_s, y_tr, y_te)
        print(f"R2={r2:.4f}")
        new_rows.append({
            "mask":       "no",
            "buffer_m":   None,
            "fold_index": fold_idx + 1,
            "R2":         round(r2,   4),
            "RMSE":       round(rmse, 4),
            "MAE":        round(mae,  4),
        })

    # Step 3: append and save
    df_out = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    df_out.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_FILE.name}  ({len(df_out)} rows).")


if __name__ == "__main__":
    main()
