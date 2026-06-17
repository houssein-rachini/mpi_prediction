"""
Build timeseries_split_predictions.csv

DNN+XGB ensemble, 5-fold TimeSeriesSplit (year-sorted).
Source: Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv (500m masked)

For each out-of-fold test observation collects:
  country, adm1_code, adm1_name, year,
  observed_MPI_t, observed_MPI_t_minus_1, predicted_MPI_t

observed_MPI_t_minus_1 is the same district's MPI at year t-1.
adm1_code is joined from mpi_training_panel.csv on country+adm1_name.

Use these columns to compute:
  - Δ-RMSE : RMSE on (predicted_MPI_t - observed_MPI_t_minus_1)
             vs (observed_MPI_t - observed_MPI_t_minus_1)
  - within-district year-on-year autocorrelation
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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import xgboost as xgb

BASE_DIR   = Path(__file__).resolve().parent
SOURCE     = BASE_DIR / "Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv"
ADM1_CODES = BASE_DIR / "training_adm1_codes.csv"
OUT_FILE   = BASE_DIR / "timeseries_split_predictions.csv"

FEATURES = [
    "Mean_GPP", "StdDev_GPP", "Median_Pop", "StdDev_Pop",
    "Mean_LST", "StdDev_LST", "Mean_NTL", "StdDev_NTL", "Sum_NTL",
    "Median_NDVI", "StdDev_NDVI", "ndvi_lst_ratio",
]

N_SPLITS     = 5
RANDOM_STATE = 42

# Ensemble defaults (from ensemble_training.py / build_cv_fold_metrics.py)
ENS_LR           = 0.0005
ENS_WEIGHT_DECAY = 1e-6
ENS_BATCH_SIZE   = 128
ENS_PATIENCE     = 20
ENS_EPOCHS       = 300
ALPHA_ENS        = 0.4

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


# ─────────────────────── helpers ──────────────────────────────────────────────

def _build_dnn(input_dim: int) -> Sequential:
    lr_schedule = CosineDecay(initial_learning_rate=ENS_LR, decay_steps=10000, alpha=0.0001)
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
        optimizer=AdamW(learning_rate=lr_schedule, weight_decay=ENS_WEIGHT_DECAY),
        loss=Huber(delta=1.0), metrics=["mae"],
    )
    return model


def _fit_ensemble(X_tr, y_tr, X_val, y_val):
    """Train DNN+XGB ensemble with early stopping on ensemble val loss."""
    loss_fn = Huber(delta=1.0)

    xgb_model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=6, min_child_weight=2,
        random_state=RANDOM_STATE, verbosity=0,
    )
    xgb_model.fit(X_tr, y_tr)
    p_xgb_val = xgb_model.predict(X_val)  # constant — compute once

    dnn = _build_dnn(X_tr.shape[1])
    best_val_loss  = float("inf")
    best_weights   = dnn.get_weights()
    patience_count = 0

    for _ in range(ENS_EPOCHS):
        dnn.fit(X_tr, y_tr, epochs=1, batch_size=ENS_BATCH_SIZE,
                validation_data=(X_val, y_val), verbose=0)
        p_dnn_val    = dnn.predict(X_val, verbose=0).flatten()
        p_ens_val    = ALPHA_ENS * p_dnn_val + (1 - ALPHA_ENS) * p_xgb_val
        ens_val_loss = float(loss_fn(y_val, p_ens_val).numpy())
        if ens_val_loss < best_val_loss:
            best_val_loss  = ens_val_loss
            best_weights   = dnn.get_weights()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= ENS_PATIENCE:
                break

    dnn.set_weights(best_weights)
    return dnn, xgb_model


def _load_source() -> pd.DataFrame:
    df = pd.read_csv(SOURCE, encoding="utf-8")
    if "ndvi_lst_ratio" not in df.columns:
        lst_col  = next((c for c in df.columns if "LST" in c and "Mean" in c), None)
        ndvi_col = next((c for c in df.columns if "NDVI" in c and "Median" in c), None)
        if lst_col and ndvi_col:
            den = df[lst_col]
            df["ndvi_lst_ratio"] = (df[ndvi_col] / den).where(den != 0)

    target_col = next((c for c in ["MPI", "observed_MPI"] if c in df.columns), None)
    df = df.rename(columns={target_col: "MPI"})

    # normalise key columns
    df = df.rename(columns={"Country": "country", "Region": "adm1_name", "Year": "year"})
    df["country"]   = df["country"].str.strip()
    df["adm1_name"] = df["adm1_name"].str.strip()

    required = FEATURES + ["MPI", "country", "adm1_name", "year"]
    df = df.dropna(subset=FEATURES + ["MPI"]).reset_index(drop=True)
    return df


def _load_adm1_codes() -> pd.DataFrame | None:
    """Return lookup from training_adm1_codes.csv (built by fetch_training_adm1_codes.py)."""
    if not ADM1_CODES.exists():
        return None
    codes = pd.read_csv(ADM1_CODES, encoding="utf-8")
    codes["_ck"] = codes["country"].str.strip().str.lower()
    codes["_ak"] = codes["adm1_name"].str.strip().str.lower()
    return codes[["_ck", "_ak", "adm1_code"]].drop_duplicates()


# ─────────────────────── main ─────────────────────────────────────────────────

def main():
    print(f"Loading {SOURCE.name} ...")
    df = _load_source()
    df_sorted = df.sort_values("year").reset_index(drop=True)
    print(f"  -> {len(df_sorted)} rows | years: {sorted(df_sorted['year'].unique())}")

    X = df_sorted[FEATURES].values
    y = df_sorted["MPI"].values

    tss    = TimeSeriesSplit(n_splits=N_SPLITS)
    splits = list(tss.split(X))

    all_records = []

    for fold_idx, (tr_idx, te_idx) in enumerate(tqdm(splits, desc="TS folds", unit="fold")):
        print(f"\n  Fold {fold_idx+1}: train={len(tr_idx)}, test={len(te_idx)} | "
              f"test years={sorted(df_sorted.iloc[te_idx]['year'].unique())}")

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X[tr_idx])
        X_te_s = scaler.transform(X[te_idx])
        y_tr   = y[tr_idx]

        # internal 80/20 val split
        X_dnn_tr, X_dnn_val, y_dnn_tr, y_dnn_val = train_test_split(
            X_tr_s, y_tr, test_size=0.2, random_state=RANDOM_STATE
        )

        dnn, xgb_model = _fit_ensemble(X_dnn_tr, y_dnn_tr, X_dnn_val, y_dnn_val)

        p_dnn  = dnn.predict(X_te_s, verbose=0).flatten()
        p_xgb  = xgb_model.predict(X_te_s)
        pred   = np.clip(ALPHA_ENS * p_dnn + (1 - ALPHA_ENS) * p_xgb, 0, 1)
        tf.keras.backend.clear_session()

        te_df = df_sorted.iloc[te_idx].copy()
        te_df["predicted_MPI_t"] = pred
        te_df["observed_MPI_t"]  = y[te_idx]
        te_df["fold"]            = fold_idx + 1

        level_rmse = float(np.sqrt(mean_squared_error(y[te_idx], pred)))
        print(f"  Level RMSE: {level_rmse:.4f}")

        all_records.append(te_df[["country", "adm1_name", "year",
                                   "observed_MPI_t", "predicted_MPI_t", "fold"]])

    predictions = pd.concat(all_records, ignore_index=True)

    # ── Join t-1 MPI ──────────────────────────────────────────────────────────
    mpi_lookup = (
        df_sorted[["country", "adm1_name", "year", "MPI"]]
        .rename(columns={"MPI": "observed_MPI_t_minus_1", "year": "year_tm1"})
    )
    mpi_lookup["year"] = mpi_lookup["year_tm1"] + 1
    predictions = predictions.merge(
        mpi_lookup[["country", "adm1_name", "year", "observed_MPI_t_minus_1"]],
        on=["country", "adm1_name", "year"], how="left",
    )

    # ── Join adm1_code from panel ─────────────────────────────────────────────
    codes = _load_adm1_codes()
    if codes is not None:
        predictions["_ck"] = predictions["country"].str.strip().str.lower()
        predictions["_ak"] = predictions["adm1_name"].str.strip().str.lower()
        predictions = predictions.merge(codes, on=["_ck", "_ak"], how="left")
        predictions = predictions.drop(columns=["_ck", "_ak"])
    else:
        predictions["adm1_code"] = pd.NA
        print("  Warning: mpi_training_panel.csv not found — adm1_code will be NaN.")

    # ── Reorder & save ────────────────────────────────────────────────────────
    col_order = [
        "country", "adm1_code", "adm1_name", "year", "fold",
        "observed_MPI_t", "observed_MPI_t_minus_1", "predicted_MPI_t",
    ]
    predictions = predictions[[c for c in col_order if c in predictions.columns]]

    with_tm1 = predictions["observed_MPI_t_minus_1"].notna().sum()
    print(f"\nRows with t-1 MPI: {with_tm1}/{len(predictions)} "
          f"({with_tm1/len(predictions)*100:.1f}%)")

    # ── Quick diagnostics ─────────────────────────────────────────────────────
    delta = predictions.dropna(subset=["observed_MPI_t_minus_1"])
    obs_delta  = delta["observed_MPI_t"]  - delta["observed_MPI_t_minus_1"]
    pred_delta = delta["predicted_MPI_t"] - delta["observed_MPI_t_minus_1"]
    delta_rmse = float(np.sqrt(mean_squared_error(obs_delta, pred_delta)))

    autocorr = (
        delta.groupby(["country", "adm1_name"])
        .apply(lambda g: g["observed_MPI_t"].corr(g["observed_MPI_t_minus_1"]))
        .dropna()
        .mean()
    )

    print(f"Δ-MPI RMSE        : {delta_rmse:.4f}")
    print(f"Mean within-district autocorr: {autocorr:.4f}")

    predictions.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_FILE.name} ({len(predictions)} rows, {len(predictions.columns)} cols).")


if __name__ == "__main__":
    main()
