"""
Build pop_weighted_cv_metrics.csv

DNN+XGBoost ensemble, 4 CV strategies x 5 folds = 20 rows.
Feature aggregation: population-weighted means from GEE exports.

Input files (downloaded from Drive folder gaul82_vars_pop_weighted):
    pop_weighted_population.csv
    pop_weighted_gpp.csv
    pop_weighted_lst.csv
    pop_weighted_ntl.csv
    pop_weighted_ndvi.csv

MPI labels joined from:
    Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv

Feature aggregation: new_pixel_i = (value_i * pop_i) / total_region_pop,
then mean/median/stdDev/sum/min/max of new_pixel per ADM1.
Population uses standard spatial stats (no self-weighting).
25 features total.

Output columns: model, mask, cv_strategy, fold_index, R2, RMSE, MAE, n_test_obs
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
PW_DIR = BASE_DIR / "all_vars_pop_weighted"
MPI_SOURCE = BASE_DIR / "Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv"
OUT_FILE = BASE_DIR / "pop_weighted_cv_metrics.csv"

# Pop-weighted export files
PW_FILES = {
    "population": PW_DIR / "pop_weighted_population.csv",
    "gpp": PW_DIR / "pop_weighted_gpp.csv",
    "lst": PW_DIR / "pop_weighted_lst.csv",
    "ntl": PW_DIR / "pop_weighted_ntl.csv",
    "ndvi": PW_DIR / "pop_weighted_ndvi.csv",
}

# Column rename map: GEE export name → FEATURES name
COL_RENAME = {
    # population (standard stats on masked pixels)
    "Mean Population": "Mean_Pop",
    "Median Population": "Median_Pop",
    "Std Dev Population": "StdDev_Pop",
    "Total Population": "Total_Pop",
    "Min Population": "Min_Pop",
    "Max Population": "Max_Pop",
    # gpp (pop-weighted transform; Sum_GPP == pop-weighted mean)
    "Mean GPP": "Mean_GPP",
    "Median GPP": "Median_GPP",
    "Std Dev GPP": "StdDev_GPP",
    "Sum GPP": "Sum_GPP",
    "Min GPP": "Min_GPP",
    "Max GPP": "Max_GPP",
    # lst
    "Mean LST (K)": "Mean_LST",
    "Median LST (K)": "Median_LST",
    "Std Dev LST": "StdDev_LST",
    "Sum LST": "Sum_LST",
    "Min LST (K)": "Min_LST",
    "Max LST (K)": "Max_LST",
    # ntl
    "Mean NTL": "Mean_NTL",
    "Median NTL": "Median_NTL",
    "Std Dev NTL": "StdDev_NTL",
    "Sum NTL": "Sum_NTL",
    "Min NTL": "Min_NTL",
    "Max NTL": "Max_NTL",
    # ndvi
    "Mean NDVI": "Mean_NDVI",
    "Median NDVI": "Median_NDVI",
    "Std Dev NDVI": "StdDev_NDVI",
    "Sum NDVI": "Sum_NDVI",
    "Min NDVI": "Min_NDVI",
    "Max NDVI": "Max_NDVI",
}

FEATURES = [
    "Mean_GPP", "StdDev_GPP", "Median_Pop", "StdDev_Pop",
    "Mean_LST", "StdDev_LST", "Mean_NTL", "StdDev_NTL", "Sum_NTL",
    "Median_NDVI", "StdDev_NDVI", "ndvi_lst_ratio",
]

N_SPLITS = 5
RANDOM_STATE = 42

# Ensemble defaults (matching ensemble_training.py / build_cv_fold_metrics.py)
ENS_LR = 0.0005
ENS_WEIGHT_DECAY = 1e-6
ENS_BATCH_SIZE = 128
ENS_PATIENCE = 20
ENS_EPOCHS = 300
ALPHA_ENS = 0.4

DEFAULT_LAYERS = [
    {"type": "Dense", "units": 256, "activation": "relu"},
    {"type": "BatchNormalization"},
    {"type": "Dropout", "rate": 0.15},
    {"type": "Dense", "units": 128, "activation": "relu"},
    {"type": "BatchNormalization"},
    {"type": "Dropout", "rate": 0.10},
    {"type": "Dense", "units": 64, "activation": "relu"},
    {"type": "BatchNormalization"},
    {"type": "Dense", "units": 32, "activation": "relu"},
    {"type": "Dense", "units": 1, "activation": "relu"},
]


# ─────────────────────── model helpers ────────────────────────────────────────


def _build_dnn(input_dim: int) -> Sequential:
    lr_schedule = CosineDecay(
        initial_learning_rate=ENS_LR, decay_steps=10000, alpha=0.0001
    )
    model = Sequential()
    for i, layer in enumerate(DEFAULT_LAYERS):
        if layer["type"] == "Dense":
            model.add(
                Dense(
                    layer["units"],
                    activation=layer["activation"],
                    input_shape=(input_dim,) if i == 0 else (),
                )
            )
        elif layer["type"] == "BatchNormalization":
            model.add(BatchNormalization())
        elif layer["type"] == "Dropout":
            model.add(Dropout(layer["rate"]))
    model.compile(
        optimizer=AdamW(learning_rate=lr_schedule, weight_decay=ENS_WEIGHT_DECAY),
        loss=Huber(delta=1.0),
        metrics=["mae"],
    )
    return model


def _fit_ensemble(X_tr, y_tr, X_val, y_val):
    loss_fn = Huber(delta=1.0)

    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    xgb_model.fit(X_tr, y_tr)
    p_xgb_val = xgb_model.predict(X_val)

    dnn = _build_dnn(X_tr.shape[1])
    best_val_loss = float("inf")
    best_weights = dnn.get_weights()
    patience_count = 0

    for _ in range(ENS_EPOCHS):
        dnn.fit(
            X_tr,
            y_tr,
            epochs=1,
            batch_size=ENS_BATCH_SIZE,
            validation_data=(X_val, y_val),
            verbose=0,
        )
        p_dnn_val = dnn.predict(X_val, verbose=0).flatten()
        p_ens_val = ALPHA_ENS * p_dnn_val + (1 - ALPHA_ENS) * p_xgb_val
        ens_val_loss = float(loss_fn(y_val, p_ens_val).numpy())
        if ens_val_loss < best_val_loss:
            best_val_loss = ens_val_loss
            best_weights = dnn.get_weights()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= ENS_PATIENCE:
                break

    dnn.set_weights(best_weights)
    return dnn, xgb_model


def _metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return r2, rmse, mae


# ─────────────────────── data loading ─────────────────────────────────────────


def _load_pw_dataset() -> pd.DataFrame:
    missing = [name for name, p in PW_FILES.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing pop-weighted export files: {missing}\n"
            f"Download from Google Drive folder 'gaul82_vars_pop_weighted' to:\n  {PW_DIR}"
        )

    key_cols = ["Country", "Region", "Year"]
    merged = None

    for name, path in PW_FILES.items():
        df = pd.read_csv(path, encoding="utf-8")
        df = df.drop(columns=[c for c in ["system:index", ".geo"] if c in df.columns])
        df = df.rename(columns=COL_RENAME)
        df["Country"] = df["Country"].str.strip()
        df["Region"] = df["Region"].str.strip()
        if merged is None:
            merged = df
        else:
            keep = key_cols + [c for c in df.columns if c not in key_cols]
            merged = merged.merge(df[keep], on=key_cols, how="inner")

    # join MPI labels from original training file
    mpi_df = pd.read_csv(MPI_SOURCE, encoding="utf-8")
    target_col = next((c for c in ["MPI", "observed_MPI"] if c in mpi_df.columns), None)
    mpi_df = mpi_df.rename(columns={target_col: "MPI"})
    mpi_df["Country"] = mpi_df["Country"].str.strip()
    mpi_df["Region"] = mpi_df["Region"].str.strip()

    merged = merged.merge(
        mpi_df[["Country", "Region", "Year", "MPI"]],
        on=key_cols,
        how="inner",
    )

    den = merged["Mean_LST"].replace(0, float("nan"))
    merged["ndvi_lst_ratio"] = merged["Median_NDVI"] / den

    missing_feats = [f for f in FEATURES if f not in merged.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns after merge: {missing_feats}")

    merged = merged.dropna(subset=FEATURES + ["MPI"]).reset_index(drop=True)
    print(
        f"  Pop-weighted dataset: {len(merged)} rows | "
        f"countries: {merged['Country'].nunique()} | "
        f"years: {sorted(merged['Year'].unique())}"
    )
    return merged


# ─────────────────────── CV split helpers ─────────────────────────────────────


def _cv_splits(strategy: str, df: pd.DataFrame):
    if strategy == "random":
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        return list(kf.split(df))

    if strategy == "governorate":
        groups = df["Region"].values
        return list(GroupKFold(n_splits=N_SPLITS).split(df, groups=groups))

    if strategy == "country":
        groups = df["Country"].values
        return list(GroupKFold(n_splits=N_SPLITS).split(df, groups=groups))

    if strategy == "timeseries":
        return list(TimeSeriesSplit(n_splits=N_SPLITS).split(df))

    raise ValueError(strategy)


# ─────────────────────── main ─────────────────────────────────────────────────


def main():
    print(f"Loading pop-weighted exports ...")
    df = _load_pw_dataset()

    df_ts = df.sort_values("Year").reset_index(drop=True)

    strategies = ["random", "governorate", "country", "timeseries"]
    splits = {s: _cv_splits(s, df_ts if s == "timeseries" else df) for s in strategies}

    jobs = [(s, fi) for s in strategies for fi in range(N_SPLITS)]  # 20 jobs

    records: list[dict] = []

    with tqdm(total=len(jobs), desc="PW CV folds", unit="fold") as pbar:
        for strategy, fold_idx in jobs:
            pbar.set_postfix(strategy=strategy, fold=fold_idx + 1)

            source_df = df_ts if strategy == "timeseries" else df
            X = source_df[FEATURES].values
            y = source_df["MPI"].values

            tr_idx, te_idx = splits[strategy][fold_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X[tr_idx])
            X_te_s = scaler.transform(X[te_idx])
            y_tr, y_te = y[tr_idx], y[te_idx]

            X_dnn_tr, X_dnn_val, y_dnn_tr, y_dnn_val = train_test_split(
                X_tr_s, y_tr, test_size=0.2, random_state=RANDOM_STATE
            )

            dnn, xgb_model = _fit_ensemble(X_dnn_tr, y_dnn_tr, X_dnn_val, y_dnn_val)

            p_dnn = dnn.predict(X_te_s, verbose=0).flatten()
            p_xgb = xgb_model.predict(X_te_s)
            pred = np.clip(ALPHA_ENS * p_dnn + (1 - ALPHA_ENS) * p_xgb, 0, 1)
            tf.keras.backend.clear_session()

            r2, rmse, mae = _metrics(y_te, pred)
            records.append(
                {
                    "model": "DNN+XGB",
                    "mask": "pop_weighted",
                    "cv_strategy": strategy,
                    "fold_index": fold_idx + 1,
                    "R2": round(r2, 4),
                    "RMSE": round(rmse, 4),
                    "MAE": round(mae, 4),
                    "n_test_obs": len(te_idx),
                }
            )

            pbar.update(1)

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_FILE.name}  ({len(df_out)} rows).")
    print(
        df_out.groupby("cv_strategy")[["R2", "RMSE", "MAE"]].mean().round(4).to_string()
    )


if __name__ == "__main__":
    main()
