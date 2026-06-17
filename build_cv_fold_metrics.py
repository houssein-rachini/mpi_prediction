"""
Build cv_fold_level_metrics.csv

7 models × 4 CV strategies × 2 masks × 5 folds = 280 rows

Models       : KNN, RF, XGB, SVR, DNN, DNN+KNN, DNN+XGB
CV strategies: random, governorate, country, timeseries
Masks        : yes (500m masked), no (unmasked 500m)
Folds        : 5

Output columns:
    model, mask, cv_strategy, fold_index, R2, RMSE, MAE, n_test_obs
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
OUT_FILE = BASE_DIR / "cv_fold_level_metrics.csv"

MASKED_FILE   = BASE_DIR / "Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv"
UNMASKED_FILE = BASE_DIR / "unmasked_Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv"

FEATURES = [
    "Mean_GPP", "StdDev_GPP", "Median_Pop", "StdDev_Pop",
    "Mean_LST", "StdDev_LST", "Mean_NTL", "StdDev_NTL", "Sum_NTL",
    "Median_NDVI", "StdDev_NDVI", "ndvi_lst_ratio",
]
TARGET = "MPI"

N_SPLITS     = 5
RANDOM_STATE = 42

# Standalone DNN defaults (from dnn_training.py)
DNN_LR           = 0.001
DNN_WEIGHT_DECAY = 1e-5
DNN_BATCH_SIZE   = 128
DNN_PATIENCE     = 10
DNN_EPOCHS       = 200

# Ensemble DNN defaults (from ensemble_training.py)
ENS_LR           = 0.0005
ENS_WEIGHT_DECAY = 1e-6
ENS_BATCH_SIZE   = 128
ENS_PATIENCE     = 20
ENS_EPOCHS       = 300
ALPHA_ENS        = 0.4   # DNN weight (slider default in ensemble_training.py)

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

def _build_dnn(input_dim: int, lr: float, weight_decay: float, cosine_alpha: float) -> Sequential:
    lr_schedule = CosineDecay(initial_learning_rate=lr, decay_steps=10000, alpha=cosine_alpha)
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
        optimizer=AdamW(learning_rate=lr_schedule, weight_decay=weight_decay),
        loss=Huber(delta=1.0),
        metrics=["mae"],
    )
    return model


def _fit_dnn_standalone(X_tr, y_tr, X_val, y_val):
    """Standalone DNN: uses dnn_training.py defaults, EarlyStopping on DNN val_loss."""
    model = _build_dnn(X_tr.shape[1], DNN_LR, DNN_WEIGHT_DECAY, cosine_alpha=0.0005)
    cb = EarlyStopping(monitor="val_loss", patience=DNN_PATIENCE, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=DNN_EPOCHS, batch_size=DNN_BATCH_SIZE, callbacks=[cb], verbose=0)
    return model


def _fit_ensemble_dnn(X_tr, y_tr, X_val, y_val, base_model):
    """
    Ensemble DNN: uses ensemble_training.py defaults.
    Early stopping monitors ensemble val loss (not DNN val loss alone),
    matching the custom training loop in ensemble_training.py.
    """
    dnn = _build_dnn(X_tr.shape[1], ENS_LR, ENS_WEIGHT_DECAY, cosine_alpha=0.0001)
    loss_fn = Huber(delta=1.0)

    best_val_loss  = float("inf")
    best_weights   = dnn.get_weights()
    patience_count = 0

    for _ in range(ENS_EPOCHS):
        dnn.fit(X_tr, y_tr, epochs=1, batch_size=ENS_BATCH_SIZE,
                validation_data=(X_val, y_val), verbose=0)

        p_dnn_val  = dnn.predict(X_val, verbose=0).flatten()
        p_base_val = base_model.predict(X_val)
        p_ens_val  = ALPHA_ENS * p_dnn_val + (1 - ALPHA_ENS) * p_base_val
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
    return dnn


def _metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return r2, rmse, mae


def _make_ml_model(name: str, ensemble: bool = False):
    """
    ensemble=True uses ensemble_training.py defaults for XGB/KNN base models.
    ensemble=False uses ml_training.py defaults for standalone KNN/XGB/RF/SVR.
    """
    if name == "KNN":
        n = 4 if ensemble else 5
        return KNeighborsRegressor(n_neighbors=n, metric="manhattan")
    if name == "RF":
        return RandomForestRegressor(n_estimators=150, min_samples_split=2,
                                     min_samples_leaf=1, random_state=RANDOM_STATE)
    if name == "XGB":
        depth = 6 if ensemble else 5
        mcw   = 2 if ensemble else 1
        return xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                 max_depth=depth, min_child_weight=mcw,
                                 random_state=RANDOM_STATE, verbosity=0)
    if name == "SVR":
        return SVR(C=100, gamma=0.1, kernel="rbf")
    raise ValueError(name)


def _cv_splits(strategy: str, df: pd.DataFrame):
    """Return list of (train_idx, test_idx) tuples for the chosen strategy."""
    if strategy == "random":
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        return list(kf.split(df))

    if strategy == "governorate":
        groups = df["Region"].values if "Region" in df.columns else df["Country"].values
        n_groups = len(set(groups))
        if n_groups < N_SPLITS:
            raise ValueError(
                f"governorate strategy needs >= {N_SPLITS} groups, found {n_groups}"
            )
        return list(GroupKFold(n_splits=N_SPLITS).split(df, groups=groups))

    if strategy == "country":
        groups = df["Country"].values if "Country" in df.columns else df["Region"].values
        n_groups = len(set(groups))
        if n_groups < N_SPLITS:
            raise ValueError(
                f"country strategy needs >= {N_SPLITS} groups, found {n_groups}"
            )
        return list(GroupKFold(n_splits=N_SPLITS).split(df, groups=groups))

    if strategy == "timeseries":
        ts = TimeSeriesSplit(n_splits=N_SPLITS)
        return list(ts.split(df))

    raise ValueError(strategy)


def _load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    # Derive ndvi_lst_ratio if missing
    if "ndvi_lst_ratio" not in df.columns:
        lst_col  = next((c for c in df.columns if "LST" in c and "Mean" in c), None)
        ndvi_col = next((c for c in df.columns if "NDVI" in c and "Median" in c), None)
        if lst_col and ndvi_col:
            den = df[lst_col]
            num = df[ndvi_col]
            df["ndvi_lst_ratio"] = (num / den).where(den != 0)

    # Rename columns to match FEATURES
    rename = {
        "Mean_LST": "Mean_LST", "StdDev_LST": "StdDev_LST",
    }
    for col_variant, target in [("ndvi_lst_ratio", "ndvi_lst_ratio")]:
        if col_variant in df.columns and target not in df.columns:
            df[target] = df[col_variant]

    target_col = next((c for c in ["MPI", "observed_MPI"] if c in df.columns), None)
    if target_col is None:
        raise ValueError(f"No MPI column in {path.name}")
    df = df.rename(columns={target_col: "MPI"})

    required = FEATURES + ["MPI"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    df = df.dropna(subset=required).reset_index(drop=True)
    return df


# ─────────────────────── per-fold runner ──────────────────────────────────────

def run_fold(X_tr_s, X_te_s, y_tr, y_te) -> dict[str, tuple]:
    """
    Run all 7 models for one fold.
    - Standalone DNN uses dnn_training.py defaults + EarlyStopping on DNN val_loss.
    - Ensemble DNNs use ensemble_training.py defaults + EarlyStopping on ensemble val_loss.
    - Internal 80/20 val split of the training fold avoids test-fold leakage.
    Returns {model_name: (r2, rmse, mae)}.
    """
    results: dict[str, tuple] = {}

    # ── pure ML models (standalone params from ml_training.py) ───────────────
    for name in ("KNN", "RF", "XGB", "SVR"):
        m = _make_ml_model(name, ensemble=False)
        m.fit(X_tr_s, y_tr)
        results[name] = _metrics(y_te, np.clip(m.predict(X_te_s), 0, 1))

    # internal val split shared by all DNN variants
    X_dnn_tr, X_dnn_val, y_dnn_tr, y_dnn_val = train_test_split(
        X_tr_s, y_tr, test_size=0.2, random_state=RANDOM_STATE
    )

    # ── standalone DNN (dnn_training.py defaults) ─────────────────────────────
    dnn_standalone = _fit_dnn_standalone(X_dnn_tr, y_dnn_tr, X_dnn_val, y_dnn_val)
    p_dnn_standalone = dnn_standalone.predict(X_te_s, verbose=0).flatten()
    tf.keras.backend.clear_session()
    results["DNN"] = _metrics(y_te, np.clip(p_dnn_standalone, 0, 1))

    # ── ensemble models (ensemble_training.py defaults) ───────────────────────
    for ens_name, base_name in (("DNN+KNN", "KNN"), ("DNN+XGB", "XGB")):
        base = _make_ml_model(base_name, ensemble=True)
        base.fit(X_dnn_tr, y_dnn_tr)   # fit base on same 80% split as DNN
        dnn_ens = _fit_ensemble_dnn(X_dnn_tr, y_dnn_tr, X_dnn_val, y_dnn_val, base)
        p_dnn  = dnn_ens.predict(X_te_s, verbose=0).flatten()
        p_base = base.predict(X_te_s)
        pred   = np.clip(ALPHA_ENS * p_dnn + (1 - ALPHA_ENS) * p_base, 0, 1)
        tf.keras.backend.clear_session()
        results[ens_name] = _metrics(y_te, pred)

    return results


# ─────────────────────── main ─────────────────────────────────────────────────

def main():
    datasets   = {"yes": MASKED_FILE, "no": UNMASKED_FILE}
    strategies = ["random", "governorate", "country", "timeseries"]
    model_names = ["KNN", "RF", "XGB", "SVR", "DNN", "DNN+KNN", "DNN+XGB"]

    # Pre-load datasets
    dfs = {label: _load_dataset(path) for label, path in datasets.items()}

    # For timeseries, splits must be computed on year-sorted data
    dfs_ts = {label: df.sort_values("Year").reset_index(drop=True)
              for label, df in dfs.items()}

    splits = {
        (label, strategy): _cv_splits(
            strategy,
            dfs_ts[label] if strategy == "timeseries" else dfs[label]
        )
        for label in datasets
        for strategy in strategies
    }

    jobs = [
        (mask_label, strategy, fold_idx)
        for mask_label in datasets
        for strategy in strategies
        for fold_idx in range(N_SPLITS)
    ]  # 2 × 4 × 5 = 40 jobs

    records: list[dict] = []

    with tqdm(total=len(jobs), desc="CV folds", unit="fold") as pbar:
        for mask_label, strategy, fold_idx in jobs:
            pbar.set_postfix(mask=mask_label, strategy=strategy, fold=fold_idx + 1)

            df = dfs_ts[mask_label] if strategy == "timeseries" else dfs[mask_label]
            X  = df[FEATURES].values
            y  = df["MPI"].values

            tr_idx, te_idx = splits[(mask_label, strategy)][fold_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X[tr_idx])
            X_te_s = scaler.transform(X[te_idx])
            y_tr, y_te = y[tr_idx], y[te_idx]

            fold_results = run_fold(X_tr_s, X_te_s, y_tr, y_te)

            for model_name in model_names:
                r2, rmse, mae = fold_results[model_name]
                records.append({
                    "model":       model_name,
                    "mask":        mask_label,
                    "cv_strategy": strategy,
                    "fold_index":  fold_idx + 1,
                    "R2":          round(r2,   4),
                    "RMSE":        round(rmse, 4),
                    "MAE":         round(mae,  4),
                    "n_test_obs":  len(te_idx),
                })

            pbar.update(1)

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_FILE.name}  ({len(df_out)} rows).")


if __name__ == "__main__":
    main()
