"""
Build buffer_sensitivity_folds.csv

DNN+XGB ensemble, 5-fold GroupKFold by country.
  - 6 masked buffer datasets (0m, 250m, 500m, 1000m, 2000m, 3000m) — mask=yes
  - 1 unmasked dataset (no building mask at all) — mask=no
Output: 35 rows (7 datasets × 5 folds)

Columns: mask, buffer_m, fold_index, R2, RMSE, MAE
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
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

MAX_WORKERS = 3  # tune to your CPU core count; each worker caps TF at 2 threads

BASE_DIR = Path(__file__).resolve().parent
OUT_FILE = BASE_DIR / "buffer_sensitivity_folds.csv"

BUFFER_FILES = {
    0:    BASE_DIR / "merged_all_vars_0m_original_ref_gaul.csv",
    250:  BASE_DIR / "merged_all_vars_250m_original_ref_gaul.csv",
    500:  BASE_DIR / "Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv",
    1000: BASE_DIR / "merged_all_vars_1000m_original_ref_gaul.csv",
    2000: BASE_DIR / "merged_all_vars_2000m_original_ref_gaul.csv",
    3000: BASE_DIR / "merged_all_vars_3000m_original_ref_gaul.csv",
}
UNMASKED_FILE = BASE_DIR / "unmasked_Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv"

FEATURES = [
    "Mean_GPP", "StdDev_GPP", "Median_Pop", "StdDev_Pop",
    "Mean_LST", "StdDev_LST", "Mean_NTL", "StdDev_NTL", "Sum_NTL",
    "Median_NDVI", "StdDev_NDVI", "ndvi_lst_ratio",
]

N_SPLITS     = 5
RANDOM_STATE = 42
LR           = 0.0005
WEIGHT_DECAY = 1e-6
BATCH_SIZE   = 128
PATIENCE     = 20
EPOCHS       = 300
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
    lr_schedule = CosineDecay(initial_learning_rate=LR, decay_steps=10000, alpha=0.0001)
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
    if target_col is None:
        raise ValueError(f"No MPI column in {path.name}")
    df = df.rename(columns={target_col: "MPI"})

    required = FEATURES + ["MPI"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    df = df.dropna(subset=required).reset_index(drop=True)
    country_col = next((c for c in ["Country", "country"] if c in df.columns), None)
    groups = df[country_col].values if country_col else None
    return df[FEATURES].values, df["MPI"].values, groups


def run_fold(X_tr_s, X_te_s, y_tr, y_te) -> tuple[float, float, float]:
    loss_fn = Huber(delta=1.0)

    X_dnn_tr, X_dnn_val, y_dnn_tr, y_dnn_val = train_test_split(
        X_tr_s, y_tr, test_size=0.2, random_state=RANDOM_STATE
    )

    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                  max_depth=6, min_child_weight=2,
                                  random_state=RANDOM_STATE, verbosity=0)
    xgb_model.fit(X_dnn_tr, y_dnn_tr)
    p_xgb_val = xgb_model.predict(X_dnn_val)  # constant — compute once

    dnn = _build_dnn(X_tr_s.shape[1])
    best_val_loss  = float("inf")
    best_weights   = dnn.get_weights()
    patience_count = 0

    for _ in range(EPOCHS):
        dnn.fit(X_dnn_tr, y_dnn_tr, epochs=1, batch_size=BATCH_SIZE,
                validation_data=(X_dnn_val, y_dnn_val), verbose=0)
        p_dnn_val    = dnn.predict(X_dnn_val, verbose=0).flatten()
        p_ens_val    = ALPHA_ENS * p_dnn_val + (1 - ALPHA_ENS) * p_xgb_val
        ens_val_loss = float(loss_fn(y_dnn_val, p_ens_val).numpy())
        if ens_val_loss < best_val_loss:
            best_val_loss  = ens_val_loss
            best_weights   = dnn.get_weights()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    dnn.set_weights(best_weights)
    p_dnn = dnn.predict(X_te_s, verbose=0).flatten()
    tf.keras.backend.clear_session()

    p_xgb = xgb_model.predict(X_te_s)
    pred  = np.clip(ALPHA_ENS * p_dnn + (1 - ALPHA_ENS) * p_xgb, 0, 1)
    r2   = r2_score(y_te, pred)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mae  = float(mean_absolute_error(y_te, pred))
    return r2, rmse, mae


def run_dataset(args) -> list[dict]:
    """Worker: runs all N_SPLITS folds for one dataset. Called in a subprocess."""
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    # cap threads per worker so parallel processes don't fight for CPU
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    mask_label, buffer_m, path = args
    X, y, groups = _load_dataset(Path(path))
    if groups is None:
        raise ValueError(f"No country column found in {path}")
    n_groups = len(set(groups))
    if n_groups < N_SPLITS:
        raise ValueError(f"Need >= {N_SPLITS} countries, found {n_groups}")

    # shuffle group order to match ensemble_training.py behaviour
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(unique_groups)
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    shuffled_group_indices = np.array([group_to_idx[g] for g in groups])

    gkf  = GroupKFold(n_splits=N_SPLITS)
    rows = []
    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X, groups=shuffled_group_indices)):
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X[tr_idx])
        X_te_s = scaler.transform(X[te_idx])
        r2, rmse, mae = run_fold(X_tr_s, X_te_s, y[tr_idx], y[te_idx])
        rows.append({
            "mask":       mask_label,
            "buffer_m":   buffer_m,
            "fold_index": fold_idx + 1,
            "R2":         round(r2,   4),
            "RMSE":       round(rmse, 4),
            "MAE":        round(mae,  4),
        })
    return rows


def main():
    jobs = (
        [("yes", bm, str(path)) for bm, path in sorted(BUFFER_FILES.items())]
        + [("no", None, str(UNMASKED_FILE))]
    )
    total_folds = len(jobs) * N_SPLITS
    all_records: list[dict] = []

    print(f"Running {len(jobs)} datasets × {N_SPLITS} folds = {total_folds} folds "
          f"with MAX_WORKERS={MAX_WORKERS} ...")

    with tqdm(total=len(jobs), unit="dataset", dynamic_ncols=True) as pbar:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(run_dataset, job): job for job in jobs}
            for future in as_completed(futures):
                mask_label, buffer_m, _ = futures[future]
                pbar.set_postfix(mask=mask_label, buffer=buffer_m)
                all_records.extend(future.result())
                pbar.update(1)

    df_out = pd.DataFrame(all_records).sort_values(
        ["mask", "buffer_m", "fold_index"]
    ).reset_index(drop=True)
    df_out.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_FILE.name}  ({len(df_out)} rows).")


if __name__ == "__main__":
    main()
