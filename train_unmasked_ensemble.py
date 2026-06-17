"""
train_unmasked_ensemble.py

Trains the DNN+XGBoost ensemble on the UNMASKED training data using the
exact same architecture and hyperparameters as the Streamlit app defaults.

Input : unmasked_Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv
Output (separate filenames, does NOT overwrite masked models):
  unmasked_ensemble_scaler.pkl
  unmasked_ensemble_xgb_model.json
  unmasked_ensemble_xgb_dnn_model.h5

Run:
    python train_unmasked_ensemble.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import os
from tqdm import tqdm
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import xgboost as xgb

BASE_DIR      = Path(__file__).resolve().parent
UNMASKED_FILE = BASE_DIR / "unmasked_Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv"
OUT_SCALER    = BASE_DIR / "unmasked_ensemble_scaler.pkl"
OUT_XGB       = BASE_DIR / "unmasked_ensemble_xgb_model.json"
OUT_DNN       = BASE_DIR / "unmasked_ensemble_xgb_dnn_model.h5"

FEATURES = [
    "Mean_GPP", "StdDev_GPP", "Median_Pop", "StdDev_Pop",
    "Mean_LST", "StdDev_LST", "Mean_NTL", "StdDev_NTL", "Sum_NTL",
    "Median_NDVI", "StdDev_NDVI", "ndvi_lst_ratio",
]

# ── Hyperparameters (mirror ensemble_training.py defaults) ────────────────────
ALPHA        = 0.4
EPOCHS       = 300
LR           = 0.0005
WEIGHT_DECAY = 1e-6
BATCH_SIZE   = 128
PATIENCE     = 20
RANDOM_STATE = 42

XGB_PARAMS = dict(
    n_estimators=200, learning_rate=0.05, max_depth=6,
    min_child_weight=2, random_state=RANDOM_STATE, verbosity=0,
)

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


def main() -> None:
    print(f"Loading {UNMASKED_FILE.name} ...")
    df = pd.read_csv(UNMASKED_FILE, encoding="utf-8")
    print(f"  {len(df)} rows | {df['Country'].nunique()} countries")

    if "ndvi_lst_ratio" not in df.columns:
        den = df["Mean_LST"].replace(0, float("nan"))
        df["ndvi_lst_ratio"] = df["Median_NDVI"] / den
        print("  ndvi_lst_ratio computed from Median_NDVI / Mean_LST")

    df = df.dropna(subset=FEATURES + ["MPI"]).reset_index(drop=True)
    print(f"  {len(df)} rows after dropping nulls")

    X = df[FEATURES]
    y = np.maximum(df["MPI"].values, 0)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}")

    # Scale
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)

    # XGBoost
    print("\nTraining XGBoost ...")
    xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_model.fit(X_tr_s, y_train)
    p_xgb_val = xgb_model.predict(X_va_s)

    # DNN with early stopping on ensemble Huber val loss
    print("Training DNN (early stop on ensemble Huber val loss) ...")
    dnn     = _build_dnn(X_tr_s.shape[1])
    loss_fn = Huber(delta=1.0)

    class _EnsembleEarlyStopping(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_loss    = float("inf")
            self.best_weights = None
            self.wait         = 0

        def on_epoch_end(self, epoch, logs=None):
            p_dnn = self.model.predict(X_va_s, verbose=0).flatten()
            ens   = ALPHA * p_dnn + (1 - ALPHA) * p_xgb_val
            val_loss = float(loss_fn(y_val, ens).numpy())
            if val_loss < self.best_loss:
                self.best_loss    = val_loss
                self.best_weights = self.model.get_weights()
                self.wait         = 0
            else:
                self.wait += 1
                if self.wait >= PATIENCE:
                    print(f"  Early stop at epoch {epoch + 1}  (best val loss={self.best_loss:.5f})")
                    self.model.stop_training = True

        def on_train_end(self, logs=None):
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)

    ens_cb = _EnsembleEarlyStopping()

    class _TqdmProgress(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.bar = tqdm(total=EPOCHS, desc="DNN training", unit="epoch", dynamic_ncols=True)
        def on_epoch_end(self, epoch, logs=None):
            self.bar.set_postfix(ens_val=f"{ens_cb.best_loss:.5f}", dnn_val=f"{logs.get('val_loss', 0):.5f}")
            self.bar.update(1)
        def on_train_end(self, logs=None):
            self.bar.close()

    dnn.fit(X_tr_s, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_data=(X_va_s, y_val),
            callbacks=[ens_cb, _TqdmProgress()],
            verbose=0)

    # Final metrics (best weights already restored by callback)
    p_dnn_val = dnn.predict(X_va_s, verbose=0).flatten()
    p_xgb_val = xgb_model.predict(X_va_s)
    pred       = np.clip(ALPHA * p_dnn_val + (1 - ALPHA) * p_xgb_val, 0, 1)

    r2   = r2_score(y_val, pred)
    rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
    mae  = float(mean_absolute_error(y_val, pred))
    print(f"\n  Val R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    # Save
    joblib.dump(scaler, OUT_SCALER)
    xgb_model.save_model(OUT_XGB)
    dnn.save(OUT_DNN)
    print(f"\nSaved:")
    print(f"  {OUT_SCALER.name}")
    print(f"  {OUT_XGB.name}")
    print(f"  {OUT_DNN.name}")


if __name__ == "__main__":
    main()
