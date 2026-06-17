"""Run DNN+XGB ensemble, mask=yes, cv_strategy=country, 5 folds."""
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from build_cv_fold_metrics import (
    MASKED_FILE, FEATURES, N_SPLITS, RANDOM_STATE,
    _load_dataset, _make_ml_model, _fit_ensemble_dnn,
    _metrics, ALPHA_ENS,
)

df = _load_dataset(MASKED_FILE)
print(f"Dataset: {len(df)} rows, {df['Country'].nunique()} countries")

groups   = df["Country"].values
n_groups = len(set(groups))
print(f"Groups (countries): {n_groups}")
if n_groups < N_SPLITS:
    raise ValueError(f"Need >= {N_SPLITS} countries, found {n_groups}")

splits = list(GroupKFold(n_splits=N_SPLITS).split(df, groups=groups))
X = df[FEATURES].values
y = df["MPI"].values

for fold_idx, (tr_idx, te_idx) in enumerate(tqdm(splits, desc="Folds", unit="fold")):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X[tr_idx])
    X_te_s = scaler.transform(X[te_idx])
    y_tr, y_te = y[tr_idx], y[te_idx]

    X_dnn_tr, X_dnn_val, y_dnn_tr, y_dnn_val = train_test_split(
        X_tr_s, y_tr, test_size=0.2, random_state=RANDOM_STATE
    )

    base = _make_ml_model("XGB", ensemble=True)
    base.fit(X_dnn_tr, y_dnn_tr)

    dnn_ens = _fit_ensemble_dnn(X_dnn_tr, y_dnn_tr, X_dnn_val, y_dnn_val, base)

    p_dnn  = dnn_ens.predict(X_te_s, verbose=0).flatten()
    p_base = base.predict(X_te_s)
    pred   = np.clip(ALPHA_ENS * p_dnn + (1 - ALPHA_ENS) * p_base, 0, 1)
    tf.keras.backend.clear_session()

    r2, rmse, mae = _metrics(y_te, pred)
    test_countries = set(groups[te_idx])
    print(f"  Fold {fold_idx+1} | n_test={len(te_idx)} | countries={test_countries}")
    print(f"           R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
