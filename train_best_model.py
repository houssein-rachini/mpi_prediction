"""
train_best_model.py

Full pipeline: data analysis → balanced GroupKFold → Optuna hyperparameter
search across XGBoost, LightGBM, Random Forest, and MLP → weighted ensemble
→ save best model.

Features (24):
  12 base satellite features + 12 climate anomaly features
  Anomaly NaNs imputed with 0 (= no deviation from baseline)

Validation:
  5-fold GroupKFold by country, folds balanced by mean MPI (round-robin)

Run:
    python train_best_model.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR   = Path(__file__).resolve().parent
DATA_FILE  = BASE_DIR / "Final_Merged_with_anomalies.csv"
OUT_DIR    = BASE_DIR / "best_model"
OUT_DIR.mkdir(exist_ok=True)

N_FOLDS    = 5
N_TRIALS   = {
    "xgb":  80,
    "lgbm": 80,
    "rf":   40,
    "mlp":  40,
}
RANDOM_STATE = 42

BASE_FEATURES = [
    "Mean_GPP", "StdDev_GPP",
    "Median_Pop", "StdDev_Pop",
    "Mean_LST", "StdDev_LST",
    "Mean_NTL", "StdDev_NTL", "Sum_NTL",
    "Median_NDVI", "StdDev_NDVI",
    "ndvi_lst_ratio",
]
ANOM_FEATURES = [
    "NDVI_anom", "LSTN_anom", "NTL_anom", "GPP_anom",
    "NDVI_anom_lag1", "LSTN_anom_lag1", "NTL_anom_lag1", "GPP_anom_lag1",
]
ALL_FEATURES = BASE_FEATURES + ANOM_FEATURES
TARGET = "MPI"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data loading & analysis
# ══════════════════════════════════════════════════════════════════════════════

def load_and_analyze() -> pd.DataFrame:
    print("=" * 65)
    print("  DATA ANALYSIS")
    print("=" * 65)

    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=[TARGET])

    print(f"\nRows: {len(df)}  |  Countries: {df['Country'].nunique()}  |  "
          f"Years: {df['Year'].min()}–{df['Year'].max()}")
    print(f"Target (MPI): min={df[TARGET].min():.3f}  max={df[TARGET].max():.3f}  "
          f"mean={df[TARGET].mean():.3f}  std={df[TARGET].std():.3f}")

    # Missing values per feature
    print("\nMissing values per feature:")
    miss = df[ALL_FEATURES].isna().sum()
    for f, n in miss[miss > 0].items():
        print(f"  {f:35s}: {n:4d} / {len(df)}  ({n/len(df)*100:.1f}%)")

    # Correlation with MPI
    print("\nCorrelation with MPI (Pearson r):")
    corrs = {}
    for f in ALL_FEATURES:
        sub = df[[f, TARGET]].dropna()
        if len(sub) > 10:
            r, p = stats.pearsonr(sub[f], sub[TARGET])
            corrs[f] = (r, p)
            sig = "*" if p < 0.05 else " "
            print(f"  {sig} {f:35s}: r={r:+.3f}  p={p:.3f}")

    # Country distribution
    country_stats = (
        df.groupby("Country")[TARGET]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mean_MPI", "count": "n_obs"})
        .sort_values("mean_MPI", ascending=False)
    )
    print(f"\nTop 5 highest MPI countries:\n{country_stats.head(5).to_string()}")
    print(f"\nTop 5 lowest MPI countries:\n{country_stats.tail(5).to_string()}")

    # Save correlation plot
    corr_vals = {f: v[0] for f, v in corrs.items()}
    corr_df = pd.Series(corr_vals).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in corr_df]
    ax.barh(corr_df.index, corr_df.values, color=colors, alpha=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Pearson r with MPI")
    ax.set_title("Feature correlations with MPI")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()

    # MPI distribution by year
    fig, ax = plt.subplots(figsize=(10, 4))
    for year, g in df.groupby("Year"):
        ax.scatter([year] * len(g), g[TARGET], alpha=0.15, s=10, color="#1f77b4")
    ax.boxplot(
        [df[df["Year"] == y][TARGET].values for y in sorted(df["Year"].unique())],
        positions=sorted(df["Year"].unique()),
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="#aec7e8", alpha=0.6),
        medianprops=dict(color="navy", lw=2),
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("MPI")
    ax.set_title("MPI distribution by year")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "mpi_by_year.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nAnalysis plots saved to {OUT_DIR}/")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Feature preparation
# ══════════════════════════════════════════════════════════════════════════════

def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = df.copy()

    # Impute anomaly NaNs with 0 (= no deviation from baseline)
    for col in ANOM_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    present = [f for f in ALL_FEATURES if f in df.columns]
    df = df.dropna(subset=present + [TARGET])

    X = df[present].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    groups = df["Country"].values

    print(f"\nFinal dataset: {len(df)} rows | {len(present)} features | "
          f"{len(np.unique(groups))} countries")
    return X, y, groups, present


# ══════════════════════════════════════════════════════════════════════════════
# 3. Balanced GroupKFold
# ══════════════════════════════════════════════════════════════════════════════

def make_balanced_folds(
    y: np.ndarray, groups: np.ndarray, n_folds: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Assign countries to folds round-robin sorted by mean MPI so that each
    fold covers the full poverty spectrum.
    """
    country_mean = (
        pd.DataFrame({"y": y, "g": groups})
        .groupby("g")["y"].mean()
        .sort_values()
    )
    country_fold = {
        c: i % n_folds for i, c in enumerate(country_mean.index)
    }
    fold_labels = np.array([country_fold[g] for g in groups])

    folds = []
    for fold in range(n_folds):
        test_idx  = np.where(fold_labels == fold)[0]
        train_idx = np.where(fold_labels != fold)[0]
        folds.append((train_idx, test_idx))

    # Report fold composition
    print("\nFold composition:")
    for i, (tr, te) in enumerate(folds):
        n_countries = len(np.unique(groups[te]))
        mpi_mean    = y[te].mean()
        print(f"  Fold {i+1}: train={len(tr):4d}  test={len(te):4d}  "
              f"test_countries={n_countries:2d}  test_mean_MPI={mpi_mean:.3f}")
    return folds


# ══════════════════════════════════════════════════════════════════════════════
# 4. CV evaluation helper
# ══════════════════════════════════════════════════════════════════════════════

def cv_score(
    model_fn,
    X: np.ndarray, y: np.ndarray,
    folds: list, scale: bool = False,
) -> dict:
    maes, rmses, r2s = [], [], []
    for train_idx, test_idx in folds:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

        model = model_fn()
        model.fit(X_tr, y_tr)
        preds = np.clip(model.predict(X_te), 0, 1)

        maes.append(mean_absolute_error(y_te, preds))
        rmses.append(np.sqrt(mean_squared_error(y_te, preds)))
        r2s.append(r2_score(y_te, preds))

    return {
        "mae":  float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "r2":   float(np.mean(r2s)),
        "mae_std":  float(np.std(maes)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Optuna studies
# ══════════════════════════════════════════════════════════════════════════════

def tune_xgboost(X, y, folds, n_trials):
    print(f"\n{'-'*50}")
    print(f"  Tuning XGBoost ({n_trials} trials) ...")

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1500),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "random_state": RANDOM_STATE, "n_jobs": -1, "tree_method": "hist",
        }
        fn = lambda: xgb.XGBRegressor(**params)
        res = cv_score(fn, X, y, folds)
        return res["mae"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best.update({"random_state": RANDOM_STATE, "n_jobs": -1, "tree_method": "hist"})
    print(f"  Best MAE: {study.best_value:.4f}")
    return best, study.best_value


def tune_lgbm(X, y, folds, n_trials):
    print(f"\n{'-'*50}")
    print(f"  Tuning LightGBM ({n_trials} trials) ...")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1500),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 300),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
        }
        fn = lambda: lgb.LGBMRegressor(**params)
        res = cv_score(fn, X, y, folds)
        return res["mae"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1})
    print(f"  Best MAE: {study.best_value:.4f}")
    return best, study.best_value


def tune_rf(X, y, folds, n_trials):
    print(f"\n{'-'*50}")
    print(f"  Tuning Random Forest ({n_trials} trials) ...")

    def objective(trial):
        params = {
            "n_estimators":  trial.suggest_int("n_estimators", 100, 800),
            "max_depth":     trial.suggest_int("max_depth", 4, 30),
            "max_features":  trial.suggest_float("max_features", 0.3, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "random_state": RANDOM_STATE, "n_jobs": -1,
        }
        fn = lambda: RandomForestRegressor(**params)
        res = cv_score(fn, X, y, folds)
        return res["mae"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best.update({"random_state": RANDOM_STATE, "n_jobs": -1})
    print(f"  Best MAE: {study.best_value:.4f}")
    return best, study.best_value


def tune_mlp(X, y, folds, n_trials):
    print(f"\n{'-'*50}")
    print(f"  Tuning MLP ({n_trials} trials) ...")

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 4)
        units    = trial.suggest_int("units", 32, 512)
        layers   = tuple([units] * n_layers)
        params = {
            "hidden_layer_sizes": layers,
            "activation":   trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha":        trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "max_iter": 300, "early_stopping": True, "random_state": RANDOM_STATE,
        }
        fn = lambda: MLPRegressor(**params)
        res = cv_score(fn, X, y, folds, scale=True)
        return res["mae"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    n_layers = best.pop("n_layers")
    units    = best.pop("units")
    best.pop("lr", None)
    best.update({
        "hidden_layer_sizes": tuple([units] * n_layers),
        "learning_rate_init": study.best_trial.params["lr"],
        "max_iter": 500, "early_stopping": True, "random_state": RANDOM_STATE,
    })
    print(f"  Best MAE: {study.best_value:.4f}")
    return best, study.best_value


# ══════════════════════════════════════════════════════════════════════════════
# 6. Ensemble: weighted average of best models
# ══════════════════════════════════════════════════════════════════════════════

def tune_ensemble_weights(preds_list: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    """Find optimal weights for ensemble via grid search."""
    best_mae, best_w = 1e9, None
    n = len(preds_list)
    steps = np.arange(0, 1.1, 0.1)

    from itertools import product
    for combo in product(steps, repeat=n):
        s = sum(combo)
        if s < 0.01:
            continue
        w = np.array(combo) / s
        blended = sum(w[i] * preds_list[i] for i in range(n))
        mae = mean_absolute_error(y, np.clip(blended, 0, 1))
        if mae < best_mae:
            best_mae = mae
            best_w = w
    return best_w


# ══════════════════════════════════════════════════════════════════════════════
# 7. Final evaluation & plots
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_and_plot(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    country_labels: np.ndarray,
):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {model_name:20s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return mae, rmse, r2


def plot_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    country_labels: np.ndarray,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: predicted vs actual
    ax = axes[0]
    countries = np.unique(country_labels)
    cmap  = plt.colormaps.get_cmap("tab20")
    color = {c: cmap((i % 20) / 20) for i, c in enumerate(countries)}
    for c in countries:
        m = country_labels == c
        ax.scatter(y_true[m], y_pred[m], s=10, alpha=0.5, color=color[c])
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    ax.set_xlabel("Actual MPI")
    ax.set_ylabel("Predicted MPI")
    ax.set_title(f"{model_name}\nMAE={mae:.4f}  R²={r2:.4f}")
    ax.grid(alpha=0.3)

    # Residuals
    ax = axes[1]
    residuals = y_pred - y_true
    ax.scatter(y_true, residuals, s=10, alpha=0.4, color="#1f77b4")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Actual MPI")
    ax.set_ylabel("Residual (pred − actual)")
    ax.set_title("Residuals")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("+", "")
    plt.savefig(OUT_DIR / f"predictions_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_names: list[str], model_name: str):
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "booster_") and hasattr(model.booster_, "feature_importance"):
            imp = model.booster_.feature_importance(importance_type="gain")
        else:
            return
        idx = np.argsort(imp)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.barh([feature_names[i] for i in idx], imp[idx], color="#1f77b4", alpha=0.8)
        ax.set_title(f"Feature Importance — {model_name}")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        safe_name = model_name.replace(" ", "_")
        plt.savefig(OUT_DIR / f"importance_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Feature importance plot saved.")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Data analysis ──────────────────────────────────────────────────────
    df = load_and_analyze()
    X, y, groups, feature_names = prepare_features(df)

    # ── 2. Balanced folds ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  BALANCED GROUPKFOLD (5 folds by country)")
    print("=" * 65)
    folds = make_balanced_folds(y, groups)

    # ── 3. Hyperparameter tuning ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  HYPERPARAMETER SEARCH")
    print("=" * 65)

    results = {}
    best_params_all = {}

    xgb_params, xgb_mae = tune_xgboost(X, y, folds, N_TRIALS["xgb"])
    results["XGBoost"] = xgb_mae
    best_params_all["xgb"] = xgb_params

    lgbm_params, lgbm_mae = tune_lgbm(X, y, folds, N_TRIALS["lgbm"])
    results["LightGBM"] = lgbm_mae
    best_params_all["lgbm"] = lgbm_params

    rf_params, rf_mae = tune_rf(X, y, folds, N_TRIALS["rf"])
    results["Random Forest"] = rf_mae
    best_params_all["rf"] = rf_params

    mlp_params, mlp_mae = tune_mlp(X, y, folds, N_TRIALS["mlp"])
    results["MLP"] = mlp_mae
    best_params_all["mlp"] = mlp_params

    # ── 4. Full CV evaluation with best params ────────────────────────────────
    print("\n" + "=" * 65)
    print("  FULL CV EVALUATION (best hyperparams)")
    print("=" * 65)

    models_to_eval = {
        "XGBoost":      (lambda: xgb.XGBRegressor(**xgb_params),   False),
        "LightGBM":     (lambda: lgb.LGBMRegressor(**lgbm_params),  False),
        "Random Forest":(lambda: RandomForestRegressor(**rf_params), False),
        "MLP":          (lambda: MLPRegressor(**mlp_params),         True),
    }

    cv_results = {}
    for name, (fn, scale) in models_to_eval.items():
        res = cv_score(fn, X, y, folds, scale=scale)
        cv_results[name] = res
        print(f"  {name:20s}  MAE={res['mae']:.4f}±{res['mae_std']:.4f}  "
              f"RMSE={res['rmse']:.4f}  R²={res['r2']:.4f}")

    # ── 5. Ensemble of top 2 models ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ENSEMBLE (top 2 models)")
    print("=" * 65)

    sorted_models = sorted(cv_results.items(), key=lambda x: x[1]["mae"])
    top2 = [sorted_models[0][0], sorted_models[1][0]]
    print(f"  Top 2 models: {top2}")

    # Collect OOF predictions for ensemble weighting
    oof_preds = {name: np.zeros(len(y)) for name in top2}
    for train_idx, test_idx in folds:
        for name, (fn, scale) in models_to_eval.items():
            if name not in top2:
                continue
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr        = y[train_idx]
            if scale:
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tr)
                X_te = sc.transform(X_te)
            m = fn()
            m.fit(X_tr, y_tr)
            oof_preds[name][test_idx] = np.clip(m.predict(X_te), 0, 1)

    preds_list = [oof_preds[n] for n in top2]
    weights = tune_ensemble_weights(preds_list, y)
    ens_pred = sum(weights[i] * preds_list[i] for i in range(2))
    ens_pred = np.clip(ens_pred, 0, 1)

    ens_mae  = mean_absolute_error(y, ens_pred)
    ens_rmse = np.sqrt(mean_squared_error(y, ens_pred))
    ens_r2   = r2_score(y, ens_pred)
    print(f"  Weights: {top2[0]}={weights[0]:.2f}  {top2[1]}={weights[1]:.2f}")
    print(f"  Ensemble OOF  MAE={ens_mae:.4f}  RMSE={ens_rmse:.4f}  R²={ens_r2:.4f}")
    cv_results["Ensemble"] = {"mae": ens_mae, "rmse": ens_rmse, "r2": ens_r2,
                               "mae_std": 0.0, "weights": weights.tolist(),
                               "components": top2}

    # ── 6. Determine best model ───────────────────────────────────────────────
    best_name = min(cv_results, key=lambda k: cv_results[k]["mae"])
    print(f"\n  ** Best model: {best_name}  (MAE={cv_results[best_name]['mae']:.4f})")

    # ── 7. Train final model on full data ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TRAINING FINAL MODEL ON FULL DATA")
    print("=" * 65)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    final_models = {}
    for name, (fn, scale) in models_to_eval.items():
        m = fn()
        m.fit(X_scaled if scale else X, y)
        final_models[name] = (m, scale)
        print(f"  Trained {name}")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  GENERATING PLOTS")
    print("=" * 65)

    for name, (m, scale) in final_models.items():
        preds = np.clip(m.predict(X_scaled if scale else X), 0, 1)
        plot_predictions(name, y, preds, groups)
        plot_feature_importance(m, feature_names, name)

    # Ensemble OOF plot
    plot_predictions("Ensemble (OOF)", y, ens_pred, groups)

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    names  = list(cv_results.keys())
    maes   = [cv_results[n]["mae"] for n in names]
    colors = ["gold" if n == best_name else "#1f77b4" for n in names]
    ax.bar(names, maes, color=colors, alpha=0.85)
    ax.set_ylabel("CV MAE (5-fold)")
    ax.set_title("Model Comparison — CV MAE (lower is better)\n★ = best")
    for i, (n, v) in enumerate(zip(names, maes)):
        ax.text(i, v + 0.0005, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved model_comparison.png")

    # ── 9. Save best model ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SAVING")
    print("=" * 65)

    joblib.dump(scaler, OUT_DIR / "scaler.pkl")

    if best_name == "Ensemble":
        for name in top2:
            m, scale = final_models[name]
            safe = name.lower().replace(" ", "_")
            joblib.dump(m, OUT_DIR / f"model_{safe}.pkl")
        meta = {
            "type":       "ensemble",
            "components": top2,
            "weights":    weights.tolist(),
            "features":   feature_names,
            "cv_results": cv_results,
        }
    else:
        m, scale = final_models[best_name]
        safe = best_name.lower().replace(" ", "_")
        joblib.dump(m, OUT_DIR / f"model_{safe}.pkl")
        meta = {
            "type":       best_name,
            "features":   feature_names,
            "scaled":     scale,
            "cv_results": cv_results,
        }

    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Model saved to {OUT_DIR}/")
    print(f"  Features used: {len(feature_names)}")

    # Final summary
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print(f"  {'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print(f"  {'-'*48}")
    for name, res in sorted(cv_results.items(), key=lambda x: x[1]['mae']):
        star = " **" if name == best_name else ""
        print(f"  {name:<22} {res['mae']:>8.4f} {res['rmse']:>8.4f} {res['r2']:>8.4f}{star}")


if __name__ == "__main__":
    main()
