"""
shap_analysis.py

Trains XGBoost on all candidate features and computes SHAP values
to determine actual feature contributions and redundancy.

Outputs saved to shap_output/:
  - shap_summary.png        : beeswarm (impact + direction)
  - shap_bar.png            : mean |SHAP| ranking
  - shap_dependence_*.png   : top-10 feature dependence plots
  - shap_redundancy.png     : correlation of SHAP values between features
  - shap_results.csv        : mean |SHAP| per feature
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR  = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "gee_all_vars_added" / "all_82_merged_500m_with_anomalies_MPI.csv"
OUT_DIR   = BASE_DIR / "shap_output"
OUT_DIR.mkdir(exist_ok=True)

CANDIDATE_FEATURES = [
    # NTL
    "Mean_NTL", "StdDev_NTL", "Median_NTL", "Sum_NTL",
    # GPP
    "Mean_GPP", "StdDev_GPP", "Median_GPP",
    # Population
    "Mean_Pop", "StdDev_Pop", "Median_Pop",
    # LST day
    "Mean_LST_Day", "StdDev_LST_Day",
    # LST night
    "Mean_LST", "StdDev_LST",
    # NDVI
    "Median_NDVI", "StdDev_NDVI", "ndvi_lst_ratio",
    # Anomalies
    "NTL_anom", "NTL_anom_lag1",
    "NDVI_anom", "NDVI_anom_lag1",
    "LSTN_anom", "LSTN_anom_lag1",
    "LST_Day_anom", "LST_Day_anom_lag1",
    "GPP_anom", "GPP_anom_lag1",
    "PDSI_anom", "PDSI_anom_lag1",
    "precipitation_anom", "precipitation_anom_lag1",
    # Climate base
    "Mean_PDSI", "Mean_Precip",
]
TARGET = "MPI"


def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA_FILE, encoding="utf-8")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    features = [f for f in CANDIDATE_FEATURES if f in df.columns]

    # Winsorise NTL anomalies at ±5 (2022-2023 sensor spike)
    for col in ["NTL_anom", "NTL_anom_lag1"]:
        if col in df.columns:
            df[col] = df[col].clip(-5, 5)

    # Impute anomaly NaNs with 0 (= no deviation from baseline)
    anom_cols = [f for f in features if "anom" in f]
    for col in anom_cols:
        df[col] = df[col].fillna(0.0)

    # Drop rows missing any non-anomaly feature
    base_feats = [f for f in features if "anom" not in f and f not in ("Mean_PDSI","Mean_Precip")]
    df = df.dropna(subset=base_feats + [TARGET]).reset_index(drop=True)

    print(f"Rows after cleaning: {len(df)} | Countries: {df['Country'].nunique()}")
    print(f"Features: {len(features)}")
    return df, features


def train_xgb_oof(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[xgb.XGBRegressor, np.ndarray]:
    """5-fold GroupKFold OOF — returns final model trained on full data + OOF preds."""
    country_mean = (
        pd.DataFrame({"y": y, "g": groups})
        .groupby("g")["y"].mean()
        .sort_values()
    )
    country_fold = {c: i % 5 for i, c in enumerate(country_mean.index)}
    fold_labels   = np.array([country_fold[g] for g in groups])

    oof = np.zeros(len(y))
    params = dict(
        n_estimators=800, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    for fold in range(5):
        tr = np.where(fold_labels != fold)[0]
        te = np.where(fold_labels == fold)[0]
        m  = xgb.XGBRegressor(**params)
        m.fit(X[tr], y[tr])
        oof[te] = np.clip(m.predict(X[te]), 0, 1)

    mae = mean_absolute_error(y, oof)
    r2  = r2_score(y, oof)
    print(f"OOF  MAE={mae:.4f}  R²={r2:.4f}")

    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model, oof


def plot_shap(shap_values: np.ndarray, X_df: pd.DataFrame, feature_names: list[str]):
    # --- 1. Bar chart: mean |SHAP| ---
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    ranked   = [(feature_names[i], mean_abs[i]) for i in order]

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.32)))
    colors = ["#d62728" if mean_abs[i] > 0.01 else "#aec7e8" for i in order[::-1]]
    ax.barh([r[0] for r in ranked[::-1]], [r[1] for r in ranked[::-1]],
            color=colors, alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP) — all candidates")
    ax.axvline(0.005, color="gray", lw=0.8, ls="--", label="0.005 threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved shap_bar.png")

    # --- 2. Beeswarm summary ---
    expl = shap.Explanation(
        values=shap_values,
        data=X_df.values,
        feature_names=feature_names,
    )
    fig = plt.figure(figsize=(10, max(6, len(feature_names) * 0.32)))
    shap.plots.beeswarm(expl, max_display=len(feature_names), show=False)
    plt.title("SHAP Beeswarm — impact on MPI prediction")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved shap_summary.png")

    # --- 3. Dependence plots for top 12 features ---
    top12 = [feature_names[i] for i in order[:12]]
    for feat in top12:
        fig, ax = plt.subplots(figsize=(7, 4))
        fi = feature_names.index(feat)
        ax.scatter(X_df[feat], shap_values[:, fi],
                   alpha=0.3, s=8, c=shap_values[:, fi],
                   cmap="RdBu_r", vmin=-0.1, vmax=0.1)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP dependence: {feat}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        safe = feat.replace("/", "_")
        plt.savefig(OUT_DIR / f"shap_dependence_{safe}.png", dpi=120, bbox_inches="tight")
        plt.close()
    print(f"Saved dependence plots for top 12 features")

    # --- 4. SHAP correlation matrix (redundancy) ---
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    top20   = [feature_names[i] for i in order[:20]]
    corr    = shap_df[top20].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(top20)))
    ax.set_yticks(range(len(top20)))
    ax.set_xticklabels(top20, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(top20, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title("SHAP value correlations — top 20 features\n(high correlation = redundant)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_redundancy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved shap_redundancy.png")

    # --- 5. CSV results ---
    results = pd.DataFrame(ranked, columns=["feature", "mean_abs_shap"])
    results.to_csv(OUT_DIR / "shap_results.csv", index=False)
    print(f"\nTop features by mean |SHAP|:")
    print(results.head(20).to_string(index=False))

    return ranked


def main():
    df, features = load_data()

    X_df    = df[features].copy()
    y       = df[TARGET].values.astype(np.float32)
    groups  = df["Country"].values
    X       = X_df.values.astype(np.float32)

    print("\nTraining XGBoost (5-fold GroupKFold)...")
    model, oof = train_xgb_oof(X, y, groups)

    print("\nComputing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print("\nGenerating plots...")
    ranked = plot_shap(shap_values, X_df, features)

    print(f"\nAll outputs saved to {OUT_DIR}/")

    # Print recommended vs drop
    print("\n=== RECOMMENDATION ===")
    keep = [(f, v) for f, v in ranked if v >= 0.005]
    drop = [(f, v) for f, v in ranked if v < 0.005]
    print(f"Keep ({len(keep)} features, mean|SHAP| >= 0.005):")
    for f, v in keep:
        print(f"  {f:<28s}  {v:.5f}")
    print(f"\nDrop ({len(drop)} features, mean|SHAP| < 0.005):")
    for f, v in drop:
        print(f"  {f:<28s}  {v:.5f}")


if __name__ == "__main__":
    main()
