"""
Run the new DNN+XGBoost ensemble and the DNN+LightGBM ensemble over the Turkey
feature cache, compare their predicted MPI, and plot them against each other.

Usage:
    python compare_tr_models.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import predict_tr_features as P

FEATURES_CSV = "tr_features_2012_2024.csv"
ALPHA = 0.15
OUT_CSV = "tr_xgb_vs_lgbm.csv"
OUT_PNG = "tr_xgb_vs_lgbm.png"


def main():
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} rows from {FEATURES_CSV}")

    xgb_pred  = P.predict(df, "DNN+XGBoost",  ALPHA)
    lgbm_pred = P.predict(df, "DNN+LightGBM", ALPHA)

    out = df[["region_code", "year"]].copy()
    out["MPI_XGB"]  = xgb_pred
    out["MPI_LGBM"] = lgbm_pred
    out["diff"] = out["MPI_XGB"] - out["MPI_LGBM"]
    out = out.sort_values(["region_code", "year"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    mae  = np.mean(np.abs(out["diff"]))
    corr = np.corrcoef(xgb_pred, lgbm_pred)[0, 1]
    print(f"\nDNN+XGBoost : min={xgb_pred.min():.4f} max={xgb_pred.max():.4f} mean={xgb_pred.mean():.4f}")
    print(f"DNN+LightGBM: min={lgbm_pred.min():.4f} max={lgbm_pred.max():.4f} mean={lgbm_pred.mean():.4f}")
    print(f"Mean |XGB - LGBM| = {mae:.4f}   corr = {corr:.3f}")
    print(f"\nSaved comparison -> {OUT_CSV}")

    # ── Plot: scatter (XGB vs LGBM) + per-region mean bars ──────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter with y=x reference
    ax1.scatter(out["MPI_LGBM"], out["MPI_XGB"], alpha=0.6, s=30, edgecolor="k", linewidth=0.3)
    lim = max(out["MPI_XGB"].max(), out["MPI_LGBM"].max()) * 1.05
    ax1.plot([0, lim], [0, lim], "r--", linewidth=1, label="y = x")
    ax1.set_xlabel("DNN+LightGBM  Predicted MPI")
    ax1.set_ylabel("DNN+XGBoost  Predicted MPI")
    ax1.set_title(f"XGB vs LGBM (n={len(out)})\ncorr={corr:.3f}  mean|diff|={mae:.4f}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Per-region mean comparison (sorted by LGBM)
    by_reg = out.groupby("region_code")[["MPI_XGB", "MPI_LGBM"]].mean()
    by_reg = by_reg.sort_values("MPI_LGBM")
    x = np.arange(len(by_reg))
    w = 0.4
    ax2.bar(x - w/2, by_reg["MPI_XGB"],  w, label="DNN+XGBoost",  color="#1f77b4")
    ax2.bar(x + w/2, by_reg["MPI_LGBM"], w, label="DNN+LightGBM", color="#ff7f0e")
    ax2.set_xticks(x)
    ax2.set_xticklabels(by_reg.index, rotation=90, fontsize=7)
    ax2.set_ylabel("Mean Predicted MPI (2013-2024)")
    ax2.set_title("Per-region mean predicted MPI")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"Saved plot       -> {OUT_PNG}")


if __name__ == "__main__":
    main()
