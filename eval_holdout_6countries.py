"""
Evaluate the DNN+XGBoost ensemble (retrained WITHOUT the 6 target countries) on
those held-out countries: predict MPI, compare to actual, plot.

The 6 held-out (in-training-normally) target countries:
    Egypt, Jordan, Kyrgyzstan, Morocco, Tajikistan, Tunisia

Usage:
    python eval_holdout_6countries.py [--alpha 0.4]
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import predict_tr_features as P  # reuses preprocess + model loading

CSV = "all_82_merged_500m_with_anomalies_MPI_ghsl.csv"
SIX = ["Egypt", "Jordan", "Kyrgyzstan", "Morocco", "Tajikistan", "Tunisia"]
MF = ["Mean_NTL","Mean_LST","Median_NTL","Mean_LST_Day","NTL_anom","StdDev_NTL",
      "StdDev_Pop","ndvi_lst_ratio","Mean_Pop","Median_Pop","Mean_GPP","Sum_NTL",
      "NDVI_anom","LSTN_anom","LST_Day_anom","NTL_anom_lag1","Mean_BUILT_S",
      "Median_BUILT_S","StdDev_BUILT_S","StdDev_BUILT_V"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="DNN blend weight (ensemble training default = 0.4)")
    args = ap.parse_args()

    df = pd.read_csv(CSV)
    sub = df[df.Country.isin(SIX)].copy()
    sub = sub[sub.MPI.notna() & sub[MF].notna().all(axis=1)].reset_index(drop=True)
    print(f"Held-out rows (6 countries, full features + MPI): {len(sub)}")
    print(sub.Country.value_counts().sort_index().to_string())

    # alpha sweep
    print("\nMAE vs blend alpha (DNN weight):")
    best = (None, 1e9)
    for a in [0.0, 0.15, 0.3, 0.4, 0.5, 1.0]:
        p = P.predict(sub, "DNN+XGBoost", a)
        mae = mean_absolute_error(sub.MPI, p)
        print(f"  alpha={a:.2f}  MAE={mae:.4f}  R2={r2_score(sub.MPI, p):+.3f}")
        if mae < best[1]:
            best = (a, mae)
    print(f"  -> best alpha by MAE: {best[0]} (MAE {best[1]:.4f})")

    # final predictions at requested alpha
    preds = P.predict(sub, "DNN+XGBoost", args.alpha)
    mae = mean_absolute_error(sub.MPI, preds)
    rmse = np.sqrt(mean_squared_error(sub.MPI, preds))
    r2 = r2_score(sub.MPI, preds)
    print(f"\n=== Held-out metrics @ alpha={args.alpha} ===")
    print(f"MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:+.3f}")

    out = sub[["Country", "Region", "Year", "MPI"]].copy()
    out["Predicted_MPI"] = preds
    out.to_csv("holdout_6countries_predictions.csv", index=False)

    # ---- Plot: actual vs predicted, colored by country + per-country MAE ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(SIX)))
    for c, col in zip(SIX, colors):
        m = out.Country == c
        if m.any():
            ax1.scatter(out.MPI[m], out.Predicted_MPI[m], s=28, color=col,
                        alpha=0.7, edgecolor="k", linewidth=0.3, label=c)
    lim = [min(out.MPI.min(), out.Predicted_MPI.min()) - 0.02,
           max(out.MPI.max(), out.Predicted_MPI.max()) + 0.02]
    ax1.plot(lim, lim, "r--", lw=1, label="perfect (y=x)")
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("Actual MPI"); ax1.set_ylabel("Predicted MPI")
    ax1.set_title(f"Held-out 6 countries (model trained WITHOUT them)\n"
                  f"alpha={args.alpha}  MAE={mae:.4f}  R2={r2:+.3f}  n={len(out)}")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    per = (out.assign(abs_err=(out.MPI - out.Predicted_MPI).abs())
           .groupby("Country").agg(MAE=("abs_err", "mean"),
                                    actual=("MPI", "mean"),
                                    pred=("Predicted_MPI", "mean")))
    x = np.arange(len(per)); w = 0.35
    ax2.bar(x - w/2, per.actual, w, label="actual MPI (mean)", color="#1f77b4")
    ax2.bar(x + w/2, per.pred, w, label="predicted MPI (mean)", color="#ff7f0e")
    ax2.set_xticks(x); ax2.set_xticklabels(per.index, rotation=30, ha="right")
    ax2.set_ylabel("MPI"); ax2.set_title("Per-country mean: actual vs predicted")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig("holdout_6countries.png", dpi=130)
    print("\nSaved holdout_6countries.png + holdout_6countries_predictions.csv")


if __name__ == "__main__":
    main()
