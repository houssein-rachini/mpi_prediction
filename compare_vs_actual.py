"""
Compare predicted MPI (DNN+XGBoost vs DNN+LightGBM) against the official Turkish
relative income-poverty rates (pov-turkey-2014-2024.xlsx), by NUTS region/year.

NOTE: the actual metric is RELATIVE income poverty (% below 50%/60% of median
income), which is conceptually different from MPI (multidimensional/absolute).
This compares pattern/ranking agreement, not absolute levels.

Usage:
    python compare_vs_actual.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ACTUAL_XLSX = "../../pov-turkey-2014-2024.xlsx"   # relative to mpi_prediction/
PRED_CSV    = "tr_xgb_vs_lgbm.csv"
OUT_PNG     = "tr_pred_vs_actual.png"
OUT_CSV     = "tr_pred_vs_actual.csv"


def load_actual():
    raw = pd.read_excel(ACTUAL_XLSX, sheet_name="Sheet1", header=None, skiprows=2)
    raw.columns = ["Name", "Code", "Year", "thr50", "num50", "rate50",
                   "thr60", "num60", "rate60"]
    raw = raw.dropna(subset=["Code", "Year"])
    raw["Code"] = raw.Code.astype(str).str.strip()
    # NUTS-2 codes are TR + digit OR letter (TR10..TR90, TRA1..TRC3); drop national 'TR'.
    raw = raw[raw.Code.str.match(r"^TR[0-9A-Z]", na=False) & (raw.Code != "TR")]
    raw["Year"] = raw.Year.astype(int)
    for c in ["rate50", "rate60"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return raw[["Code", "Year", "rate50", "rate60"]]


def main():
    pred = pd.read_csv(PRED_CSV)
    actual = load_actual()

    m = pred.merge(actual, left_on=["region_code", "year"],
                   right_on=["Code", "Year"], how="inner").dropna(subset=["rate60"])
    m.to_csv(OUT_CSV, index=False)

    missing = sorted(set(pred.region_code) - set(actual.Code))
    print(f"Merged {len(m)} region-years ({m.region_code.nunique()} regions).")
    print(f"Regions with no actual data (excluded): {missing}")

    print("\nPooled correlation (predicted MPI vs actual poverty rate):")
    for col, lab in [("rate50", "50% median"), ("rate60", "60% median")]:
        cx = np.corrcoef(m.MPI_XGB, m[col])[0, 1]
        cl = np.corrcoef(m.MPI_LGBM, m[col])[0, 1]
        print(f"  vs {lab}: XGB={cx:+.3f}  LGBM={cl:+.3f}")

    # Per-region means (collapse years) -> ranking agreement
    reg = m.groupby("region_code").agg(
        XGB=("MPI_XGB", "mean"), LGBM=("MPI_LGBM", "mean"),
        actual60=("rate60", "mean")).sort_values("actual60")
    print("\nPer-region mean ranking correlation (Spearman):")
    print(f"  XGB  vs actual60: {reg.XGB.corr(reg.actual60, method='spearman'):+.3f}")
    print(f"  LGBM vs actual60: {reg.LGBM.corr(reg.actual60, method='spearman'):+.3f}")

    # ── Plots ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    for ax, pc, name, color in [(axes[0], "MPI_XGB", "DNN+XGBoost", "#1f77b4"),
                                (axes[1], "MPI_LGBM", "DNN+LightGBM", "#ff7f0e")]:
        ax.scatter(m["rate60"], m[pc], alpha=0.6, s=30, color=color, edgecolor="k", linewidth=0.3)
        c = np.corrcoef(m[pc], m["rate60"])[0, 1]
        ax.set_xlabel("Actual poverty rate % (60% median income)")
        ax.set_ylabel(f"{name}  Predicted MPI")
        ax.set_title(f"{name} vs actual\ncorr = {c:+.3f}")
        ax.grid(alpha=0.3)

    # Per-region: actual (right axis) vs predicted (left axis)
    ax3 = axes[2]
    x = np.arange(len(reg))
    w = 0.35
    ax3.bar(x - w/2, reg.XGB,  w, label="XGB MPI",  color="#1f77b4")
    ax3.bar(x + w/2, reg.LGBM, w, label="LGBM MPI", color="#ff7f0e")
    ax3.set_ylabel("Mean Predicted MPI")
    ax3.set_xticks(x)
    ax3.set_xticklabels(reg.index, rotation=90, fontsize=7)
    axr = ax3.twinx()
    axr.plot(x, reg.actual60, "k-o", ms=4, label="Actual poverty % (60%)")
    axr.set_ylabel("Actual poverty rate % (60% median)")
    ax3.set_title("Per-region: predicted MPI vs actual poverty rate")
    ax3.legend(loc="upper left", fontsize=8)
    axr.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\nSaved plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()
