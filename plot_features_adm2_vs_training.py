"""
plot_features_adm2_vs_training.py

Compares 12 model features between:
  adm2_all_vars.csv    (ADM2-level, 9 countries, aggregated to Country+Year)
  Final_Merged_...csv  (ADM1-level training panel, filtered to same 9 countries)

Both are averaged to Country+Year level before comparison.
ndvi_lst_ratio computed from adm2_all_vars as Median_NDVI / Mean_LST.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from pathlib import Path

BASE      = Path(__file__).resolve().parent
FILE_ADM2 = BASE / "adm2_vars_9countries" / "adm2_all_vars.csv"
FILE_TRN  = BASE / "Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv"
OUT       = BASE / "plot_features_adm2_vs_final_merged.png"

FEATURES = [
    "Mean_GPP", "StdDev_GPP", "Median_Pop", "StdDev_Pop",
    "Mean_LST", "StdDev_LST", "Mean_NTL", "StdDev_NTL", "Sum_NTL",
    "Median_NDVI", "StdDev_NDVI", "ndvi_lst_ratio",
]

# ── Load & prep adm2_all_vars ─────────────────────────────────────────────────
adm2 = pd.read_csv(FILE_ADM2, encoding="utf-8")
den = adm2["Mean_LST"].replace(0, np.nan)
adm2["ndvi_lst_ratio"] = (adm2["Median_NDVI"] / den).fillna(0.0)

# Aggregate ADM2 → Country+Year (mean across districts)
adm2_agg = (
    adm2.groupby(["Country", "Year"])[FEATURES]
    .mean()
    .reset_index()
)

# ── Load & prep Final_Merged ──────────────────────────────────────────────────
trn = pd.read_csv(FILE_TRN, encoding="utf-8")

# Filter to the 9 countries present in adm2_all_vars
countries_9 = adm2["Country"].unique()
trn = trn[trn["Country"].isin(countries_9)]

# Aggregate ADM1 → Country+Year (mean across regions)
trn_agg = (
    trn.groupby(["Country", "Year"])[FEATURES]
    .mean()
    .reset_index()
)

# ── Merge on Country + Year ───────────────────────────────────────────────────
merged = adm2_agg.merge(trn_agg, on=["Country", "Year"],
                        suffixes=("_adm2", "_trn"), how="inner")
print(f"Matched Country+Year pairs: {len(merged)}")
print(f"Countries: {sorted(merged['Country'].unique())}")
print(f"Years: {sorted(merged['Year'].unique())}")

# ── Plot ──────────────────────────────────────────────────────────────────────
COUNTRIES = sorted(merged["Country"].unique())
CMAP      = plt.colormaps.get_cmap("tab10")
COLOR     = {c: CMAP(i / max(len(COUNTRIES) - 1, 1)) for i, c in enumerate(COUNTRIES)}

n_cols = 4
n_rows = int(np.ceil(len(FEATURES) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = axes.flatten()

for ax, feat in zip(axes, FEATURES):
    xcol = f"{feat}_adm2"
    ycol = f"{feat}_trn"
    sub  = merged.dropna(subset=[xcol, ycol])

    for country in COUNTRIES:
        s = sub[sub["Country"] == country]
        ax.scatter(s[xcol], s[ycol], s=20, alpha=0.7,
                   color=COLOR[country], label=country)

    lo = min(sub[xcol].min(), sub[ycol].min())
    hi = max(sub[xcol].max(), sub[ycol].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)

    r   = np.corrcoef(sub[xcol], sub[ycol])[0, 1] if len(sub) > 1 else float("nan")
    mae = np.mean(np.abs(sub[xcol] - sub[ycol]))
    ax.set_title(f"{feat}\nr={r:.3f}   MAE={mae:.4f}", fontsize=10)
    ax.set_xlabel("adm2_all_vars (mean over ADM2s)")
    ax.set_ylabel("Final_Merged (mean over ADM1s)")

for ax in axes[len(FEATURES):]:
    ax.set_visible(False)

legend_handles = [
    mlines.Line2D([], [], marker="o", color="w",
                  markerfacecolor=COLOR[c], markersize=7, label=c)
    for c in COUNTRIES
] + [mlines.Line2D([], [], color="k", linestyle="--", label="y = x")]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=9, bbox_to_anchor=(0.5, -0.03))

fig.suptitle(
    "Feature comparison: adm2_all_vars (ADM2 mean)  vs  Final_Merged (ADM1 mean)\n"
    "Matched on Country + Year  |  9 training countries",
    fontsize=13,
)
plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nSaved {OUT.name}")
plt.show()
