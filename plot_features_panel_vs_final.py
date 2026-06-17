"""
plot_features_panel_vs_final.py

Compares 12 model features between:
  mpi_training_panel.csv            (joined as Country + Region + Year)
  Final_Merged_MPI_LST_NTL_NDVI_v4  (same join keys)

One scatter subplot per feature, coloured by country.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from pathlib import Path

BASE      = Path(__file__).resolve().parent
FILE_TRN  = Path(r"C:\Users\ha333\Desktop\MPI_Data_Requested\mpi_training_panel.csv")
FILE_FIN  = BASE / "Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv"
OUT       = BASE / "plot_features_panel_vs_final.png"

FEATURES = [
    "Mean_GPP", "StdDev_GPP", "Median_Pop", "StdDev_Pop",
    "Mean_LST", "StdDev_LST", "Mean_NTL", "StdDev_NTL", "Sum_NTL",
    "Median_NDVI", "StdDev_NDVI", "ndvi_lst_ratio",
]

# ── Load & rename training panel ──────────────────────────────────────────────
trn = pd.read_csv(FILE_TRN, encoding="utf-8")
trn = trn.rename(columns={
    "country":         "Country",
    "adm1_name":       "Region",
    "year":            "Year",
    "Mean_LSTn":       "Mean_LST",
    "StdDev_LSTn":     "StdDev_LST",
    "NDVI_LSTn_ratio": "ndvi_lst_ratio",
})

# ── Load Final_Merged ─────────────────────────────────────────────────────────
fin = pd.read_csv(FILE_FIN, encoding="utf-8")

# ── Merge on Country + Region + Year ─────────────────────────────────────────
merged = trn.merge(fin, on=["Country", "Region", "Year"],
                   suffixes=("_panel", "_final"), how="inner")
print(f"Matched rows: {len(merged)} | countries: {merged['Country'].nunique()}")

COUNTRIES = sorted(merged["Country"].unique())
CMAP      = plt.colormaps.get_cmap("tab20")
COLOR     = {c: CMAP(i % 20 / 20) for i, c in enumerate(COUNTRIES)}

n_cols = 4
n_rows = int(np.ceil(len(FEATURES) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = axes.flatten()

for ax, feat in zip(axes, FEATURES):
    xcol = f"{feat}_panel"
    ycol = f"{feat}_final"
    sub  = merged.dropna(subset=[xcol, ycol])

    for country in COUNTRIES:
        s = sub[sub["Country"] == country]
        ax.scatter(s[xcol], s[ycol], s=10, alpha=0.6,
                   color=COLOR[country], label=country)

    lo = min(sub[xcol].min(), sub[ycol].min())
    hi = max(sub[xcol].max(), sub[ycol].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)

    r   = np.corrcoef(sub[xcol], sub[ycol])[0, 1] if len(sub) > 1 else float("nan")
    mae = np.mean(np.abs(sub[xcol] - sub[ycol]))
    ax.set_title(f"{feat}\nr={r:.4f}   MAE={mae:.4f}", fontsize=10)
    ax.set_xlabel("mpi_training_panel")
    ax.set_ylabel("Final_Merged")

for ax in axes[len(FEATURES):]:
    ax.set_visible(False)

legend_handles = [
    mlines.Line2D([], [], marker="o", color="w",
                  markerfacecolor=COLOR[c], markersize=6, label=c)
    for c in COUNTRIES
] + [mlines.Line2D([], [], color="k", linestyle="--", label="y = x")]

n_legend_cols = min(6, len(COUNTRIES) + 1)
fig.legend(handles=legend_handles, loc="lower center",
           ncol=n_legend_cols, fontsize=7, bbox_to_anchor=(0.5, -0.03))

fig.suptitle(
    "Feature comparison: mpi_training_panel  vs  Final_Merged\n"
    "Matched on Country + Region + Year  |  5281 rows  |  83 countries",
    fontsize=13,
)
plt.tight_layout(rect=[0, 0.06, 1, 0.97])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT.name}")
plt.show()
