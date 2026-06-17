"""
plot_ratio_comparison.py

Two scatter plots (side by side):
  Left : predicted_MPI_no_ratio   (adm2_predictions_9countries.csv, 2024)
         vs predicted_MPI         (adm2_predictions_training_countries.csv, 2024)
  Right: predicted_MPI_with_ratio (adm2_predictions_9countries.csv, 2024)
         vs predicted_MPI         (adm2_predictions_training_countries.csv, 2024)

Joined on adm2_code + year, coloured by country.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from pathlib import Path

FILE_9C  = Path(r"c:\Users\ha333\Desktop\22-MPI Prediction\6-NEW_PROD\mpi_prediction\adm2_predictions_9countries.csv")
FILE_TC  = Path(r"C:\Users\ha333\Desktop\MPI_Data_Requested\UPDATED\adm2_predictions_training_countries.csv")
OUT      = Path(__file__).resolve().parent / "plot_ratio_comparison.png"

df9 = pd.read_csv(FILE_9C, encoding="utf-8")
dft = pd.read_csv(FILE_TC, encoding="utf-8")

merged = df9.merge(
    dft[["adm2_code", "year", "predicted_MPI"]],
    on=["adm2_code", "year"],
    how="inner",
)
print(f"Matched rows: {len(merged)} | countries: {merged['country'].nunique()}")

COUNTRIES = sorted(merged["country"].unique())
CMAP      = plt.colormaps.get_cmap("tab10")
COLOR     = {c: CMAP(i / max(len(COUNTRIES) - 1, 1)) for i, c in enumerate(COUNTRIES)}

PAIRS = [
    ("predicted_MPI_no_ratio",   "No-ratio (ndvi_lst_ratio = 0)"),
    ("predicted_MPI_with_ratio", "With-ratio (actual ndvi_lst_ratio)"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "adm2_predictions_9countries  vs  adm2_predictions_training_countries  —  2024",
    fontsize=13,
)

for ax, (xcol, xlabel) in zip(axes, PAIRS):
    sub = merged.dropna(subset=[xcol, "predicted_MPI"])
    for country in COUNTRIES:
        s = sub[sub["country"] == country]
        ax.scatter(s[xcol], s["predicted_MPI"],
                   s=14, alpha=0.7, color=COLOR[country], label=country)

    lo = min(sub[xcol].min(), sub["predicted_MPI"].min())
    hi = max(sub[xcol].max(), sub["predicted_MPI"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)

    r   = np.corrcoef(sub[xcol], sub["predicted_MPI"])[0, 1]
    mae = np.mean(np.abs(sub[xcol] - sub["predicted_MPI"]))
    ax.set_title(f"{xlabel}\nr={r:.4f}   MAE={mae:.4f}", fontsize=11)
    ax.set_xlabel("9-countries pipeline")
    ax.set_ylabel("Training-countries pipeline  (predicted_MPI)")

legend_handles = [
    mlines.Line2D([], [], marker="o", color="w",
                  markerfacecolor=COLOR[c], markersize=7, label=c)
    for c in COUNTRIES
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           fontsize=9, bbox_to_anchor=(0.5, -0.06))

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT.name}")
plt.show()
