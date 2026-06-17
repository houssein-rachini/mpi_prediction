"""
plot_9c_vs_allcountries.py

Scatter: predicted_MPI_no_ratio (adm2_predictions_9countries.csv, 2024)
      vs XGB+DNN Predicted MPI    (All_Countries_MPI_POV_XGB_DNN_2020_2024, 2024)

Joined on adm2_code + year, coloured by country.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from pathlib import Path

FILE_9C  = Path(r"c:\Users\ha333\Desktop\22-MPI Prediction\6-NEW_PROD\mpi_prediction\adm2_predictions_9countries.csv")
FILE_ALL = Path(r"C:\Users\ha333\Downloads\All_Countries_MPI_POV_XGB_DNN_2020_2024 (2).csv")
OUT      = Path(__file__).resolve().parent / "plot_9c_noratio_vs_allcountries.png"

df9  = pd.read_csv(FILE_9C,  encoding="utf-8")
dall = pd.read_csv(FILE_ALL, encoding="utf-8")

merged = df9.merge(dall, on=["adm2_code", "year"], how="inner")
print(f"Matched rows: {len(merged)} | countries: {merged['country'].nunique()}")

COUNTRIES = sorted(merged["country"].unique())
CMAP      = plt.colormaps.get_cmap("tab10")
COLOR     = {c: CMAP(i / max(len(COUNTRIES) - 1, 1)) for i, c in enumerate(COUNTRIES)}

fig, ax = plt.subplots(figsize=(8, 7))

for country in COUNTRIES:
    s = merged[merged["country"] == country]
    ax.scatter(s["predicted_MPI_no_ratio"], s["XGB+DNN Predicted MPI"],
               s=16, alpha=0.7, color=COLOR[country], label=country)

lo = min(merged["predicted_MPI_no_ratio"].min(), merged["XGB+DNN Predicted MPI"].min())
hi = max(merged["predicted_MPI_no_ratio"].max(), merged["XGB+DNN Predicted MPI"].max())
ax.plot([lo, hi], [lo, hi], "k--", lw=1)

r    = np.corrcoef(merged["predicted_MPI_no_ratio"], merged["XGB+DNN Predicted MPI"])[0, 1]
mae  = np.mean(np.abs(merged["predicted_MPI_no_ratio"] - merged["XGB+DNN Predicted MPI"]))
bias = np.mean(merged["XGB+DNN Predicted MPI"] - merged["predicted_MPI_no_ratio"])

ax.set_title(
    f"predicted_MPI_no_ratio (ndvi_lst_ratio=0)  vs  XGB+DNN Predicted MPI  —  2024\n"
    f"r={r:.4f}   MAE={mae:.4f}   bias(All_Countries − 9c)={bias:+.4f}",
    fontsize=11,
)
ax.set_xlabel("9-countries pipeline  (predicted_MPI_no_ratio)")
ax.set_ylabel("All_Countries  (XGB+DNN Predicted MPI)")

legend_handles = [
    mlines.Line2D([], [], marker="o", color="w",
                  markerfacecolor=COLOR[c], markersize=7, label=c)
    for c in COUNTRIES
] + [mlines.Line2D([], [], color="k", linestyle="--", label="y = x")]
ax.legend(handles=legend_handles, fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT.name}")
plt.show()
