"""
plot_global_vs_allcountries.py

Scatter: predicted_MPI (adm2_predictions_training_countries.csv)
      vs XGB+DNN Predicted MPI (All_Countries_MPI_POV_XGB_DNN_2020_2024 (2).csv)

Joined on adm2_code + year, coloured by country.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from pathlib import Path

FILE_GLOBAL = Path(__file__).resolve().parent / "adm2_predictions_training_countries.csv"
FILE_ALL    = Path(r"C:\Users\ha333\Downloads\All_Countries_MPI_POV_XGB_DNN_2020_2024 (2).csv")
OUT         = Path(__file__).resolve().parent / "plot_global_vs_allcountries.png"

df1 = pd.read_csv(FILE_GLOBAL, encoding="utf-8")
df2 = pd.read_csv(FILE_ALL,    encoding="utf-8")

merged = df1.merge(df2, on=["adm2_code", "year"], how="inner")
print(f"Matched: {len(merged)} rows | {merged['country'].nunique()} countries")

COUNTRIES = sorted(merged["country"].unique())
CMAP      = plt.colormaps.get_cmap("tab10")
COLOR     = {c: CMAP(i / max(len(COUNTRIES) - 1, 1)) for i, c in enumerate(COUNTRIES)}

fig, ax = plt.subplots(figsize=(8, 7))

for country in COUNTRIES:
    s = merged[merged["country"] == country]
    ax.scatter(s["predicted_MPI"], s["XGB+DNN Predicted MPI"],
               s=12, alpha=0.6, color=COLOR[country], label=country)

lo = min(merged["predicted_MPI"].min(), merged["XGB+DNN Predicted MPI"].min())
hi = max(merged["predicted_MPI"].max(), merged["XGB+DNN Predicted MPI"].max())
ax.plot([lo, hi], [lo, hi], "k--", lw=1)

r    = np.corrcoef(merged["predicted_MPI"], merged["XGB+DNN Predicted MPI"])[0, 1]
mae  = np.mean(np.abs(merged["predicted_MPI"] - merged["XGB+DNN Predicted MPI"]))
bias = np.mean(merged["XGB+DNN Predicted MPI"] - merged["predicted_MPI"])

ax.set_title(
    f"adm2_predictions_training_countries  vs  All_Countries XGB+DNN Predicted MPI\n"
    f"r={r:.4f}   MAE={mae:.4f}   bias={bias:+.4f}",
    fontsize=11,
)
ax.set_xlabel("adm2_predictions_training_countries  (predicted_MPI)")
ax.set_ylabel("All_Countries  (XGB+DNN Predicted MPI)")

handles = [
    mlines.Line2D([], [], marker="o", color="w",
                  markerfacecolor=COLOR[c], markersize=7, label=c)
    for c in COUNTRIES
] + [mlines.Line2D([], [], color="k", linestyle="--", label="y = x")]
fig.legend(handles=handles, loc="lower center", ncol=5,
           fontsize=9, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT.name}")
plt.show()
