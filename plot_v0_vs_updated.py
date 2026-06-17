"""
plot_v0_vs_updated.py

Scatter: predicted_MPI from V0  vs  predicted_MPI from UPDATED
         joined on adm2_code + year (2024 only), coloured by country.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from pathlib import Path

FILE_V0  = Path(r"C:\Users\ha333\Desktop\MPI_Data_Requested\adm2_predictions_training_countries_V0.csv")
FILE_UPD = Path(r"C:\Users\ha333\Desktop\MPI_Data_Requested\UPDATED\adm2_predictions_training_countries.csv")
OUT      = Path(__file__).resolve().parent / "plot_v0_vs_updated.png"

df_v0  = pd.read_csv(FILE_V0,  encoding="utf-8")
df_upd = pd.read_csv(FILE_UPD, encoding="utf-8")

merged = df_v0.merge(
    df_upd[["adm2_code", "year", "predicted_MPI"]],
    on=["adm2_code", "year"],
    how="inner",
    suffixes=("_v0", "_updated"),
)
print(f"Matched rows: {len(merged)} | countries: {merged['country'].nunique()}")

COUNTRIES = sorted(merged["country"].unique())
CMAP      = plt.colormaps.get_cmap("tab10")
COLOR     = {c: CMAP(i / max(len(COUNTRIES) - 1, 1)) for i, c in enumerate(COUNTRIES)}

fig, ax = plt.subplots(figsize=(8, 7))

for country in COUNTRIES:
    s = merged[merged["country"] == country]
    ax.scatter(s["predicted_MPI_v0"], s["predicted_MPI_updated"],
               s=16, alpha=0.7, color=COLOR[country], label=country)

lo = min(merged["predicted_MPI_v0"].min(), merged["predicted_MPI_updated"].min())
hi = max(merged["predicted_MPI_v0"].max(), merged["predicted_MPI_updated"].max())
ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")

r   = np.corrcoef(merged["predicted_MPI_v0"], merged["predicted_MPI_updated"])[0, 1]
mae = np.mean(np.abs(merged["predicted_MPI_v0"] - merged["predicted_MPI_updated"]))
bias = np.mean(merged["predicted_MPI_updated"] - merged["predicted_MPI_v0"])

ax.set_title(
    f"predicted_MPI: V0  vs  UPDATED  —  2024\n"
    f"r={r:.4f}   MAE={mae:.4f}   bias(updated−v0)={bias:+.4f}",
    fontsize=12,
)
ax.set_xlabel("V0  (predicted_MPI)")
ax.set_ylabel("UPDATED  (predicted_MPI)")

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
