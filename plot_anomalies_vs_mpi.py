"""
plot_anomalies_vs_mpi.py

Visualises the correlation between each climate anomaly feature and MPI
using Final_Merged_with_anomalies.csv.

Outputs:
  anomalies_mpi_correlation.png  — bar chart of Pearson r per anomaly column
  anomalies_mpi_scatter.png      — scatter grid (anomaly vs MPI, coloured by country)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy import stats

BASE    = Path(__file__).resolve().parent
IN_FILE = BASE / "Final_Merged_with_anomalies.csv"

ANOM_COLS = [
    "NDVI_anom", "LSTN_anom", "NTL_anom", "GPP_anom",
    "PDSI_anom", "precipitation_anom",
    "NDVI_anom_lag1", "LSTN_anom_lag1", "NTL_anom_lag1", "GPP_anom_lag1",
    "PDSI_anom_lag1", "precipitation_anom_lag1",
]

df = pd.read_csv(IN_FILE)
df = df.dropna(subset=["MPI"])

# ── 1. Bar chart of correlations ──────────────────────────────────────────────
corrs, pvals = [], []
for col in ANOM_COLS:
    sub = df[[col, "MPI"]].dropna()
    r, p = stats.pearsonr(sub[col], sub["MPI"])
    corrs.append(r)
    pvals.append(p)

corr_df = pd.DataFrame({"feature": ANOM_COLS, "r": corrs, "p": pvals})
corr_df = corr_df.sort_values("r")

colors = ["#d62728" if r < 0 else "#1f77b4" for r in corr_df["r"]]
sig    = corr_df["p"] < 0.05

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(corr_df["feature"], corr_df["r"], color=colors, alpha=0.8)

# Mark significant bars with *
for i, (r, s) in enumerate(zip(corr_df["r"], sig)):
    if s:
        ax.text(r + (0.005 if r >= 0 else -0.005), i, "*",
                ha="left" if r >= 0 else "right", va="center", fontsize=12)

ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson r with MPI")
ax.set_title("Correlation of Climate Anomalies with MPI\n(* = p < 0.05)", fontsize=13)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(BASE / "anomalies_mpi_correlation.png", dpi=150, bbox_inches="tight")
print("Saved anomalies_mpi_correlation.png")
plt.close()

# ── 2. Scatter grid ───────────────────────────────────────────────────────────
COUNTRIES = sorted(df["Country"].dropna().unique())
cmap  = plt.colormaps.get_cmap("tab20")
COLOR = {c: cmap((i % 20) / 20) for i, c in enumerate(COUNTRIES)}

n_cols = 4
n_rows = int(np.ceil(len(ANOM_COLS) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for ax, col in zip(axes, ANOM_COLS):
    sub = df[[col, "MPI", "Country"]].dropna()
    for country in COUNTRIES:
        s = sub[sub["Country"] == country]
        if len(s):
            ax.scatter(s[col], s["MPI"], s=8, alpha=0.5, color=COLOR[country])

    r, p = stats.pearsonr(sub[col], sub["MPI"])
    m, b = np.polyfit(sub[col], sub["MPI"], 1)
    xs = np.linspace(sub[col].min(), sub[col].max(), 100)
    ax.plot(xs, m * xs + b, "k--", lw=1.2)
    ax.set_title(f"{col}\nr={r:.3f}{'*' if p < 0.05 else ''}", fontsize=9)
    ax.set_xlabel("Anomaly (z-score)", fontsize=8)
    ax.set_ylabel("MPI", fontsize=8)
    ax.tick_params(labelsize=7)

for ax in axes[len(ANOM_COLS):]:
    ax.set_visible(False)

fig.suptitle("Climate Anomalies vs MPI  (dashed = OLS trend)", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(BASE / "anomalies_mpi_scatter.png", dpi=150, bbox_inches="tight")
print("Saved anomalies_mpi_scatter.png")
plt.show()
