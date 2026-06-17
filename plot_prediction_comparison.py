"""
plot_prediction_comparison.py

Scatter plot: XGB+DNN Predicted MPI (All_Countries file, 2020-2024)
              vs predicted_MPI (adm2_predictions_training_countries, 2020-2024)

Joined on adm2_code + year. One subplot per year, coloured by country.
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from pathlib import Path

FILE1 = Path(
    r"C:\Users\ha333\Downloads\All_Countries_MPI_POV_XGB_DNN_2020_2024 (2).csv"
)
FILE2 = Path(
    r"C:\Users\ha333\Desktop\MPI_Data_Requested\UPDATED\adm2_predictions_training_countries.csv"
)
OUT = Path(__file__).resolve().parent / "plot_prediction_comparison.png"

df1 = pd.read_csv(FILE1, encoding="utf-8")
df2 = pd.read_csv(FILE2, encoding="utf-8")

merged = df1.merge(df2, on=["adm2_code", "year"], how="inner")
merged = merged.rename(
    columns={
        "XGB+DNN Predicted MPI": "pred_all_countries",
        "predicted_MPI": "pred_training",
        "Country": "country_x",
    }
)

YEARS = sorted(merged["year"].unique())
COUNTRIES = sorted(merged["country_x"].unique())
CMAP = plt.cm.get_cmap("tab10", len(COUNTRIES))
COLOR = {c: CMAP(i) for i, c in enumerate(COUNTRIES)}

n_cols = 3
n_rows = int(np.ceil(len(YEARS) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
axes = axes.flatten()

for ax, year in zip(axes, YEARS):
    sub = merged[merged["year"] == year]
    for country in COUNTRIES:
        s = sub[sub["country_x"] == country]
        if s.empty:
            continue
        ax.scatter(
            s["pred_all_countries"],
            s["pred_training"],
            s=14,
            alpha=0.7,
            color=COLOR[country],
            label=country,
        )

    lo = min(sub["pred_all_countries"].min(), sub["pred_training"].min())
    hi = max(sub["pred_all_countries"].max(), sub["pred_training"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)

    r = np.corrcoef(sub["pred_all_countries"], sub["pred_training"])[0, 1]
    mae = np.mean(np.abs(sub["pred_all_countries"] - sub["pred_training"]))
    ax.set_title(f"{year}   r={r:.3f}   MAE={mae:.4f}", fontsize=11)
    ax.set_xlabel("All_Countries  (XGB+DNN Predicted MPI)")
    ax.set_ylabel("Training pipeline  (predicted_MPI)")

for ax in axes[len(YEARS) :]:
    ax.set_visible(False)

legend_handles = [
    mlines.Line2D(
        [], [], marker="o", color="w", markerfacecolor=COLOR[c], markersize=7, label=c
    )
    for c in COUNTRIES
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=5,
    fontsize=9,
    bbox_to_anchor=(0.5, -0.04),
)

fig.suptitle(
    "XGB+DNN Predicted MPI (All_Countries) vs predicted_MPI (training pipeline)\n"
    "Joined on adm2_code + year  |  2020–2024",
    fontsize=13,
)
plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT.name}")
plt.show()
