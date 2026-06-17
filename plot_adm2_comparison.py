import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

no_ratio = pd.read_csv(
    r"C:\Users\ha333\Desktop\22-MPI Prediction\6-NEW_PROD\mpi_prediction\adm2_predictions_no_ratio.csv"
)
original = pd.read_csv(
    r"C:\Users\ha333\Desktop\22-MPI Prediction\6-NEW_PROD\mpi_prediction\All_Countries_MPI_POV_XGB_DNN_2020_2024 (1).csv"
)

orig_2024 = original[original["year"] == 2024][
    ["adm2_code", "XGB+DNN Predicted MPI"]
].copy()
merged = no_ratio.merge(orig_2024, on="adm2_code", how="inner")
print(f"Matched {len(merged)} districts")
print(
    merged[["country", "adm2_name", "predicted_MPI", "XGB+DNN Predicted MPI"]]
    .head(10)
    .to_string()
)

x = merged["XGB+DNN Predicted MPI"]
y = merged["predicted_MPI"]

corr = np.corrcoef(x, y)[0, 1]
mae = np.mean(np.abs(x - y))

fig, ax = plt.subplots(figsize=(8, 8))
countries = merged["country"].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(countries)))
for country, color in zip(countries, colors):
    mask = merged["country"] == country
    ax.scatter(x[mask], y[mask], label=country, alpha=0.7, s=30, color=color)

lims = [min(x.min(), y.min()) - 0.01, max(x.max(), y.max()) + 0.01]
ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("XGB+DNN Predicted MPI (original, 2024)")
ax.set_ylabel("predicted_MPI (no ndvi_lst_ratio, 2024)")
ax.set_title(
    f"ADM2 Predicted MPI — Original vs No-Ratio\nR={corr:.3f}  MAE={mae:.4f}  n={len(merged)}"
)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
out = r"C:\Users\ha333\Desktop\22-MPI Prediction\6-NEW_PROD\mpi_prediction\adm2_comparison_plot.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
