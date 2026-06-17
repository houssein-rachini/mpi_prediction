import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with_ratio = pd.read_csv(
    r"C:\Users\ha333\Desktop\MPI_Data_Requested\adm2_predictions_training_countries\adm2_predictions_training_countries.csv"
)
no_ratio = pd.read_csv(
    r"C:\Users\ha333\Desktop\22-MPI Prediction\6-NEW_PROD\mpi_prediction\adm2_predictions_no_ratio.csv"
)

merged = with_ratio[["adm2_code", "predicted_MPI", "country"]].merge(
    no_ratio[["adm2_code", "predicted_MPI"]],
    on="adm2_code", suffixes=("_ratio", "_noratio"), how="inner"
)
print(f"Matched {len(merged)} districts")
print(merged.head(10).to_string())

x = merged["predicted_MPI_ratio"]
y = merged["predicted_MPI_noratio"]
corr = np.corrcoef(x, y)[0, 1]
mae  = np.mean(np.abs(x - y))

fig, ax = plt.subplots(figsize=(8, 8))
countries = merged["country"].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(countries)))
for country, color in zip(countries, colors):
    mask = merged["country"] == country
    ax.scatter(x[mask], y[mask], label=country, alpha=0.7, s=30, color=color)

lims = [min(x.min(), y.min()) - 0.01, max(x.max(), y.max()) + 0.01]
ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("predicted_MPI (with ndvi_lst_ratio)")
ax.set_ylabel("predicted_MPI (no ndvi_lst_ratio)")
ax.set_title(f"Effect of removing ndvi_lst_ratio on ADM2 Predicted MPI\nR={corr:.3f}  MAE={mae:.4f}  n={len(merged)}")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
out = r"C:\Users\ha333\Desktop\22-MPI Prediction\6-NEW_PROD\mpi_prediction\ratio_vs_noratio_plot.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
