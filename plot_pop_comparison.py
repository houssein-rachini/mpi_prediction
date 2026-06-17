"""
Compare population variables between:
  - adm2_features_9countries_cache.csv  (client-side aggregated-stat extrapolation, 2024)
  - adm2_all_vars.csv                   (server-side linearFit extrapolation, year 2024)

Shared districts joined on adm2_code.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

cache = pd.read_csv(BASE / "adm2_features_9countries_cache.csv")
allv  = pd.read_csv(BASE / "adm2_vars_9countries" / "adm2_all_vars.csv")

# Filter all_vars to 2024 and normalise key column
allv2024 = allv[allv["Year"] == 2024].copy()
allv2024 = allv2024.rename(columns={"ADM2_CODE": "adm2_code"})
allv2024["adm2_code"] = pd.to_numeric(allv2024["adm2_code"], errors="coerce")
cache["adm2_code"]    = pd.to_numeric(cache["adm2_code"],    errors="coerce")

merged = cache.merge(
    allv2024[["adm2_code", "Median_Pop", "StdDev_Pop", "Total_Pop"]],
    on="adm2_code", suffixes=("_cache", "_allv"),
)
print(f"Matched districts: {len(merged)}")

POP_PAIRS = [
    ("Median_Pop_cache",  "Median_Pop_allv",  "Median_Pop"),
    ("StdDev_Pop_cache",  "StdDev_Pop_allv",  "StdDev_Pop"),
    ("population_2024",   "Total_Pop",         "Total_Pop / population_2024"),
]

COUNTRIES = sorted(merged["country"].unique())
CMAP = plt.cm.get_cmap("tab10", len(COUNTRIES))
COLOR = {c: CMAP(i) for i, c in enumerate(COUNTRIES)}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Population vars: cache (aggregated-stat extrap) vs all_vars (linearFit extrap)  —  2024",
    fontsize=13,
)

for ax, (xcol, ycol, title) in zip(axes, POP_PAIRS):
    sub = merged.dropna(subset=[xcol, ycol])
    for country in COUNTRIES:
        s = sub[sub["country"] == country]
        ax.scatter(s[xcol], s[ycol], s=12, alpha=0.6,
                   color=COLOR[country], label=country)

    lo = min(sub[xcol].min(), sub[ycol].min())
    hi = max(sub[xcol].max(), sub[ycol].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")

    # correlation
    r = np.corrcoef(sub[xcol], sub[ycol])[0, 1]
    ax.set_title(f"{title}  (r={r:.3f})", fontsize=11)
    ax.set_xlabel("cache  (aggregated-stat extrap)")
    ax.set_ylabel("all_vars  (linearFit extrap)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5,
           fontsize=9, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.05, 1, 1])
out = BASE / "plot_pop_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out.name}")
plt.show()
