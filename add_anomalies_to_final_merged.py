"""
add_anomalies_to_final_merged.py

Generates climate anomaly features from climate-indicators.xlsx and joins them
onto Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv, saving:
  Final_Merged_with_anomalies.csv

Anomaly computation (mirrors anomalies/ifad_pov_anomalies.ipynb):
  - Baseline: 2012-2019 mean and std per ADM2 entity
  - Anomaly: (value - baseline_mean) / baseline_std  (z-score)
  - Lag-1: anomaly of the previous year

Join: ADM2 anomalies aggregated to ADM1 (mean), then joined on
      (Country, Region, Year) to Final_Merged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR     = Path(__file__).resolve().parent
CLIM_FILE    = BASE_DIR / "anomalies" / "climate-indicators.xlsx"
SOURCE_FILE  = BASE_DIR / "Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv"
OUT_FILE     = BASE_DIR / "Final_Merged_with_anomalies.csv"

CLIM_VARS_EARLY = ["NDVI", "LST_Day_1km", "pdsi", "precipitation"]
CLIM_VARS_LATE  = ["LSTN", "NTL", "GPP"]
ALL_VARS        = CLIM_VARS_EARLY + CLIM_VARS_LATE

ANOM_COLS = (
    [f"{v}_anom"      for v in ALL_VARS] +
    [f"{v}_anom_lag1" for v in ALL_VARS]
)

COUNTRY_FIX = {
    "united republic of tanzania":              "tanzania",
    "the former yugoslav republic of macedonia": "north macedonia",
}


def compute_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Unique entity key (same as notebook)
    df["entity"] = (
        df["ADM0_NAME"].astype(str) + " | " +
        df["ADM1_NAME"].astype(str) + " | " +
        df["ADM2_NAME"].astype(str)
    )
    df = df.sort_values(["entity", "year"])

    # Baseline stats per entity (2012-2019)
    baseline = (
        df[df["year"].between(2012, 2019)]
        .groupby("entity")[ALL_VARS]
        .agg(["mean", "std"])
    )
    baseline.columns = ["_".join(col) for col in baseline.columns]
    df = df.merge(baseline, left_on="entity", right_index=True, how="left")

    # Z-score anomalies
    for v in ALL_VARS:
        df[f"{v}_anom"] = (df[v] - df[f"{v}_mean"]) / df[f"{v}_std"]

    # Lag-1 anomalies
    for v in ALL_VARS:
        df[f"{v}_anom_lag1"] = df.groupby("entity")[f"{v}_anom"].shift(1)

    return df[df["year"].between(2012, 2024)].copy()


def main() -> None:
    print(f"Reading {CLIM_FILE.name} ...")
    clim = pd.read_excel(CLIM_FILE)
    print(f"  {len(clim)} rows | {clim['ADM0_NAME'].nunique()} countries | "
          f"years {clim['year'].min()}-{clim['year'].max()}")

    print("Computing anomalies (baseline 2012-2019) ...")
    anom = compute_anomalies(clim)
    print(f"  {len(anom)} rows after filtering to 2012-2024")

    # ── Aggregate ADM2 → ADM1 ────────────────────────────────────────────────
    anom_adm1 = (
        anom.groupby(["ADM0_NAME", "ADM1_NAME", "year"])[ANOM_COLS]
        .mean()
        .reset_index()
    )

    # Normalise join keys
    anom_adm1["_country"] = (
        anom_adm1["ADM0_NAME"].str.strip().str.lower()
        .map(lambda x: COUNTRY_FIX.get(x, x))
    )
    anom_adm1["_region"] = anom_adm1["ADM1_NAME"].str.strip().str.lower()

    # ── Load Final_Merged ─────────────────────────────────────────────────────
    print(f"\nReading {SOURCE_FILE.name} ...")
    fm = pd.read_csv(SOURCE_FILE, encoding="utf-8")
    print(f"  {len(fm)} rows | {fm['Country'].nunique()} countries")

    fm["_country"] = fm["Country"].str.strip().str.lower()
    fm["_region"]  = fm["Region"].str.strip().str.lower()

    # ── Join ──────────────────────────────────────────────────────────────────
    merged = fm.merge(
        anom_adm1[["_country", "_region", "year"] + ANOM_COLS],
        left_on=["_country", "_region", "Year"],
        right_on=["_country", "_region", "year"],
        how="left",
    ).drop(columns=["_country", "_region", "year"])

    filled = merged["NDVI_anom"].notna().sum()
    print(f"\nAnomalies joined: {filled}/{len(merged)} rows matched "
          f"({filled / len(merged) * 100:.1f}%)")

    # Country-level match report
    report = (
        merged.groupby("Country")
        .apply(lambda g: pd.Series({
            "matched": g["NDVI_anom"].notna().sum(),
            "total":   len(g),
        }), include_groups=False)
        .reset_index()
    )
    unmatched = report[report["matched"] == 0]["Country"].tolist()
    partial   = report[(report["matched"] > 0) & (report["matched"] < report["total"])]
    if unmatched:
        print(f"  Countries with 0 matches : {unmatched}")
    if not partial.empty:
        print(f"  Countries with partial matches:\n{partial.to_string(index=False)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    merged.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_FILE.name}  ({len(merged)} rows, {len(merged.columns)} columns)")
    print(f"Anomaly columns: {ANOM_COLS}")


if __name__ == "__main__":
    main()
