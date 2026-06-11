"""
compute_anomalies_from_final_merged.py

Computes climate anomaly features from the complete GEE panel (no year gaps)
and joins them onto Final_Merged, producing Final_Merged_with_anomalies.csv.

Source for anomalies : gee_all_vars_added/all_82_merged_500m.csv
  - 17,472 rows, 84 countries, 2012-2023, zero year gaps
  - All variables exported with 500m building mask

Source for base features + MPI : Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv

Anomaly method (mirrors anomalies/ifad_pov_anomalies.ipynb):
  Baseline : entity mean and std over 2012-2019
  Anomaly  : (value - baseline_mean) / baseline_std  (z-score)
  Lag-1    : anomaly value for year t-1 (year-based lookup, not row shift)

Variables:
  Median NDVI        -> NDVI_anom
  Mean LST (K)       -> LSTN_anom    (nighttime, VIIRS VNP21A1N)
  Mean LST_Day (K)   -> LST_Day_anom (daytime,   MODIS MOD11A1)
  Mean NTL           -> NTL_anom
  Mean GPP           -> GPP_anom
  Mean_PDSI          -> PDSI_anom
  Sum_Precip         -> precipitation_anom

Output: Final_Merged_with_anomalies.csv
  All original columns from Final_Merged + 14 anomaly columns
  (7 variables × anom + anom_lag1)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR    = Path(__file__).resolve().parent
PANEL_FILE  = BASE_DIR / "gee_all_vars_added" / "all_82_merged_500m.csv"
SOURCE_FILE = BASE_DIR / "Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv"
OUT_FILE    = BASE_DIR / "Final_Merged_with_anomalies.csv"

# Panel column -> anomaly variable name
VAR_MAP = {
    "Median_NDVI":   "NDVI",
    "Mean_LST":      "LSTN",
    "Mean_LST_Day":  "LST_Day",
    "Mean_NTL":      "NTL",
    "Mean_GPP":      "GPP",
    "Mean_PDSI":     "PDSI",
    "Sum_Precip":    "precipitation",
}

ALL_VARS  = list(VAR_MAP.values())
ANOM_COLS = [f"{v}_anom" for v in ALL_VARS] + [f"{v}_anom_lag1" for v in ALL_VARS]

# GAUL country names -> Final_Merged country names
COUNTRY_FIX = {
    "united republic of tanzania":              "tanzania",
    "the former yugoslav republic of macedonia": "north macedonia",
}


def compute_anomalies(df: pd.DataFrame, var_col: str, var_name: str) -> pd.DataFrame:
    """Add z-score anomaly and year-based lag-1 anomaly for one variable."""
    entity_col = "_entity"
    anom_col   = f"{var_name}_anom"
    lag_col    = f"{var_name}_anom_lag1"

    baseline = (
        df[df["Year"].between(2012, 2019)]
        .groupby(entity_col)[var_col]
        .agg(mean="mean", std="std")
    )
    df = df.merge(baseline, left_on=entity_col, right_index=True, how="left")
    df[anom_col] = (df[var_col] - df["mean"]) / df["std"]

    # Year-based lag: look up year t-1 explicitly so panel gaps don't bleed wrong years
    lag_lookup = (
        df[[entity_col, "Year", anom_col]]
        .rename(columns={"Year": "_lag_year", anom_col: lag_col})
    )
    df["_lag_year"] = df["Year"] - 1
    df = df.merge(lag_lookup, on=[entity_col, "_lag_year"], how="left")
    df = df.drop(columns=["mean", "std", "_lag_year"])
    return df


def main() -> None:
    # ── Load complete GEE panel ───────────────────────────────────────────────
    print(f"Reading {PANEL_FILE.name} ...")
    panel = pd.read_csv(PANEL_FILE, encoding="utf-8")
    print(f"  {len(panel)} rows | {panel['Country'].nunique()} countries | "
          f"years {panel['Year'].min()}-{panel['Year'].max()}")

    panel["_entity"] = panel["Country"].str.strip() + " | " + panel["Region"].str.strip()
    panel = panel.sort_values(["_entity", "Year"])

    # ── Compute anomalies on full panel ───────────────────────────────────────
    for var_col, var_name in VAR_MAP.items():
        if var_col not in panel.columns:
            print(f"  WARNING: {var_col} not found in panel — skipping {var_name}_anom")
            panel[f"{var_name}_anom"]      = float("nan")
            panel[f"{var_name}_anom_lag1"] = float("nan")
            continue
        panel = compute_anomalies(panel, var_col, var_name)
        filled = panel[f"{var_name}_anom"].notna().sum()
        print(f"  {var_name}_anom: {filled}/{len(panel)} filled "
              f"({filled / len(panel) * 100:.1f}%)")

    # ── Save full panel with anomalies ───────────────────────────────────────
    panel_out = PANEL_FILE.parent / "all_82_merged_500m_with_anomalies_MPI.csv"
    panel_clean = panel.drop(columns=["_entity"])
    panel_clean.to_csv(panel_out, index=False, encoding="utf-8")
    print(f"\nSaved {panel_out.name} | {len(panel_clean)} rows x {len(panel_clean.columns)} cols")

    # ── Normalize country names for join onto Final_Merged ───────────────────
    panel["_country_key"] = (
        panel["Country"].str.strip().str.lower()
        .map(lambda x: COUNTRY_FIX.get(x, x))
    )
    panel["_region_key"] = panel["Region"].str.strip().str.lower()

    # ── Load Final_Merged ─────────────────────────────────────────────────────
    print(f"\nReading {SOURCE_FILE.name} ...")
    fm = pd.read_csv(SOURCE_FILE, encoding="utf-8")
    print(f"  {len(fm)} rows | {fm['Country'].nunique()} countries")

    fm["_country_key"] = fm["Country"].str.strip().str.lower()
    fm["_region_key"]  = fm["Region"].str.strip().str.lower()

    # ── Join anomaly columns onto Final_Merged ────────────────────────────────
    merge_cols = ["_country_key", "_region_key", "Year"] + ANOM_COLS
    panel_anom = panel[merge_cols].drop_duplicates(["_country_key", "_region_key", "Year"])

    fm = fm.merge(panel_anom, on=["_country_key", "_region_key", "Year"], how="left")

    for col in ANOM_COLS:
        filled = fm[col].notna().sum()
        print(f"  {col}: {filled}/{len(fm)} filled ({filled / len(fm) * 100:.1f}%)")

    # ── Clean up and save ─────────────────────────────────────────────────────
    fm = fm.drop(columns=[c for c in fm.columns if c.startswith("_")])
    fm.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_FILE.name} | {len(fm)} rows x {len(fm.columns)} cols")


if __name__ == "__main__":
    main()
