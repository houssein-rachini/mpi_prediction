"""
gee_export_adm2_population_v2.py

Re-extrapolates ADM2 population for 2021–2024 using the same aggregated-stat
linear growth method as build_adm2_predictions_9countries.py (_pop_stats_2024):

    growth = mean(diff(values) / diff(years))
    value_t = last_value + growth * (t - last_year)

Input:  adm2_vars_9countries/adm2_population.csv  (actual 2015-2020 used as base;
        existing 2021-2024 pixel-linearFit rows are replaced)
Output: adm2_vars_9countries/adm2_population_v2.csv

Run:
    python gee_export_adm2_population_v2.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "adm2_vars_9countries"
IN_CSV = CSV_DIR / "adm2_population.csv"
OUT_CSV = CSV_DIR / "adm2_population_v2.csv"

BASE_YEARS = list(range(2012, 2021))  # actual WorldPop years used as trend base
EXTRAP_YEARS = list(range(2021, 2025))  # years to re-extrapolate
KEY_COLS = ["Country", "ADM1_CODE", "ADM2_CODE", "ADM2_NAME"]
STAT_COLS = ["Median_Pop", "StdDev_Pop", "Total_Pop"]


def _extrapolate(values: np.ndarray, years: np.ndarray, target: int) -> float | None:
    """Same formula as _pop_stats_2024() in build_adm2_predictions_9countries.py."""
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return None
    v = values[mask]
    y = years[mask]
    growth = np.mean(np.diff(v) / np.diff(y))
    return float(v[-1] + growth * (target - y[-1]))


def main() -> None:
    print(f"Loading {IN_CSV.name} ...")
    df = pd.read_csv(IN_CSV, encoding="utf-8")
    df = df.drop(columns=[c for c in ["system:index", ".geo"] if c in df.columns])
    df["ADM2_CODE"] = pd.to_numeric(df["ADM2_CODE"], errors="coerce").astype("Int64")
    df["ADM1_CODE"] = pd.to_numeric(df["ADM1_CODE"], errors="coerce").astype("Int64")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    print(f"  {len(df)} rows | years {df['Year'].min()}–{df['Year'].max()}")

    base_years_arr = np.array(BASE_YEARS)

    # Keep actual rows for 2015–2020 as-is
    actual = df[df["Year"].isin(BASE_YEARS)][KEY_COLS + ["Year"] + STAT_COLS].copy()

    districts = actual[KEY_COLS].drop_duplicates().to_dict("records")
    extrap_rows: list[dict] = []

    for d in tqdm(districts, desc="Extrapolating 2021-2024", unit="district"):
        sub = actual[
            (actual["Country"] == d["Country"])
            & (actual["ADM2_CODE"] == d["ADM2_CODE"])
        ].set_index("Year")

        extrap: dict = {**d}
        for yr in EXTRAP_YEARS:
            row = {**d, "Year": yr}
            for stat in STAT_COLS:
                vals = np.array(
                    [
                        sub.loc[y, stat] if y in sub.index else np.nan
                        for y in BASE_YEARS
                    ],
                    dtype=float,
                )
                val = _extrapolate(vals, base_years_arr, yr)
                row[stat] = round(val, 5) if val is not None else None
            extrap_rows.append(row)

    extrap_df = pd.DataFrame(extrap_rows)

    df_out = (
        pd.concat([actual, extrap_df], ignore_index=True)
        .sort_values(["Country", "ADM2_CODE", "Year"])
        .reset_index(drop=True)
    )
    df_out = df_out[KEY_COLS + ["Year"] + STAT_COLS]

    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_CSV.name}")
    print(
        f"  {len(df_out)} rows | {df_out['Country'].nunique()} countries | "
        f"years {df_out['Year'].min()}–{df_out['Year'].max()}"
    )
    print(f"  Null counts: {df_out[STAT_COLS].isnull().sum().to_dict()}")


if __name__ == "__main__":
    main()
