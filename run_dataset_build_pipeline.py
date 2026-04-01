"""
Run the full dataset-build pipeline in one command.

Expected input files (inside `base_dir`, parameterized by `prefix`):
1) gee_all_vars_{prefix}_original_ref_gaul/all_82_population_{prefix}_original_ref_gaul_actual.csv
2) gee_all_vars_{prefix}_original_ref_gaul/all_82_gpp_{prefix}_original_ref_gaul.csv
3) gee_all_vars_{prefix}_original_ref_gaul/all_82_lst_{prefix}_original_ref_gaul.csv
4) gee_all_vars_{prefix}_original_ref_gaul/all_82_ntl_{prefix}_original_ref_gaul.csv
5) gee_all_vars_{prefix}_original_ref_gaul/all_82_ndvi_{prefix}_original_ref_gaul.csv
6) unmasked_Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv

Generated outputs (inside `out_dir`, default: `out_{prefix}`):
- pop.csv       : population table after 2021-2023 extrapolation
- merged.csv    : merged POP/GPP/LST/NTL/NDVI table
- with_mpi.csv  : merged table after adding MPI and dropping rows without MPI
- final.csv     : renamed/reordered features + ndvi_lst_ratio + no missing rows
- summary.txt   : run summary with row counts
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TARGET_YEARS = [2021, 2022, 2023]
KEY_COLS = ["Country", "Region", "Year"]
EXTRA_COLS = ["system:index", ".geo"]

POP_METRIC_COLS = [
    "Mean Population",
    "Total Population",
    "Min Population",
    "Max Population",
    "Median Population",
    "Std Dev Population",
]

COUNTRY_NAME_MAP = {
    "Tanzania": "United Republic of Tanzania",
    "Tanzania, United Republic of": "United Republic of Tanzania",
    "Bolivia, Plurinational State of": "Bolivia",
    "Congo, Democratic Republic of the": "Congo",
    "Democratic Republic of Congo": "Congo",
    "Republic of Congo": "Congo",
    "Congo, Republic of": "Congo",
    "Cote d'Ivoire": "C?te d'Ivoire",
    "Lao": "Lao People's Democratic Republic",
    "Lao PDR": "Lao People's Democratic Republic",
    "Laos": "Lao People's Democratic Republic",
    "Macedonia": "The former Yugoslav Republic of Macedonia",
    "TFYR of Macedonia": "The former Yugoslav Republic of Macedonia",
    "North Macedonia": "The former Yugoslav Republic of Macedonia",
    "Moldova": "Moldova, Republic of",
    "Timor Leste": "Timor-Leste",
    "Vietnam": "Viet Nam",
}

RENAME_MAP = {
    "Mean Population": "Mean_Pop",
    "Total Population": "Total_Pop",
    "Min Population": "Min_Pop",
    "Max Population": "Max_Pop",
    "Median Population": "Median_Pop",
    "Std Dev Population": "StdDev_Pop",
    "Max GPP": "Max_GPP",
    "Median GPP": "Median_GPP",
    "Min GPP": "Min_GPP",
    "Std Dev GPP": "StdDev_GPP",
    "Total GPP": "Sum_GPP",
    "Mean GPP": "Mean_GPP",
    "Max LST (K)": "Max_LST",
    "Mean LST (K)": "Mean_LST",
    "Median LST (K)": "Median_LST",
    "Min LST (K)": "Min_LST",
    "Std Dev LST": "StdDev_LST",
    "Total LST": "Sum_LST",
    "Max NTL": "Max_NTL",
    "Mean NTL": "Mean_NTL",
    "Median NTL": "Median_NTL",
    "Min NTL": "Min_NTL",
    "Std Dev NTL": "StdDev_NTL",
    "Total NTL": "Sum_NTL",
    "Max NDVI": "Max_NDVI",
    "Mean NDVI": "Mean_NDVI",
    "Median NDVI": "Median_NDVI",
    "Min NDVI": "Min_NDVI",
    "Std Dev NDVI": "StdDev_NDVI",
    "Total NDVI": "Sum_NDVI",
}

COLUMN_ORDER = [
    "Country",
    "Region",
    "Year",
    "Max_GPP",
    "Median_GPP",
    "Min_GPP",
    "StdDev_GPP",
    "Sum_GPP",
    "Mean_GPP",
    "Total_Pop",
    "Mean_Pop",
    "Min_Pop",
    "Max_Pop",
    "Median_Pop",
    "StdDev_Pop",
    "Max_LST",
    "Mean_LST",
    "Median_LST",
    "Min_LST",
    "StdDev_LST",
    "Sum_LST",
    "Max_NTL",
    "Mean_NTL",
    "Median_NTL",
    "Min_NTL",
    "StdDev_NTL",
    "Sum_NTL",
    "Max_NDVI",
    "Mean_NDVI",
    "Median_NDVI",
    "Min_NDVI",
    "StdDev_NDVI",
    "Sum_NDVI",
    "ndvi_lst_ratio",
    "MPI",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for input root, dataset prefix, and output folder."""
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run merge+MPI dataset build pipeline for a selected buffer prefix."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="3000m",
        help="Dataset prefix used in filenames and GEE folder (examples: 1000m, 2000m, 3000m).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=base_dir,
        help="Folder containing the input CSV files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder for compactly named pipeline files (default: base_dir/out_{prefix}).",
    )
    return parser.parse_args()


def normalize_precision(value: float) -> float:
    """Round long floating values to 6 decimals, keeping shorter precision unchanged."""
    if pd.isna(value):
        return np.nan
    text = format(float(value), ".15f").rstrip("0").rstrip(".")
    if "." not in text:
        return float(value)
    decimals = len(text.split(".", 1)[1])
    if decimals <= 6:
        return float(value)
    return round(float(value), 6)


def extrapolate(values: np.ndarray, years: np.ndarray, target_year: int) -> float:
    """
    Extrapolate one metric for one target year using mean annual growth.

    Requires at least two finite observations with distinct years.
    """
    values = np.asarray(values, dtype=float)
    years = np.asarray(years, dtype=int)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    valid_vals = values[mask]
    valid_years = years[mask]
    year_diffs = np.diff(valid_years)
    value_diffs = np.diff(valid_vals)
    valid_growth = year_diffs != 0
    if not valid_growth.any():
        return np.nan
    growth = np.mean(value_diffs[valid_growth] / year_diffs[valid_growth])
    estimate = valid_vals[-1] + growth * (target_year - valid_years[-1])
    return normalize_precision(estimate)


def load_metric(path: Path) -> pd.DataFrame:
    """Load a metric CSV, drop GEE extra columns, and coerce Year to Int64."""
    df = pd.read_csv(path, encoding="utf-8")
    df = df.drop(columns=[c for c in EXTRA_COLS if c in df.columns], errors="ignore")
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    return df


def interpolate_population(pop_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild 2021-2023 rows using extrapolation from data up to 2020."""
    pop_df = pop_df.copy()
    pop_df["Year"] = pd.to_numeric(pop_df["Year"], errors="coerce").astype("Int64")
    pop_df = pop_df[~pop_df["Year"].isin(TARGET_YEARS)].copy()
    base = pop_df[pop_df["Year"] <= 2020].copy()
    new_rows = []

    for (country, region), group in base.groupby(["Country", "Region"], dropna=False):
        group = group.sort_values("Year")
        years = group["Year"].astype(int).to_numpy()
        for target_year in TARGET_YEARS:
            row = {col: np.nan for col in pop_df.columns}
            row["Country"] = country
            row["Region"] = region
            row["Year"] = target_year
            for col in POP_METRIC_COLS:
                row[col] = extrapolate(group[col].values, years, target_year)
            new_rows.append(row)

    if new_rows:
        pop_df = pd.concat([pop_df, pd.DataFrame(new_rows)], ignore_index=True)
    pop_df = pop_df.sort_values(KEY_COLS).reset_index(drop=True)
    return pop_df


def merge_metrics(pop: pd.DataFrame, gpp: pd.DataFrame, lst: pd.DataFrame, ntl: pd.DataFrame, ndvi: pd.DataFrame) -> pd.DataFrame:
    """Outer-join all metrics on Country/Region/Year."""
    merged = pop.copy()
    for df in [gpp, lst, ntl, ndvi]:
        merged = merged.merge(df, on=KEY_COLS, how="outer")
    return merged.sort_values(KEY_COLS).reset_index(drop=True)


def add_mpi(merged: pd.DataFrame, original_mpi: pd.DataFrame) -> pd.DataFrame:
    """Attach MPI values by key and drop rows where MPI is missing."""
    original = original_mpi[["Country", "Region", "Year", "MPI"]].copy()
    original["Country"] = original["Country"].map(lambda c: COUNTRY_NAME_MAP.get(c, c))
    original["Year"] = pd.to_numeric(original["Year"], errors="coerce").astype("Int64")
    original = original.drop_duplicates(KEY_COLS)

    merged = merged.copy()
    merged["Year"] = pd.to_numeric(merged["Year"], errors="coerce").astype("Int64")
    if "MPI" in merged.columns:
        merged = merged.drop(columns=["MPI"])
    merged = merged.merge(original, on=KEY_COLS, how="left")
    merged = merged.dropna(subset=["MPI"]).reset_index(drop=True)
    return merged


def rename_reorder_and_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to compact headers, create `ndvi_lst_ratio`, and apply final column order."""
    out = df.rename(columns=RENAME_MAP).copy()
    if "Median_NDVI" not in out.columns or "Mean_LST" not in out.columns:
        raise ValueError("Required columns for ratio are missing: Median_NDVI or Mean_LST")

    num = pd.to_numeric(out["Median_NDVI"], errors="coerce")
    den = pd.to_numeric(out["Mean_LST"], errors="coerce")
    ratio = (num / den).where(den != 0)
    out["ndvi_lst_ratio"] = ratio

    missing_cols = [col for col in COLUMN_ORDER if col not in out.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns after rename/reorder: {missing_cols}")

    out = out[COLUMN_ORDER].copy()
    out = out.dropna().reset_index(drop=True)
    return out


def ensure_inputs_exist(paths: list[Path]) -> None:
    """Fail fast if any required input file is missing."""
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing input files: " + "; ".join(missing))


def write_summary(path: Path, lines: list[str]) -> None:
    """Persist a human-readable run summary."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run all steps and save compact outputs in a single folder."""
    args = parse_args()
    prefix = args.prefix.strip()
    if not prefix:
        raise ValueError("Argument --prefix cannot be empty.")
    base_dir = args.base_dir.resolve()
    out_dir = (args.out_dir or (base_dir / f"out_{prefix}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gee_dir = base_dir / f"gee_all_vars_{prefix}_original_ref_gaul"
    paths = {
        "population": gee_dir / f"all_82_population_{prefix}_original_ref_gaul_actual.csv",
        "gpp": gee_dir / f"all_82_gpp_{prefix}_original_ref_gaul.csv",
        "lst": gee_dir / f"all_82_lst_{prefix}_original_ref_gaul.csv",
        "ntl": gee_dir / f"all_82_ntl_{prefix}_original_ref_gaul.csv",
        "ndvi": gee_dir / f"all_82_ndvi_{prefix}_original_ref_gaul.csv",
        "original_mpi": base_dir / "unmasked_Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv",
    }
    ensure_inputs_exist(list(paths.values()))

    pop = interpolate_population(load_metric(paths["population"]))
    pop_out = out_dir / "pop.csv"
    pop.to_csv(pop_out, index=False, encoding="utf-8")

    merged = merge_metrics(
        pop,
        load_metric(paths["gpp"]),
        load_metric(paths["lst"]),
        load_metric(paths["ntl"]),
        load_metric(paths["ndvi"]),
    )
    merged_out = out_dir / "merged.csv"
    merged.to_csv(merged_out, index=False, encoding="utf-8")

    with_mpi = add_mpi(merged, pd.read_csv(paths["original_mpi"], encoding="utf-8"))
    with_mpi_out = out_dir / "with_mpi.csv"
    with_mpi.to_csv(with_mpi_out, index=False, encoding="utf-8")

    final = rename_reorder_and_ratio(with_mpi)
    final_out = out_dir / "final.csv"
    final.to_csv(final_out, index=False, encoding="utf-8")

    summary_lines = [
        f"prefix={prefix}",
        f"base_dir={base_dir}",
        f"out_dir={out_dir}",
        f"rows_pop={len(pop)}",
        f"rows_merged={len(merged)}",
        f"rows_with_mpi={len(with_mpi)}",
        f"rows_final={len(final)}",
        f"cols_final={len(final.columns)}",
        "",
        "Expected inputs:",
        *(f"- {v}" for v in paths.values()),
        "",
        "Outputs:",
        f"- {pop_out}",
        f"- {merged_out}",
        f"- {with_mpi_out}",
        f"- {final_out}",
    ]
    write_summary(out_dir / "summary.txt", summary_lines)

    print(
        {
            "out_dir": str(out_dir),
            "files": ["pop.csv", "merged.csv", "with_mpi.csv", "final.csv", "summary.txt"],
            "rows_final": int(len(final)),
            "cols_final": int(len(final.columns)),
        }
    )


if __name__ == "__main__":
    main()
