"""
Build mpi_training_panel.csv from Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv.

Adds adm1_code by querying FAO/GAUL_SIMPLIFIED_500m/2015/level1 via Earth Engine.
Fetches all ADM1 codes in a single GEE call, then joins by country+region name.

Also joins climate anomaly variables from panel_anom.xlsx (year t and t-1):
  NDVI_anom, LST_day_anom, LSTn_anom, NTL_anom, GPP_anom, precip_anom, PDSI_anom
panel_anom is at ADM2 level — aggregated to ADM1 (mean) before joining.
Join is left on country+adm1_name+year; unmatched rows get NaN.

Output columns:
  country, adm1_code, adm1_name, year, observed_MPI,
  Mean_NTL, Sum_NTL, StdDev_NTL,
  Median_NDVI, StdDev_NDVI,
  Mean_GPP, StdDev_GPP,
  Mean_LSTn, StdDev_LSTn,
  Median_Pop, StdDev_Pop,
  NDVI_LSTn_ratio,
  NDVI_anom, LST_day_anom, LSTn_anom, NTL_anom, GPP_anom, precip_anom, PDSI_anom,
  NDVI_anom_lag1, LST_day_anom_lag1, LSTn_anom_lag1, NTL_anom_lag1,
  GPP_anom_lag1, precip_anom_lag1, PDSI_anom_lag1
"""

from __future__ import annotations

from pathlib import Path

import ee
import pandas as pd

BASE_DIR    = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv"
OUTPUT_FILE = BASE_DIR / "mpi_training_panel.csv"
ANOM_FILE   = Path(r"C:\Users\ha333\Desktop\MPI_Data_Requested\panel_anom.xlsx")

RENAME_MAP = {
    "Country":       "country",
    "Region":        "adm1_name",
    "Year":          "year",
    "MPI":           "observed_MPI",
    "Mean_LST":      "Mean_LSTn",
    "StdDev_LST":    "StdDev_LSTn",
    "ndvi_lst_ratio": "NDVI_LSTn_ratio",
}

OUTPUT_COLUMNS = [
    "country", "adm1_code", "adm1_name", "year", "observed_MPI",
    "Mean_NTL", "Sum_NTL", "StdDev_NTL",
    "Median_NDVI", "StdDev_NDVI",
    "Mean_GPP", "StdDev_GPP",
    "Mean_LSTn", "StdDev_LSTn",
    "Median_Pop", "StdDev_Pop",
    "NDVI_LSTn_ratio",
    # anomalies year t
    "NDVI_anom", "LST_day_anom", "LSTn_anom", "NTL_anom",
    "GPP_anom", "precip_anom", "PDSI_anom",
    # anomalies year t-1
    "NDVI_anom_lag1", "LST_day_anom_lag1", "LSTn_anom_lag1", "NTL_anom_lag1",
    "GPP_anom_lag1", "precip_anom_lag1", "PDSI_anom_lag1",
]

# Column rename map from panel_anom.xlsx to target names
ANOM_RENAME = {
    "NDVI_anom":               "NDVI_anom",
    "LST_Day_1km_anom":        "LST_day_anom",
    "LSTN_anom":               "LSTn_anom",
    "NTL_anom":                "NTL_anom",
    "GPP_anom":                "GPP_anom",
    "precipitation_anom":      "precip_anom",
    "pdsi_anom":               "PDSI_anom",
    "NDVI_anom_lag1":          "NDVI_anom_lag1",
    "LST_Day_1km_anom_lag1":   "LST_day_anom_lag1",
    "LSTN_anom_lag1":          "LSTn_anom_lag1",
    "NTL_anom_lag1":           "NTL_anom_lag1",
    "GPP_anom_lag1":           "GPP_anom_lag1",
    "precipitation_anom_lag1": "precip_anom_lag1",
    "pdsi_anom_lag1":          "PDSI_anom_lag1",
}

# Same country name normalisation used in run_dataset_build_pipeline.py
COUNTRY_NAME_MAP = {
    "Tanzania": "United Republic of Tanzania",
    "Tanzania, United Republic of": "United Republic of Tanzania",
    "Bolivia, Plurinational State of": "Bolivia",
    "Congo, Democratic Republic of the": "Congo",
    "Democratic Republic of Congo": "Congo",
    "Republic of Congo": "Congo",
    "Congo, Republic of": "Congo",
    "Cote d'Ivoire": "Côte d'Ivoire",
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


def init_ee() -> None:
    """Initialise Earth Engine (local credentials, no Streamlit dependency)."""
    try:
        ee.Initialize()
    except Exception as e:
        raise RuntimeError(
            "Earth Engine initialisation failed. Run `earthengine authenticate` first."
        ) from e


def fetch_gaul_adm1_codes() -> dict[tuple[str, str], int]:
    """
    Pull ADM0_NAME, ADM1_NAME, ADM1_CODE from FAO GAUL level-1 in one GEE call.
    Returns {(adm0_name_lower, adm1_name_lower): adm1_code}.
    """
    print("Fetching ADM1 codes from FAO GAUL via Earth Engine …")
    fc = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")
    data = fc.select(["ADM0_NAME", "ADM1_NAME", "ADM1_CODE"]).getInfo()

    lookup: dict[tuple[str, str], int] = {}
    for feat in data["features"]:
        props = feat["properties"]
        key = (
            str(props["ADM0_NAME"]).strip().lower(),
            str(props["ADM1_NAME"]).strip().lower(),
        )
        lookup[key] = int(props["ADM1_CODE"])

    print(f"  → {len(lookup)} ADM1 entries loaded from GAUL.")
    return lookup


def fix_mojibake(s: str) -> str:
    """Fix UTF-8 bytes that were misread as Latin-1 (e.g. 'Ã©' → 'é')."""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s  # already correct, leave untouched


def join_adm1_codes(df: pd.DataFrame, lookup: dict[tuple[str, str], int]) -> pd.DataFrame:
    """
    Add adm1_code column by matching (country, adm1_name) against the GAUL lookup.
    Normalises country names using COUNTRY_NAME_MAP before matching.
    Rows with no match get adm1_code = pd.NA and are reported.
    """
    df = df.copy()

    # Normalise country names to match GAUL spelling
    df["_country_norm"] = df["country"].map(lambda c: COUNTRY_NAME_MAP.get(c, c))

    def lookup_code(row: pd.Series):
        key = (row["_country_norm"].strip().lower(), row["adm1_name"].strip().lower())
        return lookup.get(key, pd.NA)

    df["adm1_code"] = df.apply(lookup_code, axis=1)

    unmatched = df[df["adm1_code"].isna()][["country", "adm1_name"]].drop_duplicates()
    if not unmatched.empty:
        print(f"\n  ⚠  {len(unmatched)} (country, region) pairs had no GAUL match:")
        print(unmatched.to_string(index=False))
        print()

    df = df.drop(columns=["_country_norm"])
    return df


def load_anom_adm1(path: Path) -> pd.DataFrame:
    """
    Read panel_anom.xlsx, aggregate ADM2 -> ADM1 (mean), rename columns,
    and return a DataFrame keyed on (country, adm1_name, year).
    """
    print(f"Reading anomaly file: {path.name} ...")
    anom = pd.read_excel(path)

    anom["Country"]   = anom["Country"].str.strip()
    anom["ADM1_NAME"] = anom["ADM1_NAME"].str.strip()

    raw_cols = list(ANOM_RENAME.keys())
    adm1 = (
        anom.groupby(["Country", "ADM1_NAME", "year"])[raw_cols]
        .mean()
        .reset_index()
        .rename(columns={
            "Country":   "country",
            "ADM1_NAME": "adm1_name",
            **ANOM_RENAME,
        })
    )
    # normalise join keys
    adm1["country"]   = adm1["country"].str.strip().str.lower()
    adm1["adm1_name"] = adm1["adm1_name"].str.strip().str.lower()
    print(f"  -> {len(adm1)} ADM1-year rows across {adm1['country'].nunique()} countries.")
    return adm1


def main() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")

    # ── 1. Load source ──────────────────────────────────────────────────────
    print(f"Reading {SOURCE_FILE.name} …")
    df = pd.read_csv(SOURCE_FILE, encoding="utf-8")
    print(f"  → {len(df)} rows, {len(df.columns)} columns.")

    # ── 2. Fix mojibake in text columns (UTF-8 bytes misread as Latin-1) ───
    for col in ("Country", "Region"):
        if col in df.columns:
            df[col] = df[col].map(fix_mojibake)

    # ── 3. Rename to target schema ──────────────────────────────────────────
    df = df.rename(columns=RENAME_MAP)

    missing = [c for c in OUTPUT_COLUMNS if c not in ("adm1_code",) and c not in df.columns]
    if missing:
        raise ValueError(f"Source file is missing expected columns: {missing}")

    # ── 4. Fetch ADM1 codes from GEE ────────────────────────────────────────
    init_ee()
    lookup = fetch_gaul_adm1_codes()
    df = join_adm1_codes(df, lookup)

    # ── 5. Select base columns (anomaly cols added in step 7) ──────────────
    base_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    df = df[base_cols].copy()

    # ── 6. Join climate anomalies ───────────────────────────────────────────
    if ANOM_FILE.exists():
        anom_adm1 = load_anom_adm1(ANOM_FILE)

        # normalise join keys on df side (lowercase, stripped)
        df["_country_key"]   = df["country"].str.strip().str.lower()
        df["_adm1_key"]      = df["adm1_name"].str.strip().str.lower()

        anom_target_cols = list(ANOM_RENAME.values())
        df = df.merge(
            anom_adm1[["country", "adm1_name", "year"] + anom_target_cols],
            left_on=["_country_key", "_adm1_key", "year"],
            right_on=["country", "adm1_name", "year"],
            how="left",
            suffixes=("", "_anom_src"),
        )
        # drop helper cols introduced by merge
        df = df.drop(columns=[c for c in df.columns
                               if c in ("_country_key", "_adm1_key",
                                        "country_anom_src", "adm1_name_anom_src")])

        filled = df["NDVI_anom"].notna().sum()
        print(f"  Anomaly join: {filled}/{len(df)} rows matched "
              f"({filled/len(df)*100:.1f}%).")
    else:
        print(f"  Warning: {ANOM_FILE} not found — anomaly columns will be absent.")
        for col in ANOM_RENAME.values():
            df[col] = float("nan")

    # ── 7. Select and reorder final columns ────────────────────────────────
    df = df[[c for c in OUTPUT_COLUMNS if c in df.columns]].copy()

    # ── 8. Write output ─────────────────────────────────────────────────────
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    matched = df["adm1_code"].notna().sum()
    print(f"Saved {OUTPUT_FILE.name}  ({len(df)} rows, {matched} with adm1_code).")


if __name__ == "__main__":
    main()
