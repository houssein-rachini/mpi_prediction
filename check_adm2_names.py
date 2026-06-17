"""
Validate ADM2_CODE → ADM2_NAME and ADM1_CODE mapping in a CSV against GAUL.

Usage:
    python check_adm2_names.py                                  # defaults to adm2_all_vars.csv
    python check_adm2_names.py --csv adm2_predictions_no_ratio.csv

Reports:
  - ADM2_CODEs not found in GAUL
  - ADM2_NAME mismatches
  - ADM1_CODE mismatches (wrong province assignment)
"""

from __future__ import annotations

import argparse
import ee
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "adm2_vars_9countries" / "adm2_all_vars.csv"

COUNTRIES = [
    "Bosnia and Herzegovina", "Egypt", "Jordan", "Kyrgyzstan",
    "Montenegro", "Morocco", "Tajikistan", "Tunisia", "Turkey",
]

PAGE_SIZE = 5000


def initialize_ee() -> None:
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def fetch_gaul() -> pd.DataFrame:
    """Fetch ADM1_CODE, ADM2_CODE, ADM2_NAME for all 9 countries from GAUL."""
    gaul2 = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
    fc = (
        gaul2
        .filter(ee.Filter.inList("ADM0_NAME", COUNTRIES))
        .select(["ADM0_NAME", "ADM1_CODE", "ADM2_CODE", "ADM2_NAME"])
    )

    total = fc.size().getInfo()
    rows = []

    with tqdm(total=total, desc="Fetching GAUL features", unit="feature") as pbar:
        offset = 0
        while offset < total:
            page = fc.toList(PAGE_SIZE, offset).getInfo()
            for f in page:
                p = f["properties"]
                rows.append({
                    "Country":        p["ADM0_NAME"],
                    "ADM1_CODE_gaul": int(p["ADM1_CODE"]),
                    "ADM2_CODE":      int(p["ADM2_CODE"]),
                    "ADM2_NAME_gaul": p["ADM2_NAME"],
                })
            pbar.update(len(page))
            offset += PAGE_SIZE

    df = pd.DataFrame(rows)
    df["ADM2_CODE"]      = pd.to_numeric(df["ADM2_CODE"],      errors="coerce").astype("Int64")
    df["ADM1_CODE_gaul"] = pd.to_numeric(df["ADM1_CODE_gaul"], errors="coerce").astype("Int64")
    print(f"  GAUL: {len(df)} districts | {df['Country'].nunique()} countries")
    return df


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV and normalise column names to Title_Case regardless of source format."""
    df = pd.read_csv(path)
    # normalise lowercase variants (adm2_predictions_no_ratio.csv style)
    rename = {
        "country":   "Country",
        "adm1_code": "ADM1_CODE",
        "adm2_code": "ADM2_CODE",
        "adm2_name": "ADM2_NAME",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["ADM2_CODE"] = pd.to_numeric(df["ADM2_CODE"], errors="coerce").astype("Int64")
    df["ADM1_CODE"] = pd.to_numeric(df["ADM1_CODE"], errors="coerce").astype("Int64")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help="CSV file to validate (default: adm2_vars_9countries/adm2_all_vars.csv)",
    )
    args = parser.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else BASE_DIR / args.csv
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    print("Initializing GEE ...")
    initialize_ee()

    print("Fetching canonical data from GAUL ...")
    gaul = fetch_gaul()

    print(f"\nLoading {csv_path.name} ...")
    df = load_csv(csv_path)

    local_pairs = df[["Country", "ADM1_CODE", "ADM2_CODE", "ADM2_NAME"]].drop_duplicates()
    print(f"  CSV: {len(local_pairs)} unique (Country, ADM2_CODE) pairs")

    merged = local_pairs.merge(gaul, on=["Country", "ADM2_CODE"], how="left")

    # ── 1. Codes not found in GAUL ────────────────────────────────────────────
    not_in_gaul = merged[merged["ADM2_NAME_gaul"].isna()]
    print(f"\n[1] ADM2_CODEs not found in GAUL: {len(not_in_gaul)}")
    if len(not_in_gaul) > 0:
        print(not_in_gaul[["Country", "ADM2_CODE", "ADM2_NAME", "ADM1_CODE"]].to_string(index=False))

    found = merged[merged["ADM2_NAME_gaul"].notna()].copy()

    # ── 2. ADM2_NAME mismatches ───────────────────────────────────────────────
    name_mismatch = found[found["ADM2_NAME"] != found["ADM2_NAME_gaul"]]
    print(f"\n[2] ADM2_NAME mismatches: {len(name_mismatch)}")
    if len(name_mismatch) > 0:
        print(
            name_mismatch[["Country", "ADM2_CODE", "ADM2_NAME", "ADM2_NAME_gaul"]]
            .to_string(index=False)
        )

    # ── 3. ADM1_CODE mismatches ───────────────────────────────────────────────
    adm1_mismatch = found[found["ADM1_CODE"] != found["ADM1_CODE_gaul"]]
    print(f"\n[3] ADM1_CODE mismatches: {len(adm1_mismatch)}")
    if len(adm1_mismatch) > 0:
        print(
            adm1_mismatch[["Country", "ADM2_CODE", "ADM2_NAME", "ADM1_CODE", "ADM1_CODE_gaul"]]
            .to_string(index=False)
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  File                          : {csv_path.name}")
    print(f"  Total unique ADM2_CODEs in CSV: {len(local_pairs)}")
    print(f"  Matched in GAUL               : {len(found)}")
    print(f"  Not found in GAUL             : {len(not_in_gaul)}")
    print(f"  ADM2_NAME mismatches          : {len(name_mismatch)}")
    print(f"  ADM1_CODE mismatches          : {len(adm1_mismatch)}")

    if len(not_in_gaul) == 0 and len(name_mismatch) == 0 and len(adm1_mismatch) == 0:
        print("\n  All codes, names and ADM1 assignments match GAUL exactly.")


if __name__ == "__main__":
    main()
