"""
Patch missing adm1_code values in mpi_training_panel.csv.

Reads the existing output, finds rows where adm1_code is empty,
fixes mojibake in adm1_name, re-looks up from FAO GAUL, writes back.
"""

from pathlib import Path
import ee
import pandas as pd

PANEL_FILE = Path(__file__).resolve().parent / "mpi_training_panel.csv"

COUNTRY_NAME_MAP = {
    "Tanzania": "United Republic of Tanzania",
    "Bolivia, Plurinational State of": "Bolivia",
    "Congo, Democratic Republic of the": "Congo",
    "Democratic Republic of Congo": "Congo",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Lao PDR": "Lao People's Democratic Republic",
    "Moldova": "Moldova, Republic of",
    "Timor Leste": "Timor-Leste",
    "Vietnam": "Viet Nam",
}


def fix_mojibake(s: str) -> str:
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def fetch_gaul_lookup() -> dict[tuple[str, str], int]:
    print("Fetching ADM1 codes from FAO GAUL …")
    fc = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")
    data = fc.select(["ADM0_NAME", "ADM1_NAME", "ADM1_CODE"]).getInfo()
    lookup = {}
    for feat in data["features"]:
        p = feat["properties"]
        key = (str(p["ADM0_NAME"]).strip().lower(), str(p["ADM1_NAME"]).strip().lower())
        lookup[key] = int(p["ADM1_CODE"])
    print(f"  → {len(lookup)} entries loaded.")
    return lookup


def main() -> None:
    df = pd.read_csv(PANEL_FILE, encoding="utf-8")
    missing_mask = df["adm1_code"].isna()
    n_missing = missing_mask.sum()

    if n_missing == 0:
        print("No missing adm1_code values — nothing to patch.")
        return

    print(f"Found {n_missing} rows with missing adm1_code. Patching …")

    ee.Initialize()
    lookup = fetch_gaul_lookup()

    def resolve(row):
        country = COUNTRY_NAME_MAP.get(row["country"], row["country"])
        adm1 = fix_mojibake(row["adm1_name"])
        code = lookup.get((country.strip().lower(), adm1.strip().lower()), pd.NA)
        return adm1, code

    fixed_names, fixed_codes = zip(*df[missing_mask].apply(resolve, axis=1))
    df.loc[missing_mask, "adm1_name"] = list(fixed_names)
    df.loc[missing_mask, "adm1_code"] = list(fixed_codes)

    still_missing = df["adm1_code"].isna().sum()
    df.to_csv(PANEL_FILE, index=False, encoding="utf-8")
    print(f"Patched {n_missing - still_missing} rows. Still unmatched: {still_missing}.")
    print(f"Saved {PANEL_FILE.name}.")


if __name__ == "__main__":
    main()
