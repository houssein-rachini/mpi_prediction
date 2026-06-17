"""
Fetch ADM1 codes for all unique (Country, Region) pairs in the training file
via FAO GAUL and save to training_adm1_codes.csv.

Output columns: country, adm1_name, adm1_code
"""

from __future__ import annotations

from pathlib import Path
import ee
import pandas as pd

BASE_DIR    = Path(__file__).resolve().parent
SOURCE      = BASE_DIR / "Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv"
OUT_FILE    = BASE_DIR / "training_adm1_codes.csv"

COUNTRY_NAME_MAP = {
    "Tanzania":                          "United Republic of Tanzania",
    "Tanzania, United Republic of":      "United Republic of Tanzania",
    "Bolivia, Plurinational State of":   "Bolivia",
    "Congo, Democratic Republic of the": "Congo",
    "Democratic Republic of Congo":      "Congo",
    "Republic of Congo":                 "Congo",
    "Congo, Republic of":                "Congo",
    "Cote d'Ivoire":                     "Côte d'Ivoire",
    "Lao":                               "Lao People's Democratic Republic",
    "Lao PDR":                           "Lao People's Democratic Republic",
    "Laos":                              "Lao People's Democratic Republic",
    "Macedonia":                         "The former Yugoslav Republic of Macedonia",
    "TFYR of Macedonia":                 "The former Yugoslav Republic of Macedonia",
    "North Macedonia":                   "The former Yugoslav Republic of Macedonia",
    "Moldova":                           "Moldova, Republic of",
    "Timor Leste":                       "Timor-Leste",
    "Vietnam":                           "Viet Nam",
}


def fix_mojibake(s: str) -> str:
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def init_ee() -> None:
    try:
        ee.Initialize()
    except Exception as e:
        raise RuntimeError("EE init failed — run `earthengine authenticate` first.") from e


def fetch_gaul_lookup() -> dict[tuple[str, str], int]:
    print("Fetching ADM1 codes from FAO GAUL ...")
    fc   = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")
    data = fc.select(["ADM0_NAME", "ADM1_NAME", "ADM1_CODE"]).getInfo()
    lookup: dict[tuple[str, str], int] = {}
    for feat in data["features"]:
        p   = feat["properties"]
        key = (str(p["ADM0_NAME"]).strip().lower(), str(p["ADM1_NAME"]).strip().lower())
        lookup[key] = int(p["ADM1_CODE"])
    print(f"  -> {len(lookup)} ADM1 entries loaded.")
    return lookup


def main():
    print(f"Loading {SOURCE.name} ...")
    df = pd.read_csv(SOURCE, encoding="utf-8", usecols=["Country", "Region"])
    df["Country"] = df["Country"].map(fix_mojibake).str.strip()
    df["Region"]  = df["Region"].map(fix_mojibake).str.strip()

    pairs = df.drop_duplicates().reset_index(drop=True)
    pairs.columns = ["country", "adm1_name"]
    print(f"  -> {len(pairs)} unique (country, region) pairs across "
          f"{pairs['country'].nunique()} countries.")

    print("Initialising Earth Engine ...")
    init_ee()
    print("  -> EE ready.")

    lookup = fetch_gaul_lookup()

    def get_code(row):
        norm_country = COUNTRY_NAME_MAP.get(row["country"], row["country"])
        key = (norm_country.strip().lower(), row["adm1_name"].strip().lower())
        return lookup.get(key, pd.NA)

    pairs["adm1_code"] = pairs.apply(get_code, axis=1)

    matched   = pairs["adm1_code"].notna().sum()
    unmatched = pairs[pairs["adm1_code"].isna()]
    print(f"\nMatched: {matched}/{len(pairs)} ({matched/len(pairs)*100:.1f}%)")
    if not unmatched.empty:
        print(f"Unmatched ({len(unmatched)}):")
        print(unmatched.to_string(index=False))

    pairs.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_FILE.name} ({len(pairs)} rows).")


if __name__ == "__main__":
    main()
