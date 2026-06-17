"""
Build training_countries_by_region.csv

Columns: country, n_obs, n_adm1_units, years_present,
         un_region, un_subregion, wb_region, wb_income_group

Source: Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv (82 training countries)
Metadata: country_converter (UN M49 regions + World Bank classifications)
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import country_converter as coco

BASE_DIR = Path(__file__).resolve().parent
SOURCE   = BASE_DIR / "Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv"
OUT_FILE = BASE_DIR / "training_countries_by_region.xlsx"

# Name fixes so country_converter resolves them correctly
NAME_FIX = {
    "Congo":                              "Democratic Republic of the Congo",
    "Lao People's Democratic Republic":   "Laos",
    "Moldova, Republic of":               "Moldova",
    "The former Yugoslav Republic of Macedonia": "North Macedonia",
    "United Republic of Tanzania":        "Tanzania",
    "Timor-Leste":                        "Timor-Leste",
    "Viet Nam":                           "Vietnam",
    "Syrian Arab Republic":               "Syria",
    "Bolivia":                            "Bolivia",
    "Côte d'Ivoire":                      "Ivory Coast",
}


def main():
    print(f"Loading {SOURCE.name} ...")
    df = pd.read_csv(SOURCE, encoding="utf-8", usecols=["Country", "Region", "Year"])
    df["Country"] = df["Country"].str.strip()
    df["Region"]  = df["Region"].str.strip()

    # ── per-country stats ─────────────────────────────────────────────────────
    stats = (
        df.groupby("Country")
        .agg(
            n_obs        = ("Country", "count"),
            n_adm1_units = ("Region",  "nunique"),
            years_present= ("Year",    lambda x: ",".join(str(int(y)) for y in sorted(x.unique()))),
        )
        .reset_index()
        .rename(columns={"Country": "country"})
    )
    print(f"  -> {len(stats)} countries")

    # ── country_converter metadata ────────────────────────────────────────────
    cc = coco.CountryConverter()

    lookup_names = [NAME_FIX.get(c, c) for c in stats["country"]]

    stats["un_region"]  = cc.convert(lookup_names, to="UNregion",  not_found="Unknown")
    stats["continent"]  = cc.convert(lookup_names, to="continent", not_found="Unknown")

    # WB income group — hardcoded for the 82 training countries (not in this coco version)
    WB_INCOME = {
        "Afghanistan": "Low income", "Albania": "Upper middle income",
        "Angola": "Lower middle income", "Azerbaijan": "Upper middle income",
        "Bangladesh": "Lower middle income", "Belize": "Upper middle income",
        "Benin": "Low income", "Bhutan": "Lower middle income",
        "Bolivia": "Lower middle income", "Botswana": "Upper middle income",
        "Brazil": "Upper middle income", "Burkina Faso": "Low income",
        "Burundi": "Low income", "Cambodia": "Lower middle income",
        "Cameroon": "Lower middle income", "Central African Republic": "Low income",
        "Chad": "Low income", "Congo": "Lower middle income",
        "Costa Rica": "Upper middle income", "Cuba": "Upper middle income",
        "Djibouti": "Lower middle income", "Dominican Republic": "Upper middle income",
        "Ecuador": "Upper middle income", "Egypt": "Lower middle income",
        "El Salvador": "Lower middle income", "Ethiopia": "Low income",
        "Fiji": "Upper middle income", "Gabon": "Upper middle income",
        "Ghana": "Lower middle income", "Guatemala": "Upper middle income",
        "Guinea": "Low income", "Guinea-Bissau": "Low income",
        "Haiti": "Low income", "Honduras": "Lower middle income",
        "India": "Lower middle income", "Indonesia": "Upper middle income",
        "Iraq": "Upper middle income", "Jamaica": "Upper middle income",
        "Jordan": "Upper middle income", "Kenya": "Lower middle income",
        "Kyrgyzstan": "Lower middle income", "Laos": "Lower middle income",
        "Lesotho": "Lower middle income", "Liberia": "Low income",
        "Madagascar": "Low income", "Mali": "Low income",
        "Mauritania": "Lower middle income", "Mexico": "Upper middle income",
        "Moldova, Republic of": "Lower middle income", "Mongolia": "Lower middle income",
        "Morocco": "Lower middle income", "Mozambique": "Low income",
        "Myanmar": "Lower middle income", "Namibia": "Upper middle income",
        "Niger": "Low income", "Nigeria": "Lower middle income",
        "North Macedonia": "Upper middle income", "Pakistan": "Lower middle income",
        "Papua New Guinea": "Lower middle income", "Paraguay": "Upper middle income",
        "Peru": "Upper middle income", "Senegal": "Lower middle income",
        "Sierra Leone": "Low income", "Sri Lanka": "Lower middle income",
        "Sudan": "Low income", "Suriname": "Upper middle income",
        "Swaziland": "Lower middle income", "Syrian Arab Republic": "Low income",
        "Tajikistan": "Low income", "Tanzania": "Lower middle income",
        "Thailand": "Upper middle income", "Timor-Leste": "Lower middle income",
        "Togo": "Low income", "Trinidad and Tobago": "High income",
        "Tunisia": "Lower middle income", "Turkmenistan": "Upper middle income",
        "Uganda": "Low income", "Uzbekistan": "Lower middle income",
        "Viet Nam": "Lower middle income", "Yemen": "Low income",
        "Zambia": "Lower middle income", "Zimbabwe": "Lower middle income",
        "Lao People's Democratic Republic": "Lower middle income",
        "Côte d'Ivoire": "Lower middle income",
        "South Sudan": "Low income",
    }
    stats["wb_income_group"] = stats["country"].map(WB_INCOME).fillna("Unknown")

    # ── report unknowns ───────────────────────────────────────────────────────
    unknown = stats[stats["un_region"] == "Unknown"]["country"].tolist()
    if unknown:
        print(f"\n  ⚠  {len(unknown)} countries not resolved by country_converter:")
        for c in unknown:
            print(f"     {c}")

    # ── reorder & save ────────────────────────────────────────────────────────
    col_order = [
        "country", "n_obs", "n_adm1_units", "years_present",
        "un_region", "continent", "wb_income_group",
    ]
    stats = stats[col_order].sort_values("country").reset_index(drop=True)
    stats.to_excel(OUT_FILE, index=False)
    print(f"\nSaved {OUT_FILE.name} ({len(stats)} rows).")

    print(stats[["country", "un_region", "continent", "wb_income_group"]].to_string(index=False))


if __name__ == "__main__":
    main()
