"""
gee_export_pdsi_precipitation.py

Exports annual PDSI and CHIRPS precipitation for all 82 training countries
at ADM1 level with GHSL 500m building mask.

Collections:
  PDSI          : IDAHO_EPSCOR/TERRACLIMATE  (band: pdsi, scale 0.01, monthly → annual mean)
  Precipitation : UCSB-CHG/CHIRPS/DAILY      (band: precipitation, annual sum)

Output: 2 Drive CSVs in folder gaul82_pdsi_precip_500m
  all_82_pdsi_500m.csv          (Country, Region, Year, Mean_PDSI, StdDev_PDSI)
  all_82_precipitation_500m.csv (Country, Region, Year, Sum_Precip, Mean_Precip, StdDev_Precip)

Run:
    python gee_export_pdsi_precipitation.py --start-tasks
"""

from __future__ import annotations

import argparse
import ee

GAUL_COUNTRIES = [
    "Afghanistan", "Albania", "Angola", "Azerbaijan", "Bangladesh", "Belize",
    "Benin", "Bhutan", "Bolivia", "Botswana", "Brazil", "Burkina Faso", "Burundi",
    "Cambodia", "Cameroon", "Central African Republic", "Chad", "Congo",
    "Costa Rica", "Cuba", "Djibouti", "Dominican Republic", "Ecuador", "Egypt",
    "El Salvador", "Ethiopia", "Fiji", "Gabon", "Ghana", "Guatemala", "Guinea",
    "Guinea-Bissau", "Haiti", "Honduras", "India", "Indonesia", "Iraq", "Jamaica",
    "Jordan", "Kenya", "Kyrgyzstan", "Lao People's Democratic Republic",
    "Lesotho", "Liberia", "Madagascar", "Mali", "Mauritania", "Mexico",
    "Moldova, Republic of", "Mongolia", "Morocco", "Mozambique", "Myanmar",
    "Namibia", "Nepal", "Nicaragua", "Niger", "Nigeria", "Pakistan",
    "Papua New Guinea", "Paraguay", "Peru", "Senegal", "Sierra Leone",
    "South Sudan", "Sri Lanka", "Sudan", "Suriname", "Swaziland",
    "The former Yugoslav Republic of Macedonia",
    "Syrian Arab Republic", "Tajikistan", "Thailand", "Timor-Leste", "Togo",
    "Trinidad and Tobago", "Tunisia", "Turkmenistan", "Uganda",
    "United Republic of Tanzania", "Uzbekistan",
    "Yemen", "Zambia", "Zimbabwe",
]

YEARS = list(range(2012, 2024))
BUFFER_RADIUS = 500
DRIVE_FOLDER  = "gaul82_pdsi_precip_500m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-tasks", action="store_true")
    parser.add_argument("--drive-folder", default=DRIVE_FOLDER)
    return parser.parse_args()


def initialize_ee() -> None:
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def build_building_mask() -> ee.Image:
    return (
        ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018")
        .select("built_height")
        .gte(2.5)
        .focalMax(kernel=ee.Kernel.circle(radius=BUFFER_RADIUS, units="meters"))
        .selfMask()
    )


def country_adm1(gaul: ee.FeatureCollection, country: str) -> ee.FeatureCollection:
    return gaul.filter(ee.Filter.eq("ADM0_NAME", country)).select(["ADM1_NAME"])


# ── Per-year builders ──────────────────────────────────────────────────────────

def build_pdsi_year(
    country: str, year: int,
    pdsi_ic: ee.ImageCollection,
    gaul: ee.FeatureCollection,
    mask: ee.Image,
) -> ee.FeatureCollection:
    start = ee.Date.fromYMD(year, 1, 1)
    end   = ee.Date.fromYMD(year + 1, 1, 1)
    img   = (
        pdsi_ic.filterDate(start, end)
        .mean()
        .multiply(0.01)
        .updateMask(mask)
        .rename("value")
    )
    reduced = img.reduceRegions(
        collection=country_adm1(gaul, country),
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), "", True),
        scale=5000,
        tileScale=4,
    )
    return reduced.map(lambda f: ee.Feature(None, {
        "Country":     country,
        "Region":      f.get("ADM1_NAME"),
        "Year":        year,
        "Mean_PDSI":   f.get("mean"),
        "StdDev_PDSI": f.get("stdDev"),
    }))


def build_precip_year(
    country: str, year: int,
    chirps: ee.ImageCollection,
    gaul: ee.FeatureCollection,
    mask: ee.Image,
) -> ee.FeatureCollection:
    start = ee.Date.fromYMD(year, 1, 1)
    end   = ee.Date.fromYMD(year + 1, 1, 1)
    img   = (
        chirps.filterDate(start, end)
        .sum()
        .updateMask(mask)
        .rename("value")
    )
    reduced = img.reduceRegions(
        collection=country_adm1(gaul, country),
        reducer=(
            ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), "", True)
            .combine(ee.Reducer.sum(), "", True)
        ),
        scale=5000,
        tileScale=4,
    )
    return reduced.map(lambda f: ee.Feature(None, {
        "Country":       country,
        "Region":        f.get("ADM1_NAME"),
        "Year":          year,
        "Sum_Precip":    f.get("sum"),
        "Mean_Precip":   f.get("mean"),
        "StdDev_Precip": f.get("stdDev"),
    }))


# ── Collection builders ────────────────────────────────────────────────────────

def build_all_pdsi(
    pdsi_ic: ee.ImageCollection,
    gaul: ee.FeatureCollection,
    mask: ee.Image,
) -> ee.FeatureCollection:
    parts = []
    for country in GAUL_COUNTRIES:
        for year in YEARS:
            parts.append(build_pdsi_year(country, year, pdsi_ic, gaul, mask))
    return ee.FeatureCollection(parts).flatten()


def build_all_precip(
    chirps: ee.ImageCollection,
    gaul: ee.FeatureCollection,
    mask: ee.Image,
) -> ee.FeatureCollection:
    parts = []
    for country in GAUL_COUNTRIES:
        for year in YEARS:
            parts.append(build_precip_year(country, year, chirps, gaul, mask))
    return ee.FeatureCollection(parts).flatten()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    initialize_ee()

    mask  = build_building_mask()
    gaul  = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")
    pdsi_ic = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").select("pdsi")
    chirps  = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")

    tasks = [
        ee.batch.Export.table.toDrive(
            collection=build_all_pdsi(pdsi_ic, gaul, mask),
            description="all_82_pdsi_500m",
            folder=args.drive_folder,
            fileNamePrefix="all_82_pdsi_500m",
            fileFormat="CSV",
        ),
        ee.batch.Export.table.toDrive(
            collection=build_all_precip(chirps, gaul, mask),
            description="all_82_precipitation_500m",
            folder=args.drive_folder,
            fileNamePrefix="all_82_precipitation_500m",
            fileFormat="CSV",
        ),
    ]

    for task in tasks:
        if args.start_tasks:
            task.start()
            print(f"Started: {task.config['description']}")
        else:
            print(f"Created (not started): {task.config['description']}")

    if not args.start_tasks:
        print("\nRun with --start-tasks to submit to GEE.")


if __name__ == "__main__":
    main()
