"""
gee_export_adm2_population_2012_2014.py

Exports ADM2 WorldPop stats (Median_Pop, StdDev_Pop, Total_Pop) under the
GHSL 500m building mask for years 2012-2014 — the gap not covered by the
existing adm2_population.csv (which starts at 2015).

Once downloaded from Drive, merge with adm2_population.csv and re-run
gee_export_adm2_population_v2.py to get consistent 2012-2020 base for
extrapolation (matching the Streamlit app and build_adm2_predictions_9countries.py).

Output Drive folder: gaul9_adm2_vars_500m
  adm2_population_2012_2014.csv

Run:
    python gee_export_adm2_population_2012_2014.py --start-tasks
"""

from __future__ import annotations

import argparse
import ee

COUNTRIES = [
    "Bosnia and Herzegovina", "Egypt", "Jordan", "Kyrgyzstan",
    "Montenegro", "Morocco", "Tajikistan", "Tunisia", "Turkey",
]
YEARS        = [2012, 2013, 2014]
DRIVE_FOLDER = "gaul9_adm2_vars_500m"
HEIGHT_THRESH = 2.5
BUFFER_M      = 500


def initialize_ee() -> None:
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def building_mask() -> ee.Image:
    return (
        ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018")
        .select("built_height")
        .gte(HEIGHT_THRESH)
        .focal_max(kernel=ee.Kernel.circle(radius=BUFFER_M, units="meters"))
    )


def regions_fc() -> ee.FeatureCollection:
    return (
        ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
        .filter(ee.Filter.inList("ADM0_NAME", COUNTRIES))
        .select(["ADM0_NAME", "ADM1_CODE", "ADM2_CODE", "ADM2_NAME"])
    )


def pop_stats_for_year(year: int, mask: ee.Image, regions: ee.FeatureCollection) -> ee.FeatureCollection:
    worldpop = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .select("population")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .mean()
        .updateMask(mask)
    )

    reduced = worldpop.reduceRegions(
        collection=regions,
        reducer=ee.Reducer.median()
            .combine(ee.Reducer.stdDev(), None, True)
            .combine(ee.Reducer.sum(),    None, True),
        scale=100,
        tileScale=4,
    )

    def _add_fields(f: ee.Feature) -> ee.Feature:
        return ee.Feature(None, {
            "Country":    f.get("ADM0_NAME"),
            "ADM1_CODE":  f.get("ADM1_CODE"),
            "ADM2_CODE":  f.get("ADM2_CODE"),
            "ADM2_NAME":  f.get("ADM2_NAME"),
            "Year":       year,
            "Median_Pop": f.get("median"),
            "StdDev_Pop": f.get("stdDev"),
            "Total_Pop":  f.get("sum"),
        })

    return reduced.map(_add_fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-tasks", action="store_true")
    parser.add_argument("--drive-folder", default=DRIVE_FOLDER)
    args = parser.parse_args()

    print("Initialising GEE ...")
    initialize_ee()

    mask    = building_mask()
    regions = regions_fc()
    print(f"Countries: {len(COUNTRIES)} | Years: {YEARS}")

    # Combine all years into one collection
    all_features = ee.FeatureCollection([])
    for year in YEARS:
        fc = pop_stats_for_year(year, mask, regions)
        all_features = all_features.merge(fc)
        print(f"  Queued year {year}")

    task = ee.batch.Export.table.toDrive(
        collection=all_features,
        description="adm2_population_2012_2014",
        folder=args.drive_folder,
        fileNamePrefix="adm2_population_2012_2014",
        fileFormat="CSV",
    )

    print(f"\nTask: adm2_population_2012_2014")
    print(f"Drive folder: {args.drive_folder}")

    if args.start_tasks:
        task.start()
        print("Task started — monitor at code.earthengine.google.com/tasks")
    else:
        print("Re-run with --start-tasks to launch.")


if __name__ == "__main__":
    main()
