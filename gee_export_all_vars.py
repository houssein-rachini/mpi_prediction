"""
Earth Engine Python API export script for all 82 GAUL reference countries.

This mirrors the Code Editor flow:
- 5 Drive export tasks total (population, gpp, lst, ntl, ndvi)
- not one task per country
- population years: 2012-2020
- other metrics years: 2012-2023

Performance note:
- Core computation is still server-side (`reduceRegions` on Earth Engine).
- Task granularity remains identical to the JS version (5 large exports).
"""

from __future__ import annotations

import argparse
import ee


GAUL_COUNTRIES = [
    "Afghanistan",
    "Albania",
    "Angola",
    "Azerbaijan",
    "Bangladesh",
    "Belize",
    "Benin",
    "Bhutan",
    "Bolivia",
    "Botswana",
    "Brazil",
    "Burkina Faso",
    "Burundi",
    "Cambodia",
    "Cameroon",
    "Central African Republic",
    "Chad",
    "Congo",
    "Costa Rica",
    "Cuba",
    "Djibouti",
    "Dominican Republic",
    "Ecuador",
    "Egypt",
    "El Salvador",
    "Ethiopia",
    "Fiji",
    "Gabon",
    "Ghana",
    "Guatemala",
    "Guinea",
    "Guinea-Bissau",
    "Haiti",
    "Honduras",
    "India",
    "Indonesia",
    "Iraq",
    "Jamaica",
    "Jordan",
    "Kenya",
    "Kyrgyzstan",
    "Lao People's Democratic Republic",
    "Lesotho",
    "Liberia",
    "Madagascar",
    "Mali",
    "Mauritania",
    "Mexico",
    "Moldova, Republic of",
    "Mongolia",
    "Morocco",
    "Mozambique",
    "Myanmar",
    "Namibia",
    "Nepal",
    "Nicaragua",
    "Niger",
    "Nigeria",
    "Pakistan",
    "Papua New Guinea",
    "Paraguay",
    "Peru",
    "Senegal",
    "Sierra Leone",
    "South Sudan",
    "Sri Lanka",
    "Sudan",
    "Suriname",
    "Swaziland",
    "Syrian Arab Republic",
    "Tajikistan",
    "Thailand",
    "Timor-Leste",
    "Togo",
    "Trinidad and Tobago",
    "Tunisia",
    "Turkmenistan",
    "Uganda",
    "Uzbekistan",
    "Yemen",
    "Zambia",
    "Zimbabwe",
]

YEARS_POPULATION = list(range(2012, 2021))
YEARS_OTHER = list(range(2012, 2024))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all-country GAUL level-1 metric tables to Google Drive."
    )
    parser.add_argument("--prefix", default="0m", help="File prefix tag (e.g., 0m, 250m, 1000m).")
    parser.add_argument(
        "--buffer-radius",
        type=int,
        default=0,
        help="Building-mask dilation radius in meters.",
    )
    parser.add_argument(
        "--drive-folder",
        default=None,
        help="Drive folder for exports. Default: gaul82_vars_{prefix}",
    )
    parser.add_argument(
        "--start-tasks",
        action="store_true",
        help="Start the 5 export tasks immediately (otherwise create only).",
    )
    return parser.parse_args()


def initialize_ee() -> None:
    """Initialize Earth Engine, authenticating interactively if needed."""
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def stats_reducer() -> ee.Reducer:
    """Reducer bundle used by all metrics."""
    return (
        ee.Reducer.mean()
        .combine(ee.Reducer.minMax(), "", True)
        .combine(ee.Reducer.median(), "", True)
        .combine(ee.Reducer.stdDev(), "", True)
        .combine(ee.Reducer.sum(), "", True)
    )


def country_level1(gaul: ee.FeatureCollection, country_name: str) -> ee.FeatureCollection:
    """Return GAUL ADM1 regions for a single country."""
    return gaul.filter(ee.Filter.eq("ADM0_NAME", country_name)).select(["ADM1_NAME"])


def choose_value(primary_value: ee.ComputedObject, fallback_value: ee.ComputedObject) -> ee.ComputedObject:
    """Return fallback when primary is null."""
    return ee.Algorithms.If(ee.Algorithms.IsEqual(primary_value, None), fallback_value, primary_value)


def empty_masked_image(band_name: str) -> ee.Image:
    """Create an empty fully-masked image with the requested band name."""
    return ee.Image.constant(0).rename(band_name).updateMask(ee.Image.constant(0))


def build_population_year(
    country_name: str,
    year: int,
    worldpop: ee.ImageCollection,
    gaul: ee.FeatureCollection,
    building_mask: ee.Image,
) -> ee.FeatureCollection:
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    image = worldpop.filterDate(start, end).mean().updateMask(building_mask).rename("value")
    reduced = image.reduceRegions(
        collection=country_level1(gaul, country_name),
        reducer=stats_reducer(),
        scale=100,
        tileScale=4,
    )
    return reduced.map(
        lambda f: ee.Feature(
            None,
            {
                "Country": country_name,
                "Region": f.get("ADM1_NAME"),
                "Year": year,
                "Mean Population": f.get("mean"),
                "Total Population": f.get("sum"),
                "Min Population": f.get("min"),
                "Max Population": f.get("max"),
                "Median Population": f.get("median"),
                "Std Dev Population": f.get("stdDev"),
            },
        )
    )


def build_gpp_year(
    country_name: str,
    year: int,
    modis_gpp: ee.ImageCollection,
    gaul: ee.FeatureCollection,
    building_mask: ee.Image,
) -> ee.FeatureCollection:
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    image = modis_gpp.filterDate(start, end).mean().updateMask(building_mask).rename("value")
    reduced = image.reduceRegions(
        collection=country_level1(gaul, country_name),
        reducer=stats_reducer(),
        scale=500,
        tileScale=4,
    )

    def _map(feature: ee.Feature) -> ee.Feature:
        total = feature.get("sum")
        mean = ee.Algorithms.If(
            ee.Algorithms.IsEqual(total, None),
            None,
            ee.Number(total).divide(feature.geometry().area(1)),
        )
        return ee.Feature(
            None,
            {
                "Country": country_name,
                "Region": feature.get("ADM1_NAME"),
                "Year": year,
                "Mean GPP": mean,
                "Min GPP": feature.get("min"),
                "Max GPP": feature.get("max"),
                "Median GPP": feature.get("median"),
                "Std Dev GPP": feature.get("stdDev"),
                "Total GPP": total,
            },
        )

    return reduced.map(_map)


def build_lst_year(
    country_name: str,
    year: int,
    viirs_lst: ee.ImageCollection,
    modis_lst: ee.ImageCollection,
    gaul: ee.FeatureCollection,
    building_mask: ee.Image,
) -> ee.FeatureCollection:
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    regions = country_level1(gaul, country_name)
    viirs_year = viirs_lst.filterDate(start, end)
    modis_year = modis_lst.filterDate(start, end)

    viirs_image = (
        ee.Image(
            ee.Algorithms.If(
                viirs_year.size().gt(0),
                viirs_year.mean(),
                empty_masked_image("LST_1KM"),
            )
        )
        .updateMask(building_mask)
        .rename("value")
    )
    modis_image = (
        ee.Image(
            ee.Algorithms.If(
                modis_year.size().gt(0),
                modis_year.mean().multiply(0.02),
                empty_masked_image("LST_Night_1km"),
            )
        )
        .updateMask(building_mask)
        .rename("value")
    )

    viirs_reduced = viirs_image.reduceRegions(
        collection=regions, reducer=stats_reducer(), scale=1000, tileScale=4
    )
    modis_reduced = modis_image.reduceRegions(
        collection=regions, reducer=stats_reducer(), scale=1000, tileScale=4
    )

    joined = ee.Join.saveFirst("modis").apply(
        primary=viirs_reduced,
        secondary=modis_reduced,
        condition=ee.Filter.equals(leftField="ADM1_NAME", rightField="ADM1_NAME"),
    )

    def _map(feature: ee.Feature) -> ee.Feature:
        modis_feature = ee.Feature(feature.get("modis"))
        return ee.Feature(
            None,
            {
                "Country": country_name,
                "Region": feature.get("ADM1_NAME"),
                "Year": year,
                "Mean LST (K)": choose_value(feature.get("mean"), modis_feature.get("mean")),
                "Min LST (K)": choose_value(feature.get("min"), modis_feature.get("min")),
                "Max LST (K)": choose_value(feature.get("max"), modis_feature.get("max")),
                "Median LST (K)": choose_value(feature.get("median"), modis_feature.get("median")),
                "Std Dev LST": choose_value(feature.get("stdDev"), modis_feature.get("stdDev")),
                "Total LST": choose_value(feature.get("sum"), modis_feature.get("sum")),
            },
        )

    return ee.FeatureCollection(joined).map(_map)


def build_ntl_year(
    country_name: str,
    year: int,
    viirs_ntl: ee.ImageCollection,
    gaul: ee.FeatureCollection,
    building_mask: ee.Image,
) -> ee.FeatureCollection:
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    image = viirs_ntl.filterDate(start, end).mean().updateMask(building_mask).rename("value")
    reduced = image.reduceRegions(
        collection=country_level1(gaul, country_name),
        reducer=stats_reducer(),
        scale=500,
        tileScale=4,
    )
    return reduced.map(
        lambda f: ee.Feature(
            None,
            {
                "Country": country_name,
                "Region": f.get("ADM1_NAME"),
                "Year": year,
                "Mean NTL": f.get("mean"),
                "Min NTL": f.get("min"),
                "Max NTL": f.get("max"),
                "Median NTL": f.get("median"),
                "Std Dev NTL": f.get("stdDev"),
                "Total NTL": f.get("sum"),
            },
        )
    )


def build_ndvi_year(
    country_name: str,
    year: int,
    ndvi_collection: ee.ImageCollection,
    gaul: ee.FeatureCollection,
    building_mask: ee.Image,
) -> ee.FeatureCollection:
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year, 12, 31)
    image = ndvi_collection.filterDate(start, end).mean().updateMask(building_mask).rename("value")
    reduced = image.reduceRegions(
        collection=country_level1(gaul, country_name),
        reducer=stats_reducer(),
        scale=500,
        tileScale=4,
    )
    return reduced.map(
        lambda f: ee.Feature(
            None,
            {
                "Country": country_name,
                "Region": f.get("ADM1_NAME"),
                "Year": year,
                "Mean NDVI": f.get("mean"),
                "Min NDVI": f.get("min"),
                "Max NDVI": f.get("max"),
                "Median NDVI": f.get("median"),
                "Std Dev NDVI": f.get("stdDev"),
                "Total NDVI": f.get("sum"),
            },
        )
    )


def build_metric_collection(
    metric_name: str,
    gaul: ee.FeatureCollection,
    building_mask: ee.Image,
    worldpop: ee.ImageCollection,
    modis_gpp: ee.ImageCollection,
    viirs_lst: ee.ImageCollection,
    modis_lst: ee.ImageCollection,
    viirs_ntl: ee.ImageCollection,
    ndvi_collection: ee.ImageCollection,
) -> ee.FeatureCollection:
    """Build one flattened feature collection for a metric across all countries/years."""
    years = YEARS_POPULATION if metric_name == "population" else YEARS_OTHER
    collections = []
    for country_name in GAUL_COUNTRIES:
        for year in years:
            if metric_name == "population":
                collections.append(build_population_year(country_name, year, worldpop, gaul, building_mask))
            elif metric_name == "gpp":
                collections.append(build_gpp_year(country_name, year, modis_gpp, gaul, building_mask))
            elif metric_name == "lst":
                collections.append(
                    build_lst_year(country_name, year, viirs_lst, modis_lst, gaul, building_mask)
                )
            elif metric_name == "ntl":
                collections.append(build_ntl_year(country_name, year, viirs_ntl, gaul, building_mask))
            elif metric_name == "ndvi":
                collections.append(build_ndvi_year(country_name, year, ndvi_collection, gaul, building_mask))
            else:
                raise ValueError(f"Unsupported metric: {metric_name}")
    return ee.FeatureCollection(collections).flatten()


def create_export_task(
    collection: ee.FeatureCollection,
    description: str,
    folder: str,
    file_name_prefix: str,
) -> ee.batch.Task:
    """Create a Drive table export task."""
    return ee.batch.Export.table.toDrive(
        collection=collection,
        description=description,
        folder=folder,
        fileNamePrefix=file_name_prefix,
        fileFormat="CSV",
    )


def main() -> None:
    args = parse_args()
    drive_folder = args.drive_folder or f"gaul82_vars_{args.prefix}"
    initialize_ee()

    building_mask = (
        ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018")
        .select("built_height")
        .gte(2.5)
        .focalMax(kernel=ee.Kernel.circle(radius=args.buffer_radius, units="meters"))
        .selfMask()
    )
    gaul = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")
    worldpop = ee.ImageCollection("WorldPop/GP/100m/pop").select("population")
    modis_gpp = ee.ImageCollection("MODIS/061/MOD17A3HGF").select("Gpp")
    viirs_lst = ee.ImageCollection("NASA/VIIRS/002/VNP21A1N").select("LST_1KM")
    modis_lst = ee.ImageCollection("MODIS/006/MOD11A2").select("LST_Night_1km")
    viirs_ntl = ee.ImageCollection("NOAA/VIIRS/001/VNP46A2").select(
        "Gap_Filled_DNB_BRDF_Corrected_NTL"
    )
    modis_surface = ee.ImageCollection("MODIS/061/MOD09A1")
    ndvi_collection = modis_surface.map(
        lambda image: image.normalizedDifference(["sur_refl_b02", "sur_refl_b01"])
        .rename("value")
        .copyProperties(image, image.propertyNames())
    )

    population_collection = build_metric_collection(
        "population",
        gaul,
        building_mask,
        worldpop,
        modis_gpp,
        viirs_lst,
        modis_lst,
        viirs_ntl,
        ndvi_collection,
    )
    gpp_collection = build_metric_collection(
        "gpp",
        gaul,
        building_mask,
        worldpop,
        modis_gpp,
        viirs_lst,
        modis_lst,
        viirs_ntl,
        ndvi_collection,
    )
    lst_collection = build_metric_collection(
        "lst",
        gaul,
        building_mask,
        worldpop,
        modis_gpp,
        viirs_lst,
        modis_lst,
        viirs_ntl,
        ndvi_collection,
    )
    ntl_collection = build_metric_collection(
        "ntl",
        gaul,
        building_mask,
        worldpop,
        modis_gpp,
        viirs_lst,
        modis_lst,
        viirs_ntl,
        ndvi_collection,
    )
    ndvi_collection_export = build_metric_collection(
        "ndvi",
        gaul,
        building_mask,
        worldpop,
        modis_gpp,
        viirs_lst,
        modis_lst,
        viirs_ntl,
        ndvi_collection,
    )

    prefix = args.prefix
    tasks = [
        create_export_task(
            population_collection,
            f"all_82_population_{prefix}_original_ref_gaul_actual",
            drive_folder,
            f"all_82_population_{prefix}_original_ref_gaul_actual",
        ),
        create_export_task(
            gpp_collection,
            f"all_82_gpp_{prefix}_original_ref_gaul",
            drive_folder,
            f"all_82_gpp_{prefix}_original_ref_gaul",
        ),
        create_export_task(
            lst_collection,
            f"all_82_lst_{prefix}_original_ref_gaul",
            drive_folder,
            f"all_82_lst_{prefix}_original_ref_gaul",
        ),
        create_export_task(
            ntl_collection,
            f"all_82_ntl_{prefix}_original_ref_gaul",
            drive_folder,
            f"all_82_ntl_{prefix}_original_ref_gaul",
        ),
        create_export_task(
            ndvi_collection_export,
            f"all_82_ndvi_{prefix}_original_ref_gaul",
            drive_folder,
            f"all_82_ndvi_{prefix}_original_ref_gaul",
        ),
    ]

    print({"countries": len(GAUL_COUNTRIES), "tasks_created": len(tasks), "drive_folder": drive_folder})
    for task in tasks:
        print({"description": task.config.get("description"), "state": task.status().get("state", "UNSUBMITTED")})

    if args.start_tasks:
        for task in tasks:
            task.start()
        print("Started all 5 export tasks.")
    else:
        print("Tasks created but not started. Re-run with --start-tasks to launch from Python.")


if __name__ == "__main__":
    main()
