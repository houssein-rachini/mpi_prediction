"""
Server-side GEE batch export: ADM2 features for 9 countries, 2015-2024.

Replaces per-district client-side fetching with reduceRegions() batch tasks.
One Drive export per metric — all 9 countries x 10 years in a single CSV.

Countries: Bosnia and Herzegovina, Egypt, Jordan, Kyrgyzstan, Montenegro,
           Morocco, Tajikistan, Tunisia, Turkey
Years    : 2015-2024
Level    : FAO GAUL level 2
Mask     : GHSL built height >= 2.5m, dilated 500m

5 export tasks -> Drive folder gaul9_adm2_vars_500m:
  adm2_population.csv   (Median_Pop, StdDev_Pop, Total_Pop)
  adm2_gpp.csv          (Mean_GPP = sum/area_m², StdDev_GPP)
  adm2_lst.csv          (Mean_LST, StdDev_LST)
  adm2_ntl.csv          (Mean_NTL, StdDev_NTL, Sum_NTL)
  adm2_ndvi.csv         (Median_NDVI, StdDev_NDVI)

Post-2020 population: pixel-level linearFit extrapolation over WorldPop
2000-2020, computed once and reused.

Run:
    python gee_export_adm2_9countries.py --start-tasks
"""

from __future__ import annotations

import argparse
import ee

COUNTRIES = [
    "Bosnia and Herzegovina", "Egypt", "Jordan", "Kyrgyzstan",
    "Montenegro", "Morocco", "Tajikistan", "Tunisia", "Turkey",
]
YEARS = list(range(2015, 2025))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ADM2 feature tables for 9 countries to Google Drive."
    )
    parser.add_argument(
        "--drive-folder", default="gaul9_adm2_vars_500m",
        help="Google Drive folder for exports.",
    )
    parser.add_argument(
        "--start-tasks", action="store_true", help="Start export tasks immediately."
    )
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
        .focalMax(kernel=ee.Kernel.circle(radius=500, units="meters"))
        .selfMask()
    )


def adm2_regions(gaul2: ee.FeatureCollection, country: str) -> ee.FeatureCollection:
    return gaul2.filter(ee.Filter.eq("ADM0_NAME", country)).select(
        ["ADM0_NAME", "ADM1_CODE", "ADM2_CODE", "ADM2_NAME"]
    )


def build_pop_fit(worldpop: ee.ImageCollection) -> ee.Image:
    """Pixel-wise linearFit over WorldPop 2000-2020. Returns scale+offset bands."""
    def _add_year_band(img):
        y = ee.Date(img.get("system:time_start")).get("year").toFloat()
        return img.updateMask(img.mask()).addBands(
            ee.Image.constant(y).rename("year")
        ).toFloat()

    return (
        worldpop.map(_add_year_band)
        .select(["year", "population"])
        .reduce(ee.Reducer.linearFit())
    )


def _get_pop_image(
    worldpop: ee.ImageCollection, year: int, pop_fit: ee.Image
) -> ee.Image:
    if year <= 2020:
        return worldpop.filterDate(
            ee.Date.fromYMD(year, 1, 1),
            ee.Date.fromYMD(year + 1, 1, 1),
        ).mean()
    slope     = pop_fit.select("scale")
    intercept = pop_fit.select("offset")
    return intercept.add(slope.multiply(year)).max(ee.Image(0))


def _id_props(f, country, year):
    return {
        "Country":   country,
        "ADM1_CODE": f.get("ADM1_CODE"),
        "ADM2_CODE": f.get("ADM2_CODE"),
        "ADM2_NAME": f.get("ADM2_NAME"),
        "Year":      year,
    }


def build_population_year(
    country, year, worldpop, gaul2, building_mask, pop_fit
) -> ee.FeatureCollection:
    pop     = _get_pop_image(worldpop, year, pop_fit).updateMask(building_mask)
    regions = adm2_regions(gaul2, country)
    reduced = pop.reduceRegions(
        collection=regions,
        reducer=ee.Reducer.median()
            .combine(ee.Reducer.stdDev(), "", True)
            .combine(ee.Reducer.sum(), "", True),
        scale=100,
        tileScale=4,
    )
    return reduced.map(
        lambda f: ee.Feature(None, {
            **_id_props(f, country, year),
            "Median_Pop": f.get("median"),
            "StdDev_Pop": f.get("stdDev"),
            "Total_Pop":  f.get("sum"),
        })
    )


def build_gpp_year(
    country, year, modis_gpp, gaul2, building_mask
) -> ee.FeatureCollection:
    start   = ee.Date.fromYMD(year, 1, 1)
    end     = ee.Date.fromYMD(year + 1, 1, 1)
    gpp     = modis_gpp.filterDate(start, end).mean().updateMask(building_mask)
    regions = adm2_regions(gaul2, country)
    reduced = gpp.reduceRegions(
        collection=regions,
        reducer=ee.Reducer.sum().combine(ee.Reducer.stdDev(), "", True),
        scale=500,
        tileScale=4,
    )

    def _map(f):
        total    = f.get("sum")
        mean_gpp = ee.Algorithms.If(
            ee.Algorithms.IsEqual(total, None),
            None,
            ee.Number(total).divide(f.geometry().area(1)),
        )
        return ee.Feature(None, {
            **_id_props(f, country, year),
            "Mean_GPP":   mean_gpp,
            "StdDev_GPP": f.get("stdDev"),
        })

    return reduced.map(_map)


def build_lst_year(
    country, year, viirs_lst, modis_lst, gaul2, building_mask
) -> ee.FeatureCollection:
    start      = ee.Date.fromYMD(year, 1, 1)
    end        = ee.Date.fromYMD(year + 1, 1, 1)
    viirs_year = viirs_lst.filterDate(start, end)
    modis_year = modis_lst.filterDate(start, end)
    lst_image  = ee.Image(
        ee.Algorithms.If(
            viirs_year.size().gt(0),
            viirs_year.mean(),
            modis_year.mean().multiply(0.02),
        )
    ).updateMask(building_mask)
    regions = adm2_regions(gaul2, country)
    reduced = lst_image.reduceRegions(
        collection=regions,
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), "", True),
        scale=1000,
        tileScale=4,
    )
    return reduced.map(
        lambda f: ee.Feature(None, {
            **_id_props(f, country, year),
            "Mean_LST":   f.get("mean"),
            "StdDev_LST": f.get("stdDev"),
        })
    )


def build_ntl_year(
    country, year, viirs_ntl, gaul2, building_mask
) -> ee.FeatureCollection:
    start   = ee.Date.fromYMD(year, 1, 1)
    end     = ee.Date.fromYMD(year + 1, 1, 1)
    ntl     = viirs_ntl.filterDate(start, end).mean().updateMask(building_mask)
    regions = adm2_regions(gaul2, country)
    reduced = ntl.reduceRegions(
        collection=regions,
        reducer=ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), "", True)
            .combine(ee.Reducer.sum(), "", True),
        scale=500,
        tileScale=4,
    )
    return reduced.map(
        lambda f: ee.Feature(None, {
            **_id_props(f, country, year),
            "Mean_NTL":   f.get("mean"),
            "StdDev_NTL": f.get("stdDev"),
            "Sum_NTL":    f.get("sum"),
        })
    )


def build_ndvi_year(
    country, year, ndvi_col, gaul2, building_mask
) -> ee.FeatureCollection:
    start   = ee.Date.fromYMD(year, 1, 1)
    end     = ee.Date.fromYMD(year + 1, 1, 1)
    ndvi    = ndvi_col.filterDate(start, end).mean().updateMask(building_mask)
    regions = adm2_regions(gaul2, country)
    reduced = ndvi.reduceRegions(
        collection=regions,
        reducer=ee.Reducer.median().combine(ee.Reducer.stdDev(), "", True),
        scale=500,
        tileScale=4,
    )
    return reduced.map(
        lambda f: ee.Feature(None, {
            **_id_props(f, country, year),
            "Median_NDVI": f.get("median"),
            "StdDev_NDVI": f.get("stdDev"),
        })
    )


def build_collection(
    metric, gaul2, worldpop, building_mask, pop_fit,
    modis_gpp, viirs_lst, modis_lst, viirs_ntl, ndvi_col,
) -> ee.FeatureCollection:
    parts = []
    for country in COUNTRIES:
        for year in YEARS:
            if metric == "population":
                parts.append(build_population_year(country, year, worldpop, gaul2, building_mask, pop_fit))
            elif metric == "gpp":
                parts.append(build_gpp_year(country, year, modis_gpp, gaul2, building_mask))
            elif metric == "lst":
                parts.append(build_lst_year(country, year, viirs_lst, modis_lst, gaul2, building_mask))
            elif metric == "ntl":
                parts.append(build_ntl_year(country, year, viirs_ntl, gaul2, building_mask))
            elif metric == "ndvi":
                parts.append(build_ndvi_year(country, year, ndvi_col, gaul2, building_mask))
    return ee.FeatureCollection(parts).flatten()


def main() -> None:
    args = parse_args()
    initialize_ee()

    gaul2    = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
    worldpop = ee.ImageCollection("WorldPop/GP/100m/pop").select("population")
    modis_gpp = ee.ImageCollection("MODIS/061/MOD17A3HGF").select("Gpp")
    viirs_lst = ee.ImageCollection("NASA/VIIRS/002/VNP21A1N").select("LST_1KM")
    modis_lst = ee.ImageCollection("MODIS/006/MOD11A2").select("LST_Night_1km")
    viirs_ntl = ee.ImageCollection("NOAA/VIIRS/001/VNP46A2").select(
        "Gap_Filled_DNB_BRDF_Corrected_NTL"
    )
    ndvi_col  = ee.ImageCollection("MODIS/061/MOD09A1").map(
        lambda img: img.normalizedDifference(["sur_refl_b02", "sur_refl_b01"])
            .rename("value")
            .copyProperties(img, img.propertyNames())
    )

    print("Building 500m building mask ...")
    building_mask = build_building_mask()

    print("Building pixel-level population trend (linearFit over 2000-2020) ...")
    pop_fit = build_pop_fit(worldpop)

    folder  = args.drive_folder
    metrics = ["population", "gpp", "lst", "ntl", "ndvi"]
    tasks   = []

    for metric in metrics:
        fc = build_collection(
            metric, gaul2, worldpop, building_mask, pop_fit,
            modis_gpp, viirs_lst, modis_lst, viirs_ntl, ndvi_col,
        )
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=f"adm2_{metric}",
            folder=folder,
            fileNamePrefix=f"adm2_{metric}",
            fileFormat="CSV",
        )
        tasks.append(task)
        print(f"  Task created: adm2_{metric}")

    print(f"\n{len(tasks)} tasks created | Drive folder: {folder}")

    if args.start_tasks:
        for task in tasks:
            task.start()
        print("All 5 tasks started — monitor at code.earthengine.google.com/tasks")
    else:
        print("Re-run with --start-tasks to launch.")


if __name__ == "__main__":
    main()
