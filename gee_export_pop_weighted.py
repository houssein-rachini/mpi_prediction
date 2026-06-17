"""
Earth Engine export script — population-weighted feature aggregation
with 500m building mask.

Per-pixel transform (after applying building mask to both value and population):

    new_pixel_i = (pixel_value_i × pixel_population_i) / total_region_population

where total_region_population = sum of masked population pixels in the ADM1 region.

Per-ADM1 reductions on new_pixel_i values:
    mean, median, stdDev, sum, min, max

Note: sum(new_pixel) == population-weighted mean of pixel_value.

Building mask: JRC GHSL built height >= 2.5m, dilated 500m — identical
to the 500m variant of gee_export_all_vars.py.

Population metric uses standard spatial stats (mean/median/stdDev/sum/min/max)
on masked population pixels directly — pop-weighting population by itself is
not meaningful.

Post-2020 population: pixel-level linear extrapolation via linearFit over
WorldPop 2000-2020. Computed once globally and reused.

Same 5 Drive export tasks (population, gpp, lst, ntl, ndvi).
Same 82 countries, years 2012-2023.

Run:
    python gee_export_pop_weighted.py --start-tasks
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

YEARS_ALL = list(range(2012, 2024))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export population-weighted ADM1 feature tables to Google Drive."
    )
    parser.add_argument(
        "--drive-folder",
        default="gaul82_vars_pop_weighted",
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
    """JRC GHSL built height >= 2.5m, dilated 500m — same as gee_export_all_vars.py."""
    return (
        ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018")
        .select("built_height")
        .gte(2.5)
        .focalMax(kernel=ee.Kernel.circle(radius=500, units="meters"))
        .selfMask()
    )


def adm1_regions(gaul: ee.FeatureCollection, country: str) -> ee.FeatureCollection:
    return gaul.filter(ee.Filter.eq("ADM0_NAME", country)).select(["ADM1_NAME"])


def build_pop_fit(worldpop: ee.ImageCollection) -> ee.Image:
    """
    Pixel-wise linear trend over WorldPop 2000-2020.
    Returns 2-band image: 'scale' (slope), 'offset' (intercept).
    Computed once and reused across all countries, years, and metrics.
    """

    def _add_year_band(img):
        y = ee.Date(img.get("system:time_start")).get("year").toFloat()
        masked = img.updateMask(img.mask())
        return masked.addBands(ee.Image.constant(y).rename("year")).toFloat()

    return (
        worldpop.map(_add_year_band)
        .select(["year", "population"])
        .reduce(ee.Reducer.linearFit())
    )


def _get_pop_image(
    worldpop: ee.ImageCollection, year: int, pop_fit: ee.Image
) -> ee.Image:
    """
    Actual WorldPop for years <= 2020; pixel-level linear extrapolation for > 2020.
    Extrapolation: intercept + slope * year, clipped to >= 0.
    """
    if year <= 2020:
        return worldpop.filterDate(
            ee.Date.fromYMD(year, 1, 1),
            ee.Date.fromYMD(year + 1, 1, 1),
        ).mean()
    slope = pop_fit.select("scale")
    intercept = pop_fit.select("offset")
    return intercept.add(slope.multiply(year)).max(ee.Image(0))


def _all_stats_reducer() -> ee.Reducer:
    return (
        ee.Reducer.mean()
        .combine(ee.Reducer.median(), "", True)
        .combine(ee.Reducer.stdDev(), "", True)
        .combine(ee.Reducer.sum(), "", True)
        .combine(ee.Reducer.min(), "", True)
        .combine(ee.Reducer.max(), "", True)
    )


def _pop_weighted_stats(
    value_image: ee.Image,
    pop: ee.Image,
    regions: ee.FeatureCollection,
    scale: int,
) -> ee.FeatureCollection:
    """
    Transforms each masked pixel:
        new_pixel_i = (value_i * pop_i) / total_region_population

    Then computes mean, median, stdDev, sum, min, max of new_pixel per ADM1.

    Note: sum(new_pixel) == population-weighted mean of value_image.

    Both value_image and pop must already have the building mask applied before calling.

    Two-pass approach:
      Pass 1: reduceRegions to get total_pop per region.
      Pass 2: rasterize total_pop via reduceToImage, compute new_pixel, reduceRegions for all stats.
    """
    # Pass 1: total population per ADM1 at analysis scale
    pop_sum_fc = pop.reduceRegions(
        collection=regions,
        reducer=ee.Reducer.sum(),
        scale=scale,
        tileScale=4,
    )

    # Pass 2: paint total_pop back to pixel level; GEE names the output band "sum"
    pop_total_image = pop_sum_fc.reduceToImage(["sum"], ee.Reducer.first())

    # Transformed pixel: value * pop / total_pop; mask where total_pop == 0
    new_pixel = (
        value_image.multiply(pop)
        .divide(pop_total_image)
        .rename("new_pixel")
        .updateMask(pop_total_image.gt(0))
    )

    return new_pixel.reduceRegions(
        collection=regions,
        reducer=_all_stats_reducer(),
        scale=scale,
        tileScale=4,
    )


def build_population_year(
    country: str,
    year: int,
    worldpop: ee.ImageCollection,
    gaul: ee.FeatureCollection,
    building_mask: ee.Image,
    pop_fit: ee.Image,
) -> ee.FeatureCollection:
    # Standard spatial stats on masked population pixels (no self-weighting)
    pop = _get_pop_image(worldpop, year, pop_fit).updateMask(building_mask)
    regions = adm1_regions(gaul, country)

    reduced = pop.reduceRegions(
        collection=regions,
        reducer=_all_stats_reducer(),
        scale=100,
        tileScale=4,
    )
    return reduced.map(
        lambda f: ee.Feature(
            None,
            {
                "Country": country,
                "Region": f.get("ADM1_NAME"),
                "Year": year,
                "Mean Population": f.get("mean"),
                "Median Population": f.get("median"),
                "Std Dev Population": f.get("stdDev"),
                "Total Population": f.get("sum"),
                "Min Population": f.get("min"),
                "Max Population": f.get("max"),
            },
        )
    )


def build_gpp_year(
    country: str,
    year: int,
    modis_gpp: ee.ImageCollection,
    worldpop: ee.ImageCollection,
    gaul: ee.FeatureCollection,
    building_mask: ee.Image,
    pop_fit: ee.Image,
) -> ee.FeatureCollection:
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year + 1, 1, 1)
    pop = _get_pop_image(worldpop, year, pop_fit).updateMask(building_mask)
    gpp = modis_gpp.filterDate(start, end).mean().updateMask(building_mask)
    regions = adm1_regions(gaul, country)

    pw_reduced = _pop_weighted_stats(gpp, pop, regions, scale=500)

    # Mean GPP = sum(masked_gpp_pixels) / region_area_m² — same formula as gee_export_all_vars.py
    gpp_sum_fc = gpp.reduceRegions(
        collection=regions,
        reducer=ee.Reducer.sum(),
        scale=500,
        tileScale=4,
    )
    joined = ee.Join.saveFirst("gpp_sum_feat").apply(
        primary=pw_reduced,
        secondary=gpp_sum_fc,
        condition=ee.Filter.equals(leftField="ADM1_NAME", rightField="ADM1_NAME"),
    )

    def _map(f):
        raw_sum = ee.Feature(f.get("gpp_sum_feat")).get("sum")
        mean_gpp = ee.Algorithms.If(
            ee.Algorithms.IsEqual(raw_sum, None),
            None,
            ee.Number(raw_sum).divide(f.geometry().area(1)),
        )
        return ee.Feature(
            None,
            {
                "Country": country,
                "Region": f.get("ADM1_NAME"),
                "Year": year,
                "Mean GPP": mean_gpp,
                "Median GPP": f.get("median"),
                "Std Dev GPP": f.get("stdDev"),
                "Sum GPP": f.get("sum"),
                "Min GPP": f.get("min"),
                "Max GPP": f.get("max"),
            },
        )

    return ee.FeatureCollection(joined).map(_map)


def build_lst_year(
    country: str,
    year: int,
    viirs_lst: ee.ImageCollection,
    modis_lst: ee.ImageCollection,
    worldpop: ee.ImageCollection,
    gaul: ee.FeatureCollection,
    building_mask: ee.Image,
    pop_fit: ee.Image,
) -> ee.FeatureCollection:
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year + 1, 1, 1)
    pop = _get_pop_image(worldpop, year, pop_fit).updateMask(building_mask)
    regions = adm1_regions(gaul, country)

    viirs_year = viirs_lst.filterDate(start, end)
    modis_year = modis_lst.filterDate(start, end)

    lst_image = ee.Image(
        ee.Algorithms.If(
            viirs_year.size().gt(0),
            viirs_year.mean(),
            modis_year.mean().multiply(0.02),
        )
    ).updateMask(building_mask)

    reduced = _pop_weighted_stats(lst_image, pop, regions, scale=1000)

    return reduced.map(
        lambda f: ee.Feature(
            None,
            {
                "Country": country,
                "Region": f.get("ADM1_NAME"),
                "Year": year,
                "Mean LST (K)": f.get("mean"),
                "Median LST (K)": f.get("median"),
                "Std Dev LST": f.get("stdDev"),
                "Sum LST": f.get("sum"),
                "Min LST (K)": f.get("min"),
                "Max LST (K)": f.get("max"),
            },
        )
    )


def build_ntl_year(
    country: str,
    year: int,
    viirs_ntl: ee.ImageCollection,
    worldpop: ee.ImageCollection,
    gaul: ee.FeatureCollection,
    building_mask: ee.Image,
    pop_fit: ee.Image,
) -> ee.FeatureCollection:
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year + 1, 1, 1)
    pop = _get_pop_image(worldpop, year, pop_fit).updateMask(building_mask)
    ntl = viirs_ntl.filterDate(start, end).mean().updateMask(building_mask)
    regions = adm1_regions(gaul, country)

    reduced = _pop_weighted_stats(ntl, pop, regions, scale=500)

    return reduced.map(
        lambda f: ee.Feature(
            None,
            {
                "Country": country,
                "Region": f.get("ADM1_NAME"),
                "Year": year,
                "Mean NTL": f.get("mean"),
                "Median NTL": f.get("median"),
                "Std Dev NTL": f.get("stdDev"),
                "Sum NTL": f.get("sum"),
                "Min NTL": f.get("min"),
                "Max NTL": f.get("max"),
            },
        )
    )


def build_ndvi_year(
    country: str,
    year: int,
    ndvi_col: ee.ImageCollection,
    worldpop: ee.ImageCollection,
    gaul: ee.FeatureCollection,
    building_mask: ee.Image,
    pop_fit: ee.Image,
) -> ee.FeatureCollection:
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year + 1, 1, 1)
    pop = _get_pop_image(worldpop, year, pop_fit).updateMask(building_mask)
    ndvi = ndvi_col.filterDate(start, end).mean().updateMask(building_mask)
    regions = adm1_regions(gaul, country)

    reduced = _pop_weighted_stats(ndvi, pop, regions, scale=500)

    return reduced.map(
        lambda f: ee.Feature(
            None,
            {
                "Country": country,
                "Region": f.get("ADM1_NAME"),
                "Year": year,
                "Mean NDVI": f.get("mean"),
                "Median NDVI": f.get("median"),
                "Std Dev NDVI": f.get("stdDev"),
                "Sum NDVI": f.get("sum"),
                "Min NDVI": f.get("min"),
                "Max NDVI": f.get("max"),
            },
        )
    )


def build_collection(
    metric: str,
    gaul,
    worldpop,
    building_mask,
    pop_fit,
    modis_gpp,
    viirs_lst,
    modis_lst,
    viirs_ntl,
    ndvi_col,
) -> ee.FeatureCollection:
    parts = []
    for country in GAUL_COUNTRIES:
        for year in YEARS_ALL:
            if metric == "population":
                parts.append(
                    build_population_year(
                        country, year, worldpop, gaul, building_mask, pop_fit
                    )
                )
            elif metric == "gpp":
                parts.append(
                    build_gpp_year(
                        country, year, modis_gpp, worldpop, gaul, building_mask, pop_fit
                    )
                )
            elif metric == "lst":
                parts.append(
                    build_lst_year(
                        country,
                        year,
                        viirs_lst,
                        modis_lst,
                        worldpop,
                        gaul,
                        building_mask,
                        pop_fit,
                    )
                )
            elif metric == "ntl":
                parts.append(
                    build_ntl_year(
                        country, year, viirs_ntl, worldpop, gaul, building_mask, pop_fit
                    )
                )
            elif metric == "ndvi":
                parts.append(
                    build_ndvi_year(
                        country, year, ndvi_col, worldpop, gaul, building_mask, pop_fit
                    )
                )
    return ee.FeatureCollection(parts).flatten()


def main() -> None:
    args = parse_args()
    initialize_ee()

    gaul = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")
    worldpop = ee.ImageCollection("WorldPop/GP/100m/pop").select("population")
    modis_gpp = ee.ImageCollection("MODIS/061/MOD17A3HGF").select("Gpp")
    viirs_lst = ee.ImageCollection("NASA/VIIRS/002/VNP21A1N").select("LST_1KM")
    modis_lst = ee.ImageCollection("MODIS/006/MOD11A2").select("LST_Night_1km")
    viirs_ntl = ee.ImageCollection("NOAA/VIIRS/001/VNP46A2").select(
        "Gap_Filled_DNB_BRDF_Corrected_NTL"
    )
    ndvi_col = ee.ImageCollection("MODIS/061/MOD09A1").map(
        lambda img: img.normalizedDifference(["sur_refl_b02", "sur_refl_b01"])
        .rename("value")
        .copyProperties(img, img.propertyNames())
    )

    print("Building 500m building mask (GHSL >= 2.5m, focalMax 500m) ...")
    building_mask = build_building_mask()

    print("Building pixel-level population trend (linearFit over 2000-2020) ...")
    pop_fit = build_pop_fit(worldpop)

    folder = args.drive_folder
    metrics = ["population", "gpp", "lst", "ntl", "ndvi"]
    tasks = []
    for metric in metrics:
        fc = build_collection(
            metric,
            gaul,
            worldpop,
            building_mask,
            pop_fit,
            modis_gpp,
            viirs_lst,
            modis_lst,
            viirs_ntl,
            ndvi_col,
        )
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=f"pop_weighted_{metric}",
            folder=folder,
            fileNamePrefix=f"pop_weighted_{metric}",
            fileFormat="CSV",
        )
        tasks.append(task)
        print(f"  Task created: pop_weighted_{metric}")

    print(f"\n{len(tasks)} tasks created | Drive folder: {folder}")

    if args.start_tasks:
        for task in tasks:
            task.start()
        print("All 5 tasks started — monitor at code.earthengine.google.com/tasks")
    else:
        print("Re-run with --start-tasks to launch.")


if __name__ == "__main__":
    main()
