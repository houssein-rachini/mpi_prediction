import streamlit as st
import folium
import ee
import numpy as np
from streamlit_folium import folium_static
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
import xgboost as xgb
from ee_auth import initialize_earth_engine
from predictions import (
    preprocess_data,
    plot_results,
    load_dnn_model,
    load_dnn_scaler,
    load_ml_model,
    load_ml_scaler,
    load_ensemble_models,
    load_ensemble_scaler,
    load_quantile_models,
    load_quantile_scaler,
    load_stacked_artifacts,
    predict_dnn_fast,
    predict_ml_fast,
    predict_ensemble_fast,
    predict_quantile_fast,
    predict_stacked_fast,
)

from concurrent.futures import ThreadPoolExecutor
import branca.colormap as cm
import os
import time
from math import ceil

batch_size = 10
GEE_STAT_WORKERS = 1
GEE_REGION_WORKERS = 1
GEE_REGION_DELAY_SECONDS = 0.25
GEE_MAX_RATE_LIMIT_RETRIES = 4
PREDICTION_CACHE_VERSION = "v2"

initialize_earth_engine()

# ---------------- Building mask (GHSL 2018, height >= 2.5m, variable dilation) -------
BUFFER_RADIUS_MAP = {
    "0m buffer": 0,
    "250m buffer": 250,
    "500m buffer": 500,
    "1000m buffer": 1000,
    "2000m buffer": 2000,
    "3000m buffer": 3000,
}


def get_selected_buffer_radius_m():
    selected_buffer = st.session_state.get("selected_buffer", "500m buffer")
    return BUFFER_RADIUS_MAP.get(selected_buffer, 500)


@st.cache_resource
def get_building_mask(buffer_radius_m):
    return (
        ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018")
        .select("built_height")
        .gte(2.5)
        .focal_max(kernel=ee.Kernel.circle(radius=buffer_radius_m, units="meters"))
    )


def get_active_building_mask():
    return get_building_mask(get_selected_buffer_radius_m())


# --------------------------------------------------------------------------------------

# -------------------------- TR Grouped Regions (asset) --------------------------------
TR_ASSET_ID = "projects/ee-housseinrachini213/assets/TR_regions_admin1_groups"
try:
    TR_FC = ee.FeatureCollection(TR_ASSET_ID)
except Exception:
    TR_FC = None  # keep app usable if asset isn't available


@st.cache_resource
def tr_region_codes():
    if TR_FC is None:
        return []
    return TR_FC.aggregate_array("region_code").getInfo()


@st.cache_resource
def tr_code_to_adm1_list():
    if TR_FC is None:
        return {}
    codes = TR_FC.aggregate_array("region_code").getInfo()
    names = TR_FC.aggregate_array("adm1_list").getInfo()
    return dict(zip(codes, names))


@st.cache_resource
def tr_region_geometry(code):
    if TR_FC is None:
        return None
    feat = TR_FC.filter(ee.Filter.eq("region_code", code)).first()
    if feat is None:
        return None
    return ee.Feature(feat).geometry().getInfo()


# --------------------------------------------------------------------------------------

# ----------------------- Datasets -----------------------------------------------------
fao_gaul = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")
fao_gaul_lvl2 = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
worldpop = ee.ImageCollection("WorldPop/GP/100m/pop").select("population")
modis_gpp = ee.ImageCollection("MODIS/061/MOD17A3HGF").select("Gpp")
viirs_lst = ee.ImageCollection("NASA/VIIRS/002/VNP21A1N").select("LST_1KM")
viirs_ntl = ee.ImageCollection("NOAA/VIIRS/001/VNP46A2").select(
    "Gap_Filled_DNB_BRDF_Corrected_NTL"
)
ndvi_collection = ee.ImageCollection("MODIS/MOD09GA_006_NDVI").select("NDVI")
ndvi_v2 = ee.ImageCollection("MODIS/061/MOD09A1")
modis_lst_day = ee.ImageCollection("MODIS/061/MOD11A2").select("LST_Day_1km")

GHSL_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]

def _nearest_ghsl_epoch(year):
    return min(GHSL_EPOCHS, key=lambda e: abs(e - year))
# --------------------------------------------------------------------------------------

# ----------------------- Pretrained file map ------------------------------------------
REQUIRED_PRETRAINED_FILES = {
    "DNN": [
        "models/global/trained_dnn_model.h5",
        "models/global/dnn_scaler.pkl",
    ],
    "ML": [
        "models/global/trained_ml_model.pkl",
        "models/global/ml_scaler.pkl",
    ],
    "DNN+XGBoost": [
        "models/global/trained_ensemble_xgb_dnn_model.h5",
        "models/global/trained_ensemble_xgb_model.json",
        "models/global/ensemble_scaler.pkl",
    ],
    "DNN+LightGBM": [
        "models/global/trained_ensemble_lgbm_dnn_model.h5",
        "models/global/trained_ensemble_lgbm_model.pkl",
        "models/global/ensemble_scaler.pkl",
    ],
    "DNN+RF": [
        "models/global/trained_ensemble_rf_dnn_model.h5",
        "models/global/trained_ensemble_rf_model.pkl",
        "models/global/ensemble_scaler.pkl",
    ],
    "DNN+KNN": [
        "models/global/trained_ensemble_knn_dnn_model.h5",
        "models/global/trained_ensemble_knn_model.pkl",
        "models/global/ensemble_scaler.pkl",
    ],
    "Stacked": [
        "models/global/stacked/metadata.json",
        "models/global/stacked/meta_model.pkl",
        "models/global/stacked/scaler.pkl",
    ],
    "XGB-Quantile": [
        "models/global/trained_xgb_quantile_q05.json",
        "models/global/trained_xgb_quantile_q50.json",
        "models/global/trained_xgb_quantile_q95.json",
        "models/global/quantile_scaler.pkl",
    ],
}
# --------------------------------------------------------------------------------------


def compute_sev_pov(mpi):
    pov = 0.04133 + 34.58 * mpi + 263 * (mpi**2) - 180.8 * (mpi**3)
    return min(100, max(0, pov))


def compute_ndvi(image):
    ndvi = image.normalizedDifference(["sur_refl_b02", "sur_refl_b01"]).rename("NDVI")
    return ndvi.copyProperties(image, image.propertyNames())


ndvi_v2 = ndvi_v2.map(compute_ndvi)


def chunk_list(lst, size):
    return [lst[i : i + size] for i in range(0, len(lst), size)]


# ----------------------- Country/Region lists -----------------------------------------
@st.cache_resource
def get_country_list():
    c_list = fao_gaul.aggregate_array("ADM0_NAME").distinct().getInfo()
    c_list.sort()
    return c_list


# note: include use_tr_asset in signature so Streamlit cache differentiates
@st.cache_resource
def get_region_list(country, use_tr_asset=False):
    if country == "Turkey" and use_tr_asset and TR_FC is not None:
        return tr_region_codes()
    return (
        fao_gaul.filter(ee.Filter.eq("ADM0_NAME", country))
        .aggregate_array("ADM1_NAME")
        .distinct()
        .getInfo()
    )


@st.cache_resource
def get_region_list_lvl2(country):
    fc_filtered = fao_gaul_lvl2.filter(ee.Filter.eq("ADM0_NAME", country))
    return fc_filtered.aggregate_array("ADM2_CODE").getInfo()


# --------------------------------------------------------------------------------------


# ----------------------- Geometry helpers ---------------------------------------------
@st.cache_resource
def get_region_geometry(country, region, use_tr_asset=False):
    """Governorate/TR grouping geometry. If TR toggle is ON, region is a TR code."""
    if country == "Turkey" and use_tr_asset and TR_FC is not None:
        return tr_region_geometry(region)
    filtered = fao_gaul.filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", country),
            ee.Filter.eq("ADM1_NAME", region),
        )
    )
    return filtered.geometry().getInfo()


@st.cache_resource
def get_region_geometry_lvl2(country, adm2_code):
    fc = fao_gaul_lvl2.filter(ee.Filter.eq("ADM0_NAME", country))
    filtered = fc.filter(ee.Filter.eq("ADM2_CODE", adm2_code))
    geom = filtered.geometry().getInfo()
    return geom


@st.cache_resource
def get_district_name_from_adm2code(country, adm2_code):
    fc = fao_gaul_lvl2.filter(ee.Filter.eq("ADM0_NAME", country))
    filtered = fc.filter(ee.Filter.eq("ADM2_CODE", adm2_code))
    f = filtered.first()
    return f.get("ADM2_NAME").getInfo() if f is not None else None


# --------------------------------------------------------------------------------------


# ----------------------- Stats (with building mask & guards) --------------------------
def interpolate_population(region_geom, selected_year):
    def is_masked_empty(image, band, scale):
        count = image.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=ee.Geometry(region_geom),
            scale=scale,
            bestEffort=True,
        ).get(band)
        return ee.Number(count).lt(1)

    if selected_year <= 2020:
        start_date = ee.Date.fromYMD(selected_year, 1, 1)
        end_date = ee.Date.fromYMD(selected_year, 12, 31)
        image = (
            worldpop.filterDate(start_date, end_date).mean().updateMask(get_active_building_mask())
        )

        if is_masked_empty(image, "population", 100).getInfo():
            return None

        stats = image.reduceRegion(
            reducer=ee.Reducer.mean()
            .combine(ee.Reducer.min(), None, True)
            .combine(ee.Reducer.max(), None, True)
            .combine(ee.Reducer.median(), None, True)
            .combine(ee.Reducer.stdDev(), None, True)
            .combine(ee.Reducer.sum(), None, True),
            geometry=ee.Geometry(region_geom),
            scale=100,
            bestEffort=True,
        ).getInfo()

        if "population_mean" not in stats:
            return None

        return {
            "Mean Population": round(stats["population_mean"], 5),
            "Total Population": round(stats["population_sum"], 5),
            "Min Population": round(stats["population_min"], 5),
            "Max Population": round(stats["population_max"], 5),
            "Median Population": round(stats["population_median"], 5),
            "Std Dev Population": round(stats["population_stdDev"], 5),
        }

    else:
        years = list(range(2012, 2021))
        props = ["mean", "sum", "min", "max", "median", "stdDev"]
        data = {prop: [] for prop in props}

        # Stack all years into a multi-band image and reduce in one GEE call
        # instead of 9 separate calls with individual empty-checks.
        stacked = ee.Image.cat([
            worldpop
            .filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y, 12, 31))
            .mean()
            .updateMask(get_active_building_mask())
            .rename([f"pop_{y}"])
            for y in years
        ])
        all_stats = stacked.reduceRegion(
            reducer=ee.Reducer.mean()
            .combine(ee.Reducer.min(), None, True)
            .combine(ee.Reducer.max(), None, True)
            .combine(ee.Reducer.median(), None, True)
            .combine(ee.Reducer.stdDev(), None, True)
            .combine(ee.Reducer.sum(), None, True),
            geometry=ee.Geometry(region_geom),
            scale=100,
            bestEffort=True,
        ).getInfo()

        for year in years:
            mean_val = all_stats.get(f"pop_{year}_mean")
            if mean_val is not None:
                data["mean"].append(mean_val)
                data["sum"].append(all_stats.get(f"pop_{year}_sum"))
                data["min"].append(all_stats.get(f"pop_{year}_min"))
                data["max"].append(all_stats.get(f"pop_{year}_max"))
                data["median"].append(all_stats.get(f"pop_{year}_median"))
                data["stdDev"].append(all_stats.get(f"pop_{year}_stdDev"))
            else:
                for key in data:
                    data[key].append(None)

        def extrapolate(values, years, target_year):
            values = np.array(values, dtype=np.float64)
            years = np.array(years)
            mask = ~np.isnan(values)
            if mask.sum() < 2:
                return None
            growth = np.mean(np.diff(values[mask]) / np.diff(years[mask]))
            return values[mask][-1] + growth * (target_year - years[mask][-1])

        results = {}
        for key in data:
            extrapolated = extrapolate(data[key], years, selected_year)
            results[key] = round(extrapolated, 2) if extrapolated is not None else "N/A"

        return {
            "Mean Population": results["mean"],
            "Total Population": results["sum"],
            "Min Population": results["min"],
            "Max Population": results["max"],
            "Median Population": results["median"],
            "Std Dev Population": results["stdDev"],
        }


def compute_gpp_stats(region_geom, selected_year):
    start_date = ee.Date.fromYMD(selected_year, 1, 1)
    end_date = ee.Date.fromYMD(selected_year, 12, 31)

    image = modis_gpp.filterDate(start_date, end_date).mean().updateMask(get_active_building_mask())

    stats = image.reduceRegion(
        reducer=ee.Reducer.minMax()
        .combine(ee.Reducer.median(), "", True)
        .combine(ee.Reducer.stdDev(), "", True)
        .combine(ee.Reducer.sum(), "", True),
        geometry=ee.Geometry(region_geom),
        scale=500,
        bestEffort=True,
        maxPixels=1e13,
    ).getInfo()

    if "Gpp_sum" not in stats:
        return None

    masked_area = (
        image.mask()
        .multiply(ee.Image.pixelArea())
        .rename("Gpp")
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=ee.Geometry(region_geom),
            scale=500,
            bestEffort=True,
            maxPixels=1e13,
        )
    )
    area_m2 = masked_area.get("Gpp")
    mean_gpp = ee.Number(stats["Gpp_sum"]).divide(area_m2).getInfo()

    return {
        "Mean GPP": round(mean_gpp, 6),
        "Min GPP": round(stats["Gpp_min"], 5),
        "Max GPP": round(stats["Gpp_max"], 5),
        "Median GPP": round(stats["Gpp_median"], 5),
        "Std Dev GPP": round(stats["Gpp_stdDev"], 5),
        "Total GPP": round(stats["Gpp_sum"], 5),
    }


def compute_lst_stats(region_geom, selected_year):
    """
    Compute annual mean nighttime LST for a region, using VIIRS night under the building mask
    and falling back to MODIS night if VIIRS has no valid pixels.
    """
    start = ee.Date.fromYMD(selected_year, 1, 1)
    end = ee.Date.fromYMD(selected_year, 12, 31)

    viirs_img = viirs_lst.filterDate(start, end).mean().updateMask(get_active_building_mask())

    viirs_count = viirs_img.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=ee.Geometry(region_geom),
        scale=500,
        bestEffort=True,
        maxPixels=1e13,
    ).get("LST_1KM")

    modis_img = (
        ee.ImageCollection("MODIS/006/MOD11A2")
        .select("LST_Night_1km")
        .filterDate(start, end)
        .mean()
        .multiply(0.02)
        .updateMask(get_active_building_mask())
    )

    var_image = ee.Image(
        ee.Algorithms.If(ee.Number(viirs_count).gt(0), viirs_img, modis_img)
    )

    use_viirs = ee.Number(viirs_count).gt(0).getInfo()
    band = "LST_1KM" if use_viirs else "LST_Night_1km"

    stats = var_image.reduceRegion(
        reducer=(
            ee.Reducer.mean()
            .combine(ee.Reducer.minMax(), None, True)
            .combine(ee.Reducer.median(), None, True)
            .combine(ee.Reducer.stdDev(), None, True)
            .combine(ee.Reducer.sum(), None, True)
        ),
        geometry=ee.Geometry(region_geom),
        scale=500,
        bestEffort=True,
        maxPixels=1e13,
    ).getInfo()

    mean_key = f"{band}_mean"
    min_key = f"{band}_min"
    max_key = f"{band}_max"
    med_key = f"{band}_median"
    std_key = f"{band}_stdDev"
    sum_key = f"{band}_sum"

    if mean_key not in stats:
        return None

    return {
        "Mean LST (°K)": round(stats[mean_key], 5),
        "Min LST (°K)": round(stats[min_key], 5),
        "Max LST (°K)": round(stats[max_key], 5),
        "Median LST (°K)": round(stats[med_key], 5),
        "Std Dev LST": round(stats[std_key], 5),
        "Total LST": round(stats[sum_key], 5),
    }


def compute_ntl_stats(region_geom, selected_year):
    start_date = ee.Date.fromYMD(selected_year, 1, 1)
    end_date = ee.Date.fromYMD(selected_year, 12, 31)
    image = viirs_ntl.filterDate(start_date, end_date).mean().updateMask(get_active_building_mask())

    stats = image.reduceRegion(
        reducer=ee.Reducer.mean()
        .combine(ee.Reducer.minMax(), "", True)
        .combine(ee.Reducer.median(), "", True)
        .combine(ee.Reducer.stdDev(), "", True)
        .combine(ee.Reducer.sum(), "", True),
        geometry=ee.Geometry(region_geom),
        scale=500,
        bestEffort=True,
        maxPixels=1e13,
    ).getInfo()

    if "Gap_Filled_DNB_BRDF_Corrected_NTL_mean" not in stats:
        return None

    return {
        "Mean NTL": round(stats["Gap_Filled_DNB_BRDF_Corrected_NTL_mean"], 5),
        "Min NTL": round(stats["Gap_Filled_DNB_BRDF_Corrected_NTL_min"], 5),
        "Max NTL": round(stats["Gap_Filled_DNB_BRDF_Corrected_NTL_max"], 5),
        "Median NTL": round(stats["Gap_Filled_DNB_BRDF_Corrected_NTL_median"], 5),
        "Std Dev NTL": round(stats["Gap_Filled_DNB_BRDF_Corrected_NTL_stdDev"], 5),
        "Total NTL": round(stats["Gap_Filled_DNB_BRDF_Corrected_NTL_sum"], 5),
    }


def compute_ndvi_stats(region_geom, selected_year):
    start_date = ee.Date.fromYMD(selected_year, 1, 1)
    end_date = ee.Date.fromYMD(selected_year, 12, 31)
    image = ndvi_v2.filterDate(start_date, end_date).mean().updateMask(get_active_building_mask())

    stats = image.reduceRegion(
        reducer=ee.Reducer.mean()
        .combine(ee.Reducer.minMax(), "", True)
        .combine(ee.Reducer.median(), "", True)
        .combine(ee.Reducer.stdDev(), "", True)
        .combine(ee.Reducer.sum(), "", True),
        geometry=ee.Geometry(region_geom),
        scale=500,
        bestEffort=True,
        maxPixels=1e13,
    ).getInfo()

    if "NDVI_mean" not in stats:
        return None

    return {
        "Mean NDVI": round(stats["NDVI_mean"], 5),
        "Min NDVI": round(stats["NDVI_min"], 5),
        "Max NDVI": round(stats["NDVI_max"], 5),
        "Median NDVI": round(stats["NDVI_median"], 5),
        "Std Dev NDVI": round(stats["NDVI_stdDev"], 5),
        "Total NDVI": round(stats["NDVI_sum"], 5),
    }


def compute_lst_day_stats(region_geom, selected_year):
    start = ee.Date.fromYMD(selected_year, 1, 1)
    end   = ee.Date.fromYMD(selected_year, 12, 31)
    img = (
        modis_lst_day
        .filterDate(start, end)
        .mean()
        .multiply(0.02)
        .updateMask(get_active_building_mask())
    )
    stats = img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=ee.Geometry(region_geom),
        scale=1000,
        bestEffort=True,
        maxPixels=1e13,
    ).getInfo()
    if not stats or stats.get("LST_Day_1km") is None:
        return None
    return {"Mean_LST_Day": round(stats["LST_Day_1km"], 5)}


def compute_ghsl_stats(region_geom, selected_year):
    epoch = _nearest_ghsl_epoch(selected_year)
    built_s = (
        ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{epoch}")
        .select("built_surface")
        .updateMask(get_active_building_mask())
    )
    built_v = (
        ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_V/{epoch}")
        .select("built_volume_total")
        .updateMask(get_active_building_mask())
    )
    s_stats = built_s.reduceRegion(
        reducer=(
            ee.Reducer.mean()
            .combine(ee.Reducer.median(), None, True)
            .combine(ee.Reducer.stdDev(), None, True)
        ),
        geometry=ee.Geometry(region_geom),
        scale=100,
        bestEffort=True,
        maxPixels=1e13,
    ).getInfo()
    v_stats = built_v.reduceRegion(
        reducer=ee.Reducer.stdDev(),
        geometry=ee.Geometry(region_geom),
        scale=100,
        bestEffort=True,
        maxPixels=1e13,
    ).getInfo()
    if not s_stats or s_stats.get("built_surface_mean") is None:
        return None
    return {
        "Mean_BUILT_S":   round(s_stats.get("built_surface_mean")         or 0, 5),
        "Median_BUILT_S": round(s_stats.get("built_surface_median")       or 0, 5),
        "StdDev_BUILT_S": round(s_stats.get("built_surface_stdDev")       or 0, 5),
        "StdDev_BUILT_V": round(v_stats.get("built_volume_total_stdDev")  or 0, 5),
    }


def compute_anomaly_stats(region_geom, target_year):
    """
    Fetches 2012-2019 baseline + lag year + target year for NTL, NDVI, LST_Night,
    LST_Day in two stacked GEE calls, then returns z-score anomalies.
    """
    baseline_yrs = list(range(2012, 2020))
    lag_year = target_year - 1
    extra = [lag_year] if lag_year >= 2012 else []
    all_years = sorted(set(baseline_yrs + extra + [target_year]))

    mask = get_active_building_mask()

    def _ntl(y):
        return (viirs_ntl.filterDate(f"{y}-01-01", f"{y}-12-31")
                .mean().select(["Gap_Filled_DNB_BRDF_Corrected_NTL"]).rename([f"NTL_{y}"]))

    def _lstn(y):
        return (viirs_lst.filterDate(f"{y}-01-01", f"{y}-12-31")
                .mean().select(["LST_1KM"]).rename([f"LSTN_{y}"]))

    def _lstd(y):
        return (modis_lst_day.filterDate(f"{y}-01-01", f"{y}-12-31")
                .mean().multiply(0.02).select(["LST_Day_1km"]).rename([f"LSTD_{y}"]))

    def _ndvi(y):
        return (ndvi_v2.filterDate(f"{y}-01-01", f"{y}-12-31")
                .mean().select(["NDVI"]).rename([f"NDVI_{y}"]))

    geom = ee.Geometry(region_geom)

    mean_img = ee.Image.cat(
        [_ntl(y)  for y in all_years] +
        [_lstn(y) for y in all_years] +
        [_lstd(y) for y in all_years]
    ).updateMask(mask)

    ndvi_img = ee.Image.cat([_ndvi(y) for y in all_years]).updateMask(mask)

    mean_stats = mean_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom, scale=500, bestEffort=True, maxPixels=1e13,
    ).getInfo()

    ndvi_stats_d = ndvi_img.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=geom, scale=500, bestEffort=True, maxPixels=1e13,
    ).getInfo()

    bl = [y for y in baseline_yrs if y in all_years]

    def _z(d, prefix, year):
        vals = [d.get(f"{prefix}_{y}") for y in bl]
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            return None
        mu  = np.mean(vals)
        sig = np.std(vals, ddof=1)
        if sig == 0:
            return 0.0
        val = d.get(f"{prefix}_{year}")
        return float((val - mu) / sig) if val is not None else None

    return {
        "NTL_anom":      _z(mean_stats,   "NTL",  target_year),
        "NDVI_anom":     _z(ndvi_stats_d, "NDVI", target_year),
        "LSTN_anom":     _z(mean_stats,   "LSTN", target_year),
        "LST_Day_anom":  _z(mean_stats,   "LSTD", target_year),
        "NTL_anom_lag1": _z(mean_stats,   "NTL",  lag_year) if lag_year >= 2012 else None,
    }


# --------------------------------------------------------------------------------------

MODEL_PATHS = {
    "DNN": "trained_dnn_model.h5",
    "ML": "trained_ml_model.pkl",
    "DNN+RF": "trained_ensemble_rf_dnn_model.h5",
    "DNN+XGBoost": "trained_ensemble_xgb_dnn_model.h5",
    "DNN+LightGBM": "trained_ensemble_lgbm_dnn_model.h5",
    "DNN+KNN": "trained_ensemble_knn_dnn_model.h5",
}

SCALER_PATHS = {
    "DNN": "dnn_scaler.pkl",
    "ML": "ml_scaler.pkl",
    "Ensemble": "ensemble_scaler.pkl",
}


# ----------------------- Cached wrappers ----------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_population_stats(region_geom, selected_year):
    return interpolate_population(region_geom, selected_year)


@st.cache_data(show_spinner=False)
def get_cached_gpp_stats(region_geom, selected_year):
    return compute_gpp_stats(region_geom, selected_year)


@st.cache_data(show_spinner=False)
def get_cached_lst_stats(region_geom, selected_year):
    return compute_lst_stats(region_geom, selected_year)


@st.cache_data(show_spinner=False)
def get_cached_ntl_stats(region_geom, selected_year):
    return compute_ntl_stats(region_geom, selected_year)


@st.cache_data(show_spinner=False)
def get_cached_ndvi_stats(region_geom, selected_year):
    return compute_ndvi_stats(region_geom, selected_year)


@st.cache_data(show_spinner=False)
def get_cached_lst_day_stats(region_geom, selected_year):
    return compute_lst_day_stats(region_geom, selected_year)


@st.cache_data(show_spinner=False)
def get_cached_ghsl_stats(region_geom, selected_year):
    return compute_ghsl_stats(region_geom, selected_year)


@st.cache_data(show_spinner=False)
def get_cached_anomaly_stats(region_geom, target_year):
    return compute_anomaly_stats(region_geom, target_year)


@st.cache_resource
def get_country_center(country):
    filtered = fao_gaul.filter(ee.Filter.eq("ADM0_NAME", country))
    coords = filtered.geometry().centroid().coordinates().getInfo()
    return coords if coords else [0, 0]


@st.cache_resource
def get_adm2code_to_governorate_map(country):
    features = fao_gaul_lvl2.filter(ee.Filter.eq("ADM0_NAME", country))
    adm2_codes = features.aggregate_array("ADM2_CODE").getInfo()
    govs       = features.aggregate_array("ADM1_NAME").getInfo()
    adm1_codes = features.aggregate_array("ADM1_CODE").getInfo()
    return {
        adc: {"adm1_name": g, "adm1_code": c1}
        for adc, g, c1 in zip(adm2_codes, govs, adm1_codes)
    }


# --------------------------------------------------------------------------------------


# ----------------------- Batch stat getters -------------------------------------------
def _is_ee_rate_limit_error(exc):
    message = str(exc).lower()
    return (
        "429" in message
        or "too many requests" in message
        or "rate limit" in message
        or "quota" in message
    )


def _call_gee_with_backoff(fn, *args):
    for attempt in range(GEE_MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return fn(*args)
        except Exception as exc:
            if not _is_ee_rate_limit_error(exc) or attempt == GEE_MAX_RATE_LIMIT_RETRIES:
                raise
            delay = min(2 ** attempt, 16)
            time.sleep(delay)
    return None


def _fetch_stat_parts(region_geom, selected_year):
    stat_calls = [
        get_cached_population_stats,
        get_cached_gpp_stats,
        get_cached_lst_stats,
        get_cached_ntl_stats,
        get_cached_ndvi_stats,
        get_cached_lst_day_stats,
        get_cached_ghsl_stats,
        get_cached_anomaly_stats,
    ]

    if GEE_STAT_WORKERS <= 1:
        return [
            _call_gee_with_backoff(fn, region_geom, selected_year)
            for fn in stat_calls
        ]

    with ThreadPoolExecutor(max_workers=GEE_STAT_WORKERS) as ex:
        futures = [
            ex.submit(_call_gee_with_backoff, fn, region_geom, selected_year)
            for fn in stat_calls
        ]
        return [future.result() for future in futures]


def get_all_stats_parallel(region, country, selected_year, use_tr_asset=False):
    try:
        region_geom = get_region_geometry(country, region, use_tr_asset)
        (
            pop_stats,
            gpp_stats,
            lst_stats,
            ntl_stats,
            ndvi_stats,
            lstd_stats,
            ghsl_stats,
            anom_stats,
        ) = _fetch_stat_parts(region_geom, selected_year)
        if not all([pop_stats, gpp_stats, lst_stats, ntl_stats, ndvi_stats]):
            print(f"[SKIP] No valid pixels in {country} - {region} - {selected_year}")
            return None
        feature_row = {
            "Mean_Pop": pop_stats["Mean Population"],
            "Total_Pop": pop_stats["Total Population"],
            "Min_Pop": pop_stats["Min Population"],
            "Max_Pop": pop_stats["Max Population"],
            "Median_Pop": pop_stats["Median Population"],
            "StdDev_Pop": pop_stats["Std Dev Population"],
            "Mean_GPP": gpp_stats["Mean GPP"],
            "Sum_GPP": gpp_stats["Total GPP"],
            "Min_GPP": gpp_stats["Min GPP"],
            "Max_GPP": gpp_stats["Max GPP"],
            "Median_GPP": gpp_stats["Median GPP"],
            "StdDev_GPP": gpp_stats["Std Dev GPP"],
            "Mean_LST": lst_stats["Mean LST (°K)"],
            "Sum_LST": lst_stats["Total LST"],
            "Min_LST": lst_stats["Min LST (°K)"],
            "Max_LST": lst_stats["Max LST (°K)"],
            "Median_LST": lst_stats["Median LST (°K)"],
            "StdDev_LST": lst_stats["Std Dev LST"],
            "Mean_NTL": ntl_stats["Mean NTL"],
            "Sum_NTL": ntl_stats["Total NTL"],
            "Min_NTL": ntl_stats["Min NTL"],
            "Max_NTL": ntl_stats["Max NTL"],
            "Median_NTL": ntl_stats["Median NTL"],
            "StdDev_NTL": ntl_stats["Std Dev NTL"],
            "Mean_NDVI": ndvi_stats["Mean NDVI"],
            "Sum_NDVI": ndvi_stats["Total NDVI"],
            "Min_NDVI": ndvi_stats["Min NDVI"],
            "Max_NDVI": ndvi_stats["Max NDVI"],
            "Median_NDVI": ndvi_stats["Median NDVI"],
            "StdDev_NDVI": ndvi_stats["Std Dev NDVI"],
            # LST Day
            "Mean_LST_Day": lstd_stats["Mean_LST_Day"] if lstd_stats else None,
            # Derived ratio
            "ndvi_lst_ratio": (
                ndvi_stats["Median NDVI"] / lst_stats["Mean LST (°K)"]
                if lst_stats.get("Mean LST (°K)") else None
            ),
            # GHSL building features
            "Mean_BUILT_S":   ghsl_stats.get("Mean_BUILT_S")   if ghsl_stats else None,
            "Median_BUILT_S": ghsl_stats.get("Median_BUILT_S") if ghsl_stats else None,
            "StdDev_BUILT_S": ghsl_stats.get("StdDev_BUILT_S") if ghsl_stats else None,
            "StdDev_BUILT_V": ghsl_stats.get("StdDev_BUILT_V") if ghsl_stats else None,
            # Anomalies
            "NTL_anom":      anom_stats.get("NTL_anom")      if anom_stats else None,
            "NDVI_anom":     anom_stats.get("NDVI_anom")     if anom_stats else None,
            "LSTN_anom":     anom_stats.get("LSTN_anom")     if anom_stats else None,
            "LST_Day_anom":  anom_stats.get("LST_Day_anom")  if anom_stats else None,
            "NTL_anom_lag1": anom_stats.get("NTL_anom_lag1") if anom_stats else None,
        }
        return (feature_row, pop_stats["Total Population"])
    except:
        return None


def get_all_stats_parallel_lvl2(region, country, selected_year):
    try:
        region_geom = get_region_geometry_lvl2(country, region)
        (
            pop_stats,
            gpp_stats,
            lst_stats,
            ntl_stats,
            ndvi_stats,
            lstd_stats,
            ghsl_stats,
            anom_stats,
        ) = _fetch_stat_parts(region_geom, selected_year)
        if not all([pop_stats, gpp_stats, lst_stats, ntl_stats, ndvi_stats]):
            print(f"[SKIP] No valid pixels in {country} - {region} - {selected_year}")
            return None
        feature_row = {
            "Mean_Pop": pop_stats["Mean Population"],
            "Total_Pop": pop_stats["Total Population"],
            "Min_Pop": pop_stats["Min Population"],
            "Max_Pop": pop_stats["Max Population"],
            "Median_Pop": pop_stats["Median Population"],
            "StdDev_Pop": pop_stats["Std Dev Population"],
            "Mean_GPP": gpp_stats["Mean GPP"],
            "Sum_GPP": gpp_stats["Total GPP"],
            "Min_GPP": gpp_stats["Min GPP"],
            "Max_GPP": gpp_stats["Max GPP"],
            "Median_GPP": gpp_stats["Median GPP"],
            "StdDev_GPP": gpp_stats["Std Dev GPP"],
            "Mean_LST": lst_stats["Mean LST (°K)"],
            "Sum_LST": lst_stats["Total LST"],
            "Min_LST": lst_stats["Min LST (°K)"],
            "Max_LST": lst_stats["Max LST (°K)"],
            "Median_LST": lst_stats["Median LST (°K)"],
            "StdDev_LST": lst_stats["Std Dev LST"],
            "Mean_NTL": ntl_stats["Mean NTL"],
            "Sum_NTL": ntl_stats["Total NTL"],
            "Min_NTL": ntl_stats["Min NTL"],
            "Max_NTL": ntl_stats["Max NTL"],
            "Median_NTL": ntl_stats["Median NTL"],
            "StdDev_NTL": ntl_stats["Std Dev NTL"],
            "Mean_NDVI": ndvi_stats["Mean NDVI"],
            "Sum_NDVI": ndvi_stats["Total NDVI"],
            "Min_NDVI": ndvi_stats["Min NDVI"],
            "Max_NDVI": ndvi_stats["Max NDVI"],
            "Median_NDVI": ndvi_stats["Median NDVI"],
            "StdDev_NDVI": ndvi_stats["Std Dev NDVI"],
            # LST Day
            "Mean_LST_Day": lstd_stats["Mean_LST_Day"] if lstd_stats else None,
            # Derived ratio
            "ndvi_lst_ratio": (
                ndvi_stats["Median NDVI"] / lst_stats["Mean LST (°K)"]
                if lst_stats.get("Mean LST (°K)") else None
            ),
            # GHSL building features
            "Mean_BUILT_S":   ghsl_stats.get("Mean_BUILT_S")   if ghsl_stats else None,
            "Median_BUILT_S": ghsl_stats.get("Median_BUILT_S") if ghsl_stats else None,
            "StdDev_BUILT_S": ghsl_stats.get("StdDev_BUILT_S") if ghsl_stats else None,
            "StdDev_BUILT_V": ghsl_stats.get("StdDev_BUILT_V") if ghsl_stats else None,
            # Anomalies
            "NTL_anom":      anom_stats.get("NTL_anom")      if anom_stats else None,
            "NDVI_anom":     anom_stats.get("NDVI_anom")     if anom_stats else None,
            "LSTN_anom":     anom_stats.get("LSTN_anom")     if anom_stats else None,
            "LST_Day_anom":  anom_stats.get("LST_Day_anom")  if anom_stats else None,
            "NTL_anom_lag1": anom_stats.get("NTL_anom_lag1") if anom_stats else None,
        }
        return (feature_row, pop_stats["Total Population"])
    except:
        return None


# --------------------------------------------------------------------------------------


def _fetch_stats_for_regions(regions, country, year, get_stats_func, max_workers=GEE_REGION_WORKERS):
    """Fetch GEE stats for a list of regions concurrently. Returns {region: result}."""
    results = {}
    if max_workers <= 1:
        for region in regions:
            try:
                results[region] = get_stats_func(region, country, year)
            except Exception:
                results[region] = None
            if GEE_REGION_DELAY_SECONDS:
                time.sleep(GEE_REGION_DELAY_SECONDS)
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(get_stats_func, r, country, year): r for r in regions}
        for future in future_map:
            region = future_map[future]
            try:
                results[region] = future.result()
            except Exception:
                results[region] = None
    return results


def show_helper_tab(df_actual):
    active_buffer = st.session_state.get("selected_buffer")
    if active_buffer:
        st.caption(f"Active buffer: {active_buffer}")
    st.title("Countrywide MPI Prediction")

    country = st.selectbox(
        "Select a Country", get_country_list(), key="country_pred_new"
    )

    # TR grouping toggle only when Turkey and asset available
    use_tr_asset = False
    if country == "Turkey" and TR_FC is not None:
        use_tr_asset = st.checkbox(
            "Use Turkish TR grouped regions (asset)", value=True, key="use_tr_asset"
        )

    level_choice = st.selectbox(
        "Select Region Level",
        ["Level 1 (Governorate)", "Level 2 (District)", "Both"],
        key="level_choice",
    )

    selected_years = st.multiselect(
        "Select Years to Predict MPI for", list(range(2012, 2025)), default=[2024]
    )
    selected_year = st.selectbox(
        "Year to Display on Map", selected_years, key="display_year"
    )

    model_choice = st.selectbox(
        "Select a model for prediction:",
        [
            "ML",
            "DNN",
            "DNN+RF",
            "DNN+XGBoost",
            "DNN+LightGBM",
            "DNN+KNN",
            "XGB-Quantile",
            "Stacked",
        ],
        key="model_choice_new",
    )

    alpha = None
    if model_choice in ["DNN+RF", "DNN+XGBoost", "DNN+LightGBM", "DNN+KNN"]:
        alpha = st.slider(
            "Ensemble Weight (DNN Contribution)", 0.0, 1.0, 0.4, key="alpha_new"
        )

    if model_choice == "Stacked":
        stacked_local_files = [
            "models/stacked/metadata.json",
            "models/stacked/meta_model.pkl",
            "models/stacked/scaler.pkl",
        ]
        stacked_global_files = REQUIRED_PRETRAINED_FILES["Stacked"]
        local_available = all(os.path.exists(path) for path in stacked_local_files)
        pretrained_available = all(os.path.exists(path) for path in stacked_global_files)

        if pretrained_available:
            use_pretrained_model = st.checkbox(
                " Use Pre-trained Model", value=True, key="use_pretrained_model"
            )
        elif local_available:
            use_pretrained_model = False
            st.info("Using local stacked artifacts from models/stacked/.")
        else:
            st.error(
                "Stacked artifacts not found. Expected files in models/stacked/ (or models/global/stacked/)."
            )
            return
    else:
        required_files = REQUIRED_PRETRAINED_FILES.get(model_choice, [])
        pretrained_available = all(os.path.exists(path) for path in required_files)

        if pretrained_available:
            use_pretrained_model = st.checkbox(
                " Use Pre-trained Model", value=True, key="use_pretrained_model"
            )
        else:
            use_pretrained_model = False
            st.info(
                f"🔧 Pre-trained model for '{model_choice}' not found. Please train your own model."
            )

    use_satellite = st.checkbox(
        "🛰️ Show Satellite Imagery", value=True, key="toggle_satellite_pred"
    )
    fill_opacity = st.slider(
        "🔆 Adjust MPI Layer Transparency", 0.0, 1.0, 0.5, step=0.05
    )
    show_actual = st.checkbox(
        "📌 Use Actual MPI for Coloring Map (Governorates only)", value=False
    )
    display_sev_pov = st.checkbox(
        "📊 Use Severe Poverty % for Coloring Map (Governorates only)", value=False
    )

    district_range = None
    if level_choice in ["Level 2 (District)", "Both"]:
        all_dist_regions = get_region_list_lvl2(country)
        if all_dist_regions:
            district_range = st.slider(
                "Select range of district indices (1-indexed)",
                min_value=1,
                max_value=len(all_dist_regions),
                value=(1, min(10, len(all_dist_regions))),
                key="district_range",
            )
            selected_districts = all_dist_regions[
                district_range[0] - 1 : district_range[1]
            ]
        else:
            st.error("No district data found for the selected country.")

    # include TR toggle in cache key
    cache_key = f"{PREDICTION_CACHE_VERSION}_{country}_{'_'.join(map(str, selected_years))}_{model_choice}_{alpha}_{level_choice}_TR{int(bool(use_tr_asset))}"
    if "mpi_cache" not in st.session_state:
        st.session_state["mpi_cache"] = {}
    if "mpi_feature_cache" not in st.session_state:
        st.session_state["mpi_feature_cache"] = {}

    if cache_key not in st.session_state["mpi_cache"]:
        if st.button("🌐 Generate Predictions"):
            with st.spinner("Fetching data and generating predictions..."):
                all_predictions = []
                debug_rows = []
                # Load models once
                dnn_model = ml_model = base_model = scaler = stacked_artifacts = None
                quantile_models = None
                if model_choice == "DNN":
                    dnn_model = load_dnn_model(use_pretrained_model)
                    scaler = load_dnn_scaler(use_pretrained_model)
                elif model_choice == "ML":
                    ml_model = load_ml_model(use_pretrained_model)
                    scaler = load_ml_scaler(use_pretrained_model)
                elif model_choice == "XGB-Quantile":
                    quantile_models = load_quantile_models(use_pretrained_model)
                    scaler = load_quantile_scaler(use_pretrained_model)
                elif model_choice == "Stacked":
                    stacked_artifacts = load_stacked_artifacts(use_pretrained_model)
                else:
                    dnn_model, base_model = load_ensemble_models(
                        model_choice, use_pretrained_model
                    )
                    scaler = load_ensemble_scaler(use_pretrained_model)

                for year in selected_years:
                    if level_choice != "Both":
                        if level_choice == "Level 1 (Governorate)":
                            regions = get_region_list(country, use_tr_asset)
                            get_stats_func = lambda r, c, y: get_all_stats_parallel(
                                r, c, y, use_tr_asset
                            )
                            get_geom_func = lambda c, r: get_region_geometry(
                                c, r, use_tr_asset
                            )
                            # label is TR code when TR mode, ADM1 name otherwise
                            get_name = lambda c, r: r
                        else:
                            regions = (
                                selected_districts
                                if district_range
                                else get_region_list_lvl2(country)
                            )
                            get_stats_func = get_all_stats_parallel_lvl2
                            get_geom_func = get_region_geometry_lvl2
                            get_name = get_district_name_from_adm2code

                        stats_cache = _fetch_stats_for_regions(regions, country, year, get_stats_func)
                        for region in regions:
                                name = get_name(country, region)
                                result = stats_cache.get(region)
                                if not result:
                                    continue

                                feature_row, weight = result
                                debug_row = {
                                    "Country": country,
                                    "Region": name,
                                    "Year": year,
                                    **feature_row,
                                }
                                if level_choice == "Level 2 (District)":
                                    debug_row["ADM2_CODE"] = region
                                debug_rows.append(debug_row)

                                df_input = pd.DataFrame([feature_row])
                                quant_pred = None

                                if model_choice == "DNN":
                                    pred = predict_dnn_fast(df_input, dnn_model, scaler)
                                elif model_choice == "ML":
                                    pred = predict_ml_fast(df_input, ml_model, scaler)
                                elif model_choice == "XGB-Quantile":
                                    quant_pred = predict_quantile_fast(
                                        df_input, quantile_models, scaler
                                    )
                                    pred = quant_pred["median"]
                                elif model_choice == "Stacked":
                                    pred = predict_stacked_fast(
                                        df_input, stacked_artifacts
                                    )
                                else:
                                    pred = predict_ensemble_fast(
                                        df_input, dnn_model, base_model, scaler, alpha
                                    )

                                if pred is not None:
                                    geom = get_geom_func(country, region)
                                    if (
                                        geom
                                        and geom.get("type") == "GeometryCollection"
                                    ):
                                        polys = [
                                            g
                                            for g in geom.get("geometries", [])
                                            if g.get("type")
                                            in ["Polygon", "MultiPolygon"]
                                        ]
                                        if not polys:
                                            continue
                                        geom = (
                                            {
                                                "type": "MultiPolygon",
                                                "coordinates": [
                                                    p["coordinates"] for p in polys
                                                ],
                                            }
                                            if len(polys) > 1
                                            else polys[0]
                                        )
                                    if geom is None:
                                        continue

                                    entry = {
                                        "Country": country,
                                        "Region": name,
                                        "Year": year,
                                        "Predicted MPI": float(pred[0]),
                                        "Weight": weight,
                                        "Geometry": geom,
                                    }
                                    if quant_pred is not None:
                                        entry["MPI Lower 90%"] = float(quant_pred["lower"][0])
                                        entry["MPI Upper 90%"] = float(quant_pred["upper"][0])
                                        entry["MPI Interval Width"] = float(
                                            quant_pred["width"][0]
                                        )
                                    if level_choice == "Level 2 (District)":
                                        entry["ADM2_CODE"] = region
                                    all_predictions.append(entry)

                    else:
                        # Governorates/TR
                        gov_regions = get_region_list(country, use_tr_asset)
                        # Districts
                        dist_regions = (
                            selected_districts
                            if district_range
                            else get_region_list_lvl2(country)
                        )

                        for regions, get_stats_func, get_geom_func, get_name, is_district_level in [
                            (
                                gov_regions,
                                (
                                    lambda r, c, y: get_all_stats_parallel(
                                        r, c, y, use_tr_asset
                                    )
                                ),
                                (lambda c, r: get_region_geometry(c, r, use_tr_asset)),
                                (lambda c, r: r),
                                False,
                            ),
                            (
                                dist_regions,
                                get_all_stats_parallel_lvl2,
                                get_region_geometry_lvl2,
                                get_district_name_from_adm2code,
                                True,
                            ),
                        ]:
                            stats_cache = _fetch_stats_for_regions(regions, country, year, get_stats_func)
                            for region in regions:
                                    name = get_name(country, region)
                                    result = stats_cache.get(region)
                                    if not result:
                                        continue
                                    feature_row, weight = result
                                    debug_row = {
                                        "Country": country,
                                        "Region": name,
                                        "Year": year,
                                        **feature_row,
                                    }
                                    if is_district_level:
                                        debug_row["ADM2_CODE"] = region
                                    debug_rows.append(debug_row)

                                    df_input = pd.DataFrame([feature_row])
                                    quant_pred = None

                                    if model_choice == "DNN":
                                        pred = predict_dnn_fast(
                                            df_input, dnn_model, scaler
                                        )
                                    elif model_choice == "ML":
                                        pred = predict_ml_fast(
                                            df_input, ml_model, scaler
                                        )
                                    elif model_choice == "XGB-Quantile":
                                        quant_pred = predict_quantile_fast(
                                            df_input, quantile_models, scaler
                                        )
                                        pred = quant_pred["median"]
                                    elif model_choice == "Stacked":
                                        pred = predict_stacked_fast(
                                            df_input, stacked_artifacts
                                        )
                                    else:
                                        pred = predict_ensemble_fast(
                                            df_input,
                                            dnn_model,
                                            base_model,
                                            scaler,
                                            alpha,
                                        )

                                    if pred is not None:
                                        geom = get_geom_func(country, region)
                                        if (
                                            geom
                                            and geom.get("type") == "GeometryCollection"
                                        ):
                                            polys = [
                                                g
                                                for g in geom.get("geometries", [])
                                                if g.get("type")
                                                in ["Polygon", "MultiPolygon"]
                                            ]
                                            if not polys:
                                                continue
                                            geom = (
                                                {
                                                    "type": "MultiPolygon",
                                                    "coordinates": [
                                                        p["coordinates"] for p in polys
                                                    ],
                                                }
                                                if len(polys) > 1
                                                else polys[0]
                                            )
                                        if geom is None:
                                            continue

                                        entry = {
                                            "Country": country,
                                            "Region": name,
                                            "Year": year,
                                            "Predicted MPI": float(pred[0]),
                                            "Weight": weight,
                                            "Geometry": geom,
                                        }
                                        if quant_pred is not None:
                                            entry["MPI Lower 90%"] = float(
                                                quant_pred["lower"][0]
                                            )
                                            entry["MPI Upper 90%"] = float(
                                                quant_pred["upper"][0]
                                            )
                                            entry["MPI Interval Width"] = float(
                                                quant_pred["width"][0]
                                            )

                                        if is_district_level:
                                            entry["ADM2_CODE"] = region

                                        all_predictions.append(entry)

                df_debug = pd.DataFrame(debug_rows)
                st.session_state["mpi_feature_cache"][cache_key] = df_debug

                if not df_debug.empty:
                    try:
                        duplicated_inputs = (
                            df_debug.groupby(["Country", "Region"])
                            .apply(
                                lambda group: group.drop(
                                    columns=["Country", "Region", "Year"]
                                )
                                .nunique()
                                .max()
                                == 1
                                and len(group) > 1
                            )
                            .reset_index(name="Identical Inputs Across Years")
                        )

                        duplicated_inputs = duplicated_inputs[
                            duplicated_inputs["Identical Inputs Across Years"] == True
                        ]

                        if not duplicated_inputs.empty:
                            st.warning(
                                "⚠️ Some regions have identical input features across different years. This likely caused identical predictions."
                            )
                            st.dataframe(duplicated_inputs)

                            st.download_button(
                                "📥 Download Debug Feature Vectors",
                                data=df_debug.to_csv(index=False).encode("utf-8"),
                                file_name="debug_features.csv",
                                mime="text/csv",
                            )
                    except Exception as e:
                        st.error(f"Debugging error: {e}")

                st.session_state["mpi_cache"][cache_key] = all_predictions

    if cache_key in st.session_state["mpi_cache"]:
        prediction_results = st.session_state["mpi_cache"][cache_key]
        if not prediction_results:
            st.error("No predictions were generated.")
            return

        df_pred = pd.DataFrame(prediction_results).drop(
            columns=["Geometry"], errors="ignore"
        )

        merged = pd.merge(
            df_pred,
            df_actual[["Country", "Region", "Year", "MPI"]],
            how="left",
            on=["Country", "Region", "Year"],
        )
        merged.rename(columns={"MPI": "Actual MPI"}, inplace=True)

        df_features = st.session_state.get("mpi_feature_cache", {}).get(cache_key)
        if df_features is not None and not df_features.empty:
            with st.expander("Fetched prediction features", expanded=False):
                st.dataframe(df_features, use_container_width=True)
                st.download_button(
                    "Download fetched features CSV",
                    data=df_features.to_csv(index=False).encode("utf-8"),
                    file_name=f"{country}_fetched_prediction_features.csv",
                    mime="text/csv",
                    key=f"download_features_{cache_key}",
                )

        if level_choice == "Level 1 (Governorate)":
            df = merged.rename(columns={"Region": "Governorate"})
            df["Predicted Severe Poverty %"] = df["Predicted MPI"].apply(
                compute_sev_pov
            )
            st.subheader("📊 MPI Predictions by Governorate / TR Region")
            st.dataframe(df.drop(columns=["Weight"], errors="ignore"))
            filtered = df[df["Year"] == selected_year]
            if not filtered.empty:
                weighted_avg = np.average(
                    filtered["Predicted MPI"], weights=filtered["Weight"]
                )
                st.metric("🏛️ Countrywide Weighted MPI", round(weighted_avg, 5))
            csv = (
                df.drop(columns=["Weight"], errors="ignore")
                .to_csv(index=False)
                .encode("utf-8")
            )

        elif level_choice == "Level 2 (District)":
            df = merged.rename(columns={"Region": "District"})
            adm2_to_gov = get_adm2code_to_governorate_map(country)
            df["Governorate"] = df["ADM2_CODE"].map(lambda c: adm2_to_gov.get(c, {}).get("adm1_name"))
            df["ADM1_CODE"]   = df["ADM2_CODE"].map(lambda c: adm2_to_gov.get(c, {}).get("adm1_code"))
            df["Predicted Severe Poverty %"] = df["Predicted MPI"].apply(
                compute_sev_pov
            )
            st.subheader("📊 MPI Predictions by District")
            st.dataframe(df.drop(columns=["Weight", "Actual MPI"], errors="ignore"))
            filtered = df[df["Year"] == selected_year]
            if not filtered.empty:
                weighted_avg = np.average(
                    filtered["Predicted MPI"], weights=filtered["Weight"]
                )
                st.metric("🏛️ Countrywide Weighted MPI", round(weighted_avg, 5))
            csv = (
                df.drop(columns=["Weight"], errors="ignore")
                .to_csv(index=False)
                .encode("utf-8")
            )

        else:  # Both levels
            merged["Level"] = merged["ADM2_CODE"].apply(
                lambda x: "District" if pd.notnull(x) else "Governorate"
            )

            df_lvl1 = (
                merged[merged["Level"] == "Governorate"].copy().drop(columns=["Level"])
            )
            df_lvl2 = (
                merged[merged["Level"] == "District"].copy().drop(columns=["Level"])
            )

            df_lvl1 = df_lvl1.rename(columns={"Region": "Governorate"})
            df_lvl1["Predicted Severe Poverty %"] = df_lvl1["Predicted MPI"].apply(
                compute_sev_pov
            )

            df_lvl2 = df_lvl2.rename(columns={"Region": "District"})
            adm2_to_gov = get_adm2code_to_governorate_map(country)
            df_lvl2["Governorate"] = df_lvl2["ADM2_CODE"].map(lambda c: adm2_to_gov.get(c, {}).get("adm1_name"))
            df_lvl2["ADM1_CODE"]   = df_lvl2["ADM2_CODE"].map(lambda c: adm2_to_gov.get(c, {}).get("adm1_code"))

            df_lvl2["Predicted Severe Poverty %"] = df_lvl2["Predicted MPI"].apply(
                compute_sev_pov
            )

            st.subheader("📊 MPI Predictions by Governorate / TR Region")
            st.dataframe(df_lvl1.drop(columns=["Weight", "ADM2_CODE"], errors="ignore"))

            st.subheader("📊 MPI Predictions by District")
            st.dataframe(
                df_lvl2.drop(columns=["Weight", "Actual MPI"], errors="ignore")
            )

            filt1 = df_lvl1[df_lvl1["Year"] == selected_year]
            if not filt1.empty:
                w_mpi = np.average(filt1["Predicted MPI"], weights=filt1["Weight"])
                st.metric("🏛️ Countrywide Weighted MPI (Gov Level)", round(w_mpi, 5))

            # Two separate CSVs in Both mode
            gov_csv = (
                df_lvl1.drop(columns=["Weight", "ADM2_CODE"], errors="ignore")
                .to_csv(index=False)
                .encode("utf-8")
            )
            dist_csv = (
                df_lvl2.drop(columns=["Weight", "Actual MPI"], errors="ignore")
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                label="📥 Download Governorate/TR Predictions (CSV)",
                data=gov_csv,
                file_name=f"{country}_Governorate_MPI.csv",
                mime="text/csv",
            )
            st.download_button(
                label="📥 Download District Predictions (CSV)",
                data=dist_csv,
                file_name=f"{country}_District_MPI.csv",
                mime="text/csv",
            )

        if level_choice != "Both":
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"{country}_MPI_Predictions.csv",
                mime="text/csv",
            )

        if level_choice == "Both":
            map_level_choice = st.radio(
                "🗺️ Choose which level to show on the map:",
                ["Governorates", "Districts"],
                index=0,
                key="map_level_choice",
            )

        # Map data for selected year
        all_year = [d for d in prediction_results if d["Year"] == selected_year]

        if level_choice == "Both":
            if map_level_choice == "Governorates":
                selected_year_data = [d for d in all_year if "ADM2_CODE" not in d]
            else:
                selected_year_data = [d for d in all_year if "ADM2_CODE" in d]
        else:
            selected_year_data = all_year

        # For pretty tooltips on TR asset
        TR_ADM1_MAP = (
            tr_code_to_adm1_list() if (country == "Turkey" and use_tr_asset) else {}
        )
        has_quantile_interval = any("MPI Lower 90%" in d for d in prediction_results)

        geojson_features = []
        for d in selected_year_data:
            is_governorate = "ADM2_CODE" not in d

            # Actual only for governorates/TR
            actual_val = (
                df_actual[
                    (df_actual["Country"] == d["Country"])
                    & (df_actual["Region"] == d["Region"])
                    & (df_actual["Year"] == d["Year"])
                ]["MPI"]
                if is_governorate
                else pd.Series([])
            )

            # Decide coloring value
            if display_sev_pov and is_governorate:
                value = round(compute_sev_pov(d["Predicted MPI"]), 5)
            elif show_actual and is_governorate:
                if actual_val.empty:
                    continue
                value = float(actual_val.values[0])
            else:
                value = round(d["Predicted MPI"], 5)

            pred_pov = (
                round(compute_sev_pov(d["Predicted MPI"]), 5)
                if d["Predicted MPI"] is not None
                else None
            )

            props = {
                "Governorate": d["Region"],
                "MPI": round(d["Predicted MPI"], 5),
                "Actual MPI": (
                    float(actual_val.values[0])
                    if is_governorate and not actual_val.empty
                    else None
                ),
                "Predicted Severe Poverty": pred_pov,
                "Year": d["Year"],
                "Value to Color": value,
            }
            if has_quantile_interval:
                lower_90 = d.get("MPI Lower 90%")
                upper_90 = d.get("MPI Upper 90%")
                interval_width = d.get("MPI Interval Width")
                props["MPI Lower 90%"] = (
                    round(lower_90, 5) if lower_90 is not None else None
                )
                props["MPI Upper 90%"] = (
                    round(upper_90, 5) if upper_90 is not None else None
                )
                props["MPI Interval Width"] = (
                    round(interval_width, 5) if interval_width is not None else None
                )

            # Add ADM1 members when TR grouped
            if country == "Turkey" and use_tr_asset and is_governorate:
                props["ADM1 list"] = TR_ADM1_MAP.get(d["Region"])

            geojson_features.append(
                {
                    "type": "Feature",
                    "geometry": d["Geometry"],
                    "properties": props,
                }
            )

        geojson = {"type": "FeatureCollection", "features": geojson_features}

        center = get_country_center(country)
        values = [
            f["properties"]["Value to Color"]
            for f in geojson["features"]
            if f["properties"]["Value to Color"] is not None
        ]

        if not values:
            st.warning("⚠️ No data available to render map for the selected settings.")
            return

        colormap = cm.linear.YlOrRd_09.scale(min(values), max(values))
        if display_sev_pov and (level_choice != "Level 2 (District)"):
            colormap.caption = "Severe Poverty %"
        else:
            if show_actual and (level_choice != "Level 2 (District)"):
                colormap.caption = "Actual MPI Value"
            else:
                colormap.caption = "Predicted MPI Value"

        tiles = (
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            if use_satellite
            else "OpenStreetMap"
        )
        attr = "Esri World Imagery" if use_satellite else "OpenStreetMap"

        m = folium.Map(
            location=[center[1], center[0]], zoom_start=6, tiles=tiles, attr=attr
        )

        if use_satellite:
            folium.TileLayer(
                tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri Boundaries & Labels",
                name="Labels & Boundaries",
                overlay=True,
                control=False,
            ).add_to(m)

        admin_alias = (
            "TR Region"
            if (country == "Turkey" and use_tr_asset)
            else (
                "District"
                if (
                    level_choice == "Level 2 (District)"
                    or (
                        level_choice == "Both"
                        and "ADM2_CODE" in next(iter(prediction_results), {})
                    )
                )
                else "Governorate"
            )
        )

        # Tooltip fields/aliases (insert ADM1 list when TR)
        tooltip_fields = [
            "Governorate",
            "Year",
            "MPI",
            "Actual MPI",
            "Predicted Severe Poverty",
        ]
        tooltip_aliases = [
            admin_alias,
            "Year",
            "Predicted MPI",
            "Actual MPI",
            "Predicted Severe Poverty %",
        ]
        if has_quantile_interval:
            tooltip_fields.extend(
                ["MPI Lower 90%", "MPI Upper 90%", "MPI Interval Width"]
            )
            tooltip_aliases.extend(
                ["Predicted MPI (P05)", "Predicted MPI (P95)", "Prediction Interval Width"]
            )
        if country == "Turkey" and use_tr_asset:
            tooltip_fields.insert(1, "ADM1 list")
            tooltip_aliases.insert(1, "ADM1 members")

        folium.GeoJson(
            geojson,
            style_function=lambda feature: {
                "fillColor": colormap(feature["properties"]["Value to Color"]),
                "color": "black",
                "weight": 1,
                "fillOpacity": fill_opacity,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields, aliases=tooltip_aliases
            ),
        ).add_to(m)

        colormap.add_to(m)
        folium_static(m, width=750, height=550)


