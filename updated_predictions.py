import streamlit as st
import folium
import ee
import numpy as np
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
import xgboost as xgb
from predictions import (
    preprocess_data,
    plot_results,
    load_dnn_model,
    load_dnn_scaler,
    load_ml_model,
    load_ml_scaler,
    load_ensemble_models,
    load_ensemble_scaler,
    predict_dnn_fast,
    predict_ml_fast,
    predict_ensemble_fast,
)

from concurrent.futures import ThreadPoolExecutor
import branca.colormap as cm
import os
from math import ceil

batch_size = 10


batch_size = 10
from google.oauth2 import service_account

service_account_info = dict(st.secrets["google_ee"])  # No need for .to_json()

credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=["https://www.googleapis.com/auth/earthengine"]
)

ee.Initialize(credentials)

# Load FAO GAUL and WorldPop datasets
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


# Mapping of required files for each model
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
}


def compute_sev_pov(mpi):
    pov = 0.04133 + 34.58 * mpi + 263 * (mpi**2) - 180.8 * (mpi**3)
    return min(100, max(0, pov))


def compute_ndvi(image):
    ndvi = image.normalizedDifference(["sur_refl_b02", "sur_refl_b01"]).rename("NDVI")
    return ndvi.copyProperties(image, image.propertyNames())


ndvi_v2 = ndvi_v2.map(compute_ndvi)


def chunk_list(lst, size):
    return [lst[i : i + size] for i in range(0, len(lst), size)]


@st.cache_resource
def get_country_list():
    c_list = fao_gaul.aggregate_array("ADM0_NAME").distinct().getInfo()
    # filter the countries for Morocco, Tunisia, Mauritania, Iraq, Syrian Arab Republic, Azerbaijan, Afghanistan, Pakistan, Uzbekistan, Tajikistan,Kyrgyzstan:
    # c_list = [
    #     "Morocco",
    #     "Tunisia",
    #     "Mauritania",
    #     "Iraq",
    #     "Syrian Arab Republic",
    #     "Azerbaijan",
    #     "Afghanistan",
    #     "Pakistan",
    #     "Uzbekistan",
    #     "Tajikistan",
    #     "Kyrgyzstan",
    #     "Egypt",
    #     "Turkey",
    #     "Bosnia and Herzegovina",
    #     "Jordan",
    #     "Lebanon",
    #     "Turkmenistan",
    #     "Montenegro",
    # ]
    c_list.sort()
    return c_list


@st.cache_resource
def get_region_list(country):
    return (
        fao_gaul.filter(ee.Filter.eq("ADM0_NAME", country))
        .aggregate_array("ADM1_NAME")
        .distinct()
        .getInfo()
    )


@st.cache_resource
def get_region_list_lvl2(country):
    features = fao_gaul_lvl2.filter(ee.Filter.eq("ADM0_NAME", country)).map(
        lambda f: f.set("uid", f.id())
    )
    return features.aggregate_array("uid").getInfo()


# using batch prediction
def get_all_stats_parallel(region, country, selected_year):
    try:
        region_geom = get_region_geometry(country, region)
        pop_stats = get_cached_population_stats(region_geom, selected_year)
        gpp_stats = get_cached_gpp_stats(region_geom, selected_year)
        lst_stats = get_cached_lst_stats(region_geom, selected_year)
        ntl_stats = get_cached_ntl_stats(region_geom, selected_year)
        ndvi_stats = get_cached_ndvi_stats(region_geom, selected_year)
        if not all([pop_stats, gpp_stats, lst_stats, ntl_stats, ndvi_stats]):
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
        }
        return (feature_row, pop_stats["Total Population"])
    except:
        return None


def get_all_stats_parallel_lvl2(region, country, selected_year):
    try:
        region_geom = get_region_geometry_lvl2(country, region)
        pop_stats = get_cached_population_stats(region_geom, selected_year)
        gpp_stats = get_cached_gpp_stats(region_geom, selected_year)
        lst_stats = get_cached_lst_stats(region_geom, selected_year)
        ntl_stats = get_cached_ntl_stats(region_geom, selected_year)
        ndvi_stats = get_cached_ndvi_stats(region_geom, selected_year)
        if not all([pop_stats, gpp_stats, lst_stats, ntl_stats, ndvi_stats]):
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
        }
        return (feature_row, pop_stats["Total Population"])
    except:
        return None


@st.cache_resource
def get_region_geometry(country, region):
    filtered = fao_gaul.filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", country),
            ee.Filter.eq("ADM1_NAME", region),
        )
    )
    return filtered.geometry().getInfo()


@st.cache_resource
def get_region_geometry_lvl2(country, uid):
    feature = fao_gaul_lvl2.filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", country), ee.Filter.eq("system:index", uid)
        )
    ).first()
    return feature.geometry().getInfo()


@st.cache_resource
def get_district_name_from_uid(country, uid):
    feature = fao_gaul_lvl2.filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", country), ee.Filter.eq("system:index", uid)
        )
    ).first()
    return feature.get("ADM2_NAME").getInfo()


def interpolate_population(region_geom, selected_year):
    if selected_year <= 2020:
        # Use real data if available
        start_date = ee.Date.fromYMD(selected_year, 1, 1)
        end_date = ee.Date.fromYMD(selected_year, 12, 31)

        pop_image = worldpop.filterDate(start_date, end_date).mean()
        stats = pop_image.reduceRegion(
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
        # Extrapolate from historical stats
        years = list(range(2012, 2021))
        props = ["mean", "sum", "min", "max", "median", "stdDev"]
        data = {prop: [] for prop in props}

        for year in years:
            date_start = ee.Date.fromYMD(year, 1, 1)
            date_end = ee.Date.fromYMD(year, 12, 31)
            image = worldpop.filterDate(date_start, date_end).mean()
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
            )
            info = stats.getInfo()
            if "population_mean" in info:
                data["mean"].append(info["population_mean"])
                data["sum"].append(info["population_sum"])
                data["min"].append(info["population_min"])
                data["max"].append(info["population_max"])
                data["median"].append(info["population_median"])
                data["stdDev"].append(info["population_stdDev"])
            else:
                for key in data:
                    data[key].append(None)

        # Now extrapolate each

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

    gpp_image = modis_gpp.filterDate(start_date, end_date).mean()

    stats = gpp_image.reduceRegion(
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

    # Compute area of the region in m²
    region_area = ee.Geometry(region_geom).area()
    total_gpp = ee.Number(stats["Gpp_sum"])
    mean_gpp = total_gpp.divide(region_area)

    return {
        "Mean GPP": round(mean_gpp.getInfo(), 6),
        "Min GPP": round(stats["Gpp_min"], 5),
        "Max GPP": round(stats["Gpp_max"], 5),
        "Median GPP": round(stats["Gpp_median"], 5),
        "Std Dev GPP": round(stats["Gpp_stdDev"], 5),
        "Total GPP": round(stats["Gpp_sum"], 5),
    }


def compute_lst_stats(region_geom, selected_year):
    start_date = ee.Date.fromYMD(selected_year, 1, 1)
    end_date = ee.Date.fromYMD(selected_year, 12, 31)

    lst_image = viirs_lst.filterDate(start_date, end_date).mean()

    stats = lst_image.reduceRegion(
        reducer=ee.Reducer.mean()
        .combine(ee.Reducer.minMax(), "", True)
        .combine(ee.Reducer.median(), "", True)
        .combine(ee.Reducer.stdDev(), "", True)
        .combine(ee.Reducer.sum(), "", True),
        geometry=ee.Geometry(region_geom),
        scale=1000,
        bestEffort=True,
    ).getInfo()

    if "LST_1KM_mean" not in stats:
        return None

    return {
        "Mean LST (°K)": round(stats["LST_1KM_mean"], 5),
        "Min LST (°K)": round(stats["LST_1KM_min"], 5),
        "Max LST (°K)": round(stats["LST_1KM_max"], 5),
        "Median LST (°K)": round(stats["LST_1KM_median"], 5),
        "Std Dev LST": round(stats["LST_1KM_stdDev"], 5),
        "Total LST": round(stats["LST_1KM_sum"], 5),
    }


def compute_ntl_stats(region_geom, selected_year):
    start_date = ee.Date.fromYMD(selected_year, 1, 1)
    end_date = ee.Date.fromYMD(selected_year, 12, 31)

    # Filter and average the NTL values from VIIRS
    ntl_image = viirs_ntl.filterDate(start_date, end_date).mean()

    # Compute regional statistics
    stats = ntl_image.reduceRegion(
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

    # Filter and average the NDVI values
    # ndvi_image = ndvi_collection.filterDate(start_date, end_date).mean()
    ndvi_image = ndvi_v2.filterDate(start_date, end_date).mean()
    # Compute regional statistics
    stats = ndvi_image.reduceRegion(
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


MODEL_PATHS = {
    "DNN": "trained_dnn_model.h5",
    "ML": "trained_ml_model.pkl",
    "DNN+RF": "trained_ensemble_rf_dnn_model.h5",
    "DNN+XGBoost": "trained_ensemble_xgb_dnn_model.h5",
}

SCALER_PATHS = {
    "DNN": "dnn_scaler.pkl",
    "ML": "ml_scaler.pkl",
    "Ensemble": "ensemble_scaler.pkl",
}


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


@st.cache_resource
def get_country_center(country):
    filtered = fao_gaul.filter(ee.Filter.eq("ADM0_NAME", country))
    coords = filtered.geometry().centroid().coordinates().getInfo()
    return coords if coords else [0, 0]


@st.cache_resource
def get_uid_to_governorate_map(country):
    features = fao_gaul_lvl2.filter(ee.Filter.eq("ADM0_NAME", country)).map(
        lambda f: f.set("uid", f.id())
    )
    uids = features.aggregate_array("uid").getInfo()
    govs = features.aggregate_array("ADM1_NAME").getInfo()
    return dict(zip(uids, govs))


def show_helper_tab(df_actual):
    st.title("🌍 Countrywide MPI Prediction ")

    country = st.selectbox(
        "Select a Country", get_country_list(), key="country_pred_new"
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
        ["ML", "DNN", "DNN+RF", "DNN+XGBoost", "DNN+KNN"],
        key="model_choice_new",
    )

    alpha = None
    if model_choice in ["DNN+RF", "DNN+XGBoost", "DNN+KNN"]:
        alpha = st.slider(
            "Ensemble Weight (DNN Contribution)", 0.0, 1.0, 0.4, key="alpha_new"
        )

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
    show_actual = st.checkbox("📌 Show Actual MPI on Map (if available)", value=False)
    display_sev_pov = st.checkbox("📊 Show Severe Poverty % on Map", value=False)

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

    cache_key = f"{country}_{'_'.join(map(str, selected_years))}_{model_choice}_{alpha}_{level_choice}"
    if "mpi_cache" not in st.session_state:
        st.session_state["mpi_cache"] = {}

    if cache_key not in st.session_state["mpi_cache"]:
        if st.button("🌐 Generate Predictions"):
            with st.spinner("Fetching data and generating predictions..."):
                all_predictions = []
                debug_rows = []
                # Load models once
                dnn_model = ml_model = base_model = scaler = None
                if model_choice == "DNN":
                    dnn_model = load_dnn_model(use_pretrained_model)
                    scaler = load_dnn_scaler(use_pretrained_model)
                elif model_choice == "ML":
                    ml_model = load_ml_model(use_pretrained_model)
                    scaler = load_ml_scaler(use_pretrained_model)
                else:
                    dnn_model, base_model = load_ensemble_models(
                        model_choice, use_pretrained_model
                    )
                    scaler = load_ensemble_scaler(use_pretrained_model)

                for year in selected_years:
                    if level_choice != "Both":
                        if level_choice == "Level 1 (Governorate)":
                            regions = get_region_list(country)
                            get_stats_func = get_all_stats_parallel
                            get_geom_func = get_region_geometry
                            get_name = lambda c, r: r
                        else:
                            regions = (
                                selected_districts
                                if district_range
                                else get_region_list_lvl2(country)
                            )
                            get_stats_func = get_all_stats_parallel_lvl2
                            get_geom_func = get_region_geometry_lvl2
                            get_name = get_district_name_from_uid

                        for region_batch in chunk_list(regions, batch_size):
                            for region in region_batch:
                                name = get_name(country, region)
                                result = get_stats_func(region, country, year)
                                if not result:
                                    continue

                                feature_row, weight = result
                                debug_rows.append(
                                    {
                                        "Country": country,
                                        "Region": name,
                                        "Year": year,
                                        **feature_row,
                                    }
                                )

                                df_input = pd.DataFrame([feature_row])

                                if model_choice == "DNN":
                                    pred = predict_dnn_fast(df_input, dnn_model, scaler)
                                elif model_choice == "ML":
                                    pred = predict_ml_fast(df_input, ml_model, scaler)
                                else:
                                    pred = predict_ensemble_fast(
                                        df_input, dnn_model, base_model, scaler, alpha
                                    )

                                if pred is not None:
                                    geom = get_geom_func(country, region)
                                    if geom["type"] == "GeometryCollection":
                                        polys = [
                                            g
                                            for g in geom["geometries"]
                                            if g["type"] in ["Polygon", "MultiPolygon"]
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

                                    all_predictions.append(
                                        {
                                            "Country": country,
                                            "Region": name,
                                            "Year": year,
                                            "Predicted MPI": float(pred[0]),
                                            "Weight": weight,
                                            "Geometry": geom,
                                        }
                                    )

                    else:
                        for regions, get_stats_func, get_geom_func, get_name in [
                            (
                                get_region_list(country),
                                get_all_stats_parallel,
                                get_region_geometry,
                                lambda c, r: r,
                            ),
                            (
                                (
                                    selected_districts
                                    if district_range
                                    else get_region_list_lvl2(country)
                                ),
                                get_all_stats_parallel_lvl2,
                                get_region_geometry_lvl2,
                                get_district_name_from_uid,
                            ),
                        ]:
                            for region_batch in chunk_list(regions, batch_size):
                                for region in region_batch:
                                    name = get_name(country, region)
                                    result = get_stats_func(region, country, year)
                                    if not result:
                                        continue
                                    feature_row, weight = result
                                    debug_rows.append(
                                        {
                                            "Country": country,
                                            "Region": name,
                                            "Year": year,
                                            **feature_row,
                                        }
                                    )

                                    df_input = pd.DataFrame([feature_row])

                                    if model_choice == "DNN":
                                        pred = predict_dnn_fast(
                                            df_input, dnn_model, scaler
                                        )
                                    elif model_choice == "ML":
                                        pred = predict_ml_fast(
                                            df_input, ml_model, scaler
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
                                        if geom["type"] == "GeometryCollection":
                                            polys = [
                                                g
                                                for g in geom["geometries"]
                                                if g["type"]
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

                                        all_predictions.append(
                                            {
                                                "Country": country,
                                                "Region": name,
                                                "District UID": region,
                                                "Year": year,
                                                "Predicted MPI": float(pred[0]),
                                                "Weight": weight,
                                                "Geometry": geom,
                                            }
                                        )

                df_debug = pd.DataFrame(debug_rows)

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

        if level_choice == "Level 1 (Governorate)":
            df = merged.rename(columns={"Region": "Governorate"})
            df["Predicted Severe Poverty %"] = df["Predicted MPI"].apply(
                compute_sev_pov
            )
            st.subheader("📊 MPI Predictions by Governorate")
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
            lvl1_set = set(get_region_list(country))
            merged["Level"] = merged["Region"].apply(
                lambda r: "Governorate" if r in lvl1_set else "District"
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
            uid_to_gov = get_uid_to_governorate_map(country)
            df_lvl2["Governorate"] = df_lvl2["District UID"].map(uid_to_gov.get)

            df_lvl2["Predicted Severe Poverty %"] = df_lvl2["Predicted MPI"].apply(
                compute_sev_pov
            )

            st.subheader("📊 MPI Predictions by Governorate")
            st.dataframe(df_lvl1.drop(columns=["Weight"], errors="ignore"))

            st.subheader("📊 MPI Predictions by District")
            st.dataframe(
                df_lvl2.drop(columns=["Weight", "Actual MPI"], errors="ignore")
            )

            filt1 = df_lvl1[df_lvl1["Year"] == selected_year]
            if not filt1.empty:
                w_mpi = np.average(filt1["Predicted MPI"], weights=filt1["Weight"])
                st.metric("🏛️ Countrywide Weighted MPI (Gov Level)", round(w_mpi, 5))

            csv = (
                pd.concat([df_lvl1, df_lvl2], ignore_index=True)
                .drop(columns=["Weight"], errors="ignore")
                .to_csv(index=False)
                .encode("utf-8")
            )

        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"{country}_MPI_Predictions.csv",
            mime="text/csv",
        )

        # Show map for selected year
        all_year = [d for d in prediction_results if d["Year"] == selected_year]

        if level_choice == "Both":
            gov_names = set(get_region_list(country))
            selected_year_data = [d for d in all_year if d["Region"] in gov_names]
        else:
            selected_year_data = all_year

        geojson_features = []

        for d in selected_year_data:
            actual_val = df_actual[
                (df_actual["Country"] == d["Country"])
                & (df_actual["Region"] == d["Region"])
                & (df_actual["Year"] == d["Year"])
            ]["MPI"]
            # if display_sev_pov:
            #     show_actual = False
            if show_actual:
                if actual_val.empty:
                    continue  # Skip if no actual MPI and showing actual
                value = float(actual_val.values[0])
            else:
                value = round(d["Predicted MPI"], 5)
            pred_pov = round(
                (
                    compute_sev_pov(d["Predicted MPI"])
                    if d["Predicted MPI"] is not None
                    else None
                ),
                5,
            )
            if display_sev_pov:
                value = pred_pov
            geojson_features.append(
                {
                    "type": "Feature",
                    "geometry": d["Geometry"],
                    "properties": {
                        "Governorate": d["Region"],
                        "MPI": round(d["Predicted MPI"], 5),
                        "Actual MPI": (
                            float(actual_val.values[0])
                            if not actual_val.empty
                            else None
                        ),
                        "Predicted Severe Poverty": pred_pov,
                        "Year": d["Year"],
                        "Value to Color": value,  # for colormap
                    },
                }
            )

        geojson = {
            "type": "FeatureCollection",
            "features": geojson_features,
        }

        center = get_country_center(country)
        values = [
            f["properties"]["Value to Color"]
            for f in geojson["features"]
            if f["properties"]["Value to Color"] is not None
        ]

        if not values:
            st.warning("⚠️ No Actual data available to render map. Use Predicted MPI.")
            return  # Exit early to avoid using undefined colormap

        colormap = cm.linear.YlOrRd_09.scale(min(values), max(values))
        if display_sev_pov:
            colormap.caption = "Severe Poverty %"
        else:
            if show_actual:
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
            "District" if level_choice == "Level 2 (District)" else "Governorate"
        )
        folium.GeoJson(
            geojson,
            style_function=lambda feature: {
                "fillColor": colormap(feature["properties"]["Value to Color"]),
                "color": "black",
                "weight": 1,
                "fillOpacity": fill_opacity,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "Governorate",
                    "Year",
                    "MPI",
                    "Actual MPI",
                    "Predicted Severe Poverty",
                ],
                aliases=[
                    admin_alias,
                    "Year",
                    "Predicted MPI",
                    "Actual MPI",
                    "Predicted Severe Poverty %",
                ],
            ),
        ).add_to(m)

        colormap.add_to(m)
        folium_static(m, width=750, height=550)
