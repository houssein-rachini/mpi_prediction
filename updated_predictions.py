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
    load_scaler,
    preprocess_data,
    predict_dnn,
    predict_ml,
    predict_ensemble,
    plot_results,
)
from concurrent.futures import ThreadPoolExecutor
import branca.colormap as cm
import os
from math import ceil

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
}


# fct Used for calculating Severe Poverty Percentage based on predicted MPI
def severe_poverty_percentage(mpi):
    y = 0.04133 + 34.58 * mpi + 263 * (mpi**2) - 180.8 * (mpi**3)
    return min(100, max(0, y))


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
    #    "Lebanon",
    #     "Jordan",
    #    "Turkmenistan",
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
    return (
        fao_gaul_lvl2.filter(ee.Filter.eq("ADM0_NAME", country))
        .aggregate_array("ADM2_NAME")
        .distinct()
        .getInfo()
    )


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
def get_region_geometry_lvl2(country, region):
    filtered = fao_gaul_lvl2.filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", country),
            ee.Filter.eq("ADM2_NAME", region),
        )
    )
    return filtered.geometry().getInfo()


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


def show_helper_tab(df_actual):

    # Inputs
    st.title("🌍 Countrywide MPI Prediction")
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
        ["ML", "DNN", "DNN+RF", "DNN+XGBoost"],
        key="model_choice_new",
    )
    alpha = None
    if model_choice in ["DNN+RF", "DNN+XGBoost"]:
        alpha = st.slider(
            "Ensemble Weight (DNN Contribution)", 0.0, 1.0, 0.4, key="alpha_new"
        )
    required_files = REQUIRED_PRETRAINED_FILES.get(model_choice, [])
    pretrained_available = all(os.path.exists(p) for p in required_files)
    if pretrained_available:
        use_pretrained_model = st.checkbox(
            "Use Pre-trained Model", True, key="use_pretrained_model"
        )
    else:
        use_pretrained_model = False
        st.info(
            f"🔧 Pre-trained model for '{model_choice}' not found. Please train your own model."
        )
    use_satellite = st.checkbox(
        "🛰️ Show Satellite Imagery", True, key="toggle_satellite_pred"
    )
    fill_opacity = st.slider("🔆 MPI Layer Transparency", 0.0, 1.0, 0.5, step=0.05)
    show_actual = st.checkbox("📌 Show Actual MPI on Map (if available)", False)

    # District range
    district_range = None
    if level_choice in ["Level 2 (District)", "Both"]:
        all_dist = get_region_list_lvl2(country)
        if all_dist:
            district_range = st.slider(
                "Select district index range",
                1,
                len(all_dist),
                (1, min(10, len(all_dist))),
                key="district_range",
            )
            selected_districts = all_dist[district_range[0] - 1 : district_range[1]]
        else:
            st.error("No district data found for this country.")

    # Cache key
    cache_key = f"{country}_{'_'.join(map(str,selected_years))}_{model_choice}_{alpha}_{level_choice}"
    st.session_state.setdefault("mpi_cache", {})

    # Generate predictions
    if cache_key not in st.session_state["mpi_cache"]:
        if st.button("🌐 Generate Predictions"):
            with st.spinner("Generating predictions..."):
                all_predictions = []
                for year in selected_years:
                    # Prepare region lists
                    gov_regs = get_region_list(country)
                    dist_regs = (
                        selected_districts
                        if district_range
                        else get_region_list_lvl2(country)
                    )

                    # Helper to process
                    def process(regs, stats_fn, geom_fn):
                        for batch in chunk_list(regs, batch_size):
                            for region in batch:
                                res = stats_fn(region, country, year)
                                if not res:
                                    continue
                                row, weight = res
                                df_in = pd.DataFrame([row])
                                if model_choice == "DNN":
                                    pred = predict_dnn(df_in, use_pretrained_model)
                                elif model_choice == "ML":
                                    pred = predict_ml(df_in, use_pretrained_model)
                                else:
                                    pred = predict_ensemble(
                                        df_in, model_choice, alpha, use_pretrained_model
                                    )
                                if pred is None:
                                    continue
                                geom = geom_fn(country, region)
                                if geom.get("type") == "GeometryCollection":
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
                                        "Region": region,
                                        "Year": year,
                                        "Predicted MPI": float(pred[0]),
                                        "Weight": weight,
                                        "Geometry": geom,
                                    }
                                )

                    # Run for governorates and/or districts
                    if level_choice in ["Level 1 (Governorate)", "Both"]:
                        process(gov_regs, get_all_stats_parallel, get_region_geometry)
                    if level_choice in ["Level 2 (District)", "Both"]:
                        process(
                            dist_regs,
                            get_all_stats_parallel_lvl2,
                            get_region_geometry_lvl2,
                        )

                st.session_state["mpi_cache"][cache_key] = all_predictions

    # Display results
    if cache_key in st.session_state["mpi_cache"]:
        preds = st.session_state["mpi_cache"][cache_key]
        if not preds:
            st.error("No predictions generated.")
            return
        df_pred = pd.DataFrame(preds)
        merged = pd.merge(
            df_pred,
            df_actual[["Country", "Region", "Year", "MPI"]],
            on=["Country", "Region", "Year"],
            how="left",
        ).rename(columns={"MPI": "Actual MPI"})
        # Compute severe poverty
        merged["Predicted Severe Poverty"] = merged["Predicted MPI"].apply(
            severe_poverty_percentage
        )
        merged["Actual Severe Poverty"] = merged["Actual MPI"].apply(
            lambda x: severe_poverty_percentage(x) if pd.notna(x) else None
        )

        # Table and metrics
        def show_table(df_t, label):
            st.subheader(f"📊 Predictions by {label}")
            st.dataframe(df_t.drop(columns=["Weight"], errors="ignore"))
            filt = df_t[df_t["Year"] == selected_year]
            if not filt.empty:
                st.metric(
                    "🏛️ Weighted Pred MPI",
                    round(np.average(filt["Predicted MPI"], weights=filt["Weight"]), 5),
                )
                if filt["Actual MPI"].notna().any():
                    st.metric(
                        "🏛️ Weighted Act MPI",
                        round(
                            np.average(
                                filt.loc[filt["Actual MPI"].notna(), "Actual MPI"],
                                weights=filt.loc[filt["Actual MPI"].notna(), "Weight"],
                            ),
                            5,
                        ),
                    )
                st.metric(
                    "⚠️ Weighted Pred Severe Pov",
                    round(
                        np.average(
                            filt["Predicted Severe Poverty"], weights=filt["Weight"]
                        ),
                        5,
                    ),
                )
                if filt["Actual Severe Poverty"].notna().any():
                    st.metric(
                        "⚠️ Weighted Act Severe Pov",
                        round(
                            np.average(
                                filt.loc[
                                    filt["Actual Severe Poverty"].notna(),
                                    "Actual Severe Poverty",
                                ],
                                weights=filt.loc[
                                    filt["Actual Severe Poverty"].notna(), "Weight"
                                ],
                            ),
                            5,
                        ),
                    )

        if level_choice == "Level 1 (Governorate)":
            df1 = merged.rename(columns={"Region": "Governorate"})
            show_table(df1, "Governorate")
            csv = df1.to_csv(index=False).encode()
        elif level_choice == "Level 2 (District)":
            df2 = merged.rename(columns={"Region": "District"})
            show_table(df2, "District")
            csv = df2.to_csv(index=False).encode()
        else:
            df1 = merged[merged["Region"].isin(get_region_list(country))].rename(
                columns={"Region": "Governorate"}
            )
            df2 = merged[merged["Region"].isin(get_region_list_lvl2(country))].rename(
                columns={"Region": "District"}
            )
            show_table(df1, "Governorate")
            show_table(df2, "District")
            csv = pd.concat([df1, df2]).to_csv(index=False).encode()

        st.download_button(
            "📥 Download CSV", data=csv, file_name=f"{country}_MPI.csv", mime="text/csv"
        )

        selected_year_data = [d for d in preds if d["Year"] == selected_year]
        geojson_features = []

        for d in selected_year_data:
            # Always include Actual MPI in props, but only use it for color when show_actual is True
            if show_actual and pd.notna(d.get("Actual MPI")):
                color_val = float(d["Actual MPI"])
            else:
                color_val = float(d["Predicted MPI"])

            geojson_features.append(
                {
                    "type": "Feature",
                    "geometry": d["Geometry"],
                    "properties": {
                        "Governorate": d["Region"],
                        "Predicted MPI": round(d["Predicted MPI"], 5),
                        "Actual MPI": d.get("Actual MPI"),
                        "Predicted Severe Poverty": severe_poverty_percentage(
                            d["Predicted MPI"]
                        ),
                        "Actual Severe Poverty": (
                            severe_poverty_percentage(d["Actual MPI"])
                            if pd.notna(d.get("Actual MPI"))
                            else None
                        ),
                        "Year": d["Year"],
                        "Value to Color": color_val,
                    },
                }
            )

        geojson = {"type": "FeatureCollection", "features": geojson_features}

        # Build colormap off whatever Value to Color holds
        values = [f["properties"]["Value to Color"] for f in geojson_features]
        cmap = cm.linear.YlOrRd_09.scale(min(values), max(values))
        cmap.caption = "MPI (Actual or Predicted)"

        # Render map
        ctr = get_country_center(country)
        m = folium.Map(
            location=[ctr[1], ctr[0]],
            zoom_start=6,
            tiles=("Esri World Imagery" if use_satellite else "OpenStreetMap"),
        )
        if use_satellite:
            folium.TileLayer(
                tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri Boundaries & Labels",
                overlay=True,
                control=False,
            ).add_to(m)

        folium.GeoJson(
            geojson,
            style_function=lambda feat: {
                "fillColor": cmap(feat["properties"]["Value to Color"]),
                "color": "black",
                "weight": 1,
                "fillOpacity": fill_opacity,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "Governorate",
                    "Predicted MPI",
                    "Actual MPI",
                    "Predicted Severe Poverty",
                    "Actual Severe Poverty",
                    "Year",
                ],
                aliases=[
                    "Region",
                    "Pred MPI",
                    "Act MPI",
                    "Pred Severe Pov",
                    "Act Severe Pov",
                    "Year",
                ],
            ),
        ).add_to(m)

        cmap.add_to(m)
        folium_static(m, width=750, height=550)
