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

from google.oauth2 import service_account

service_account_info = dict(st.secrets["google_ee"])  # No need for .to_json()

credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=["https://www.googleapis.com/auth/earthengine"]
)

ee.Initialize(credentials)

# Load FAO GAUL and WorldPop datasets
fao_gaul = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")
worldpop = ee.ImageCollection("WorldPop/GP/100m/pop").select("population")
modis_gpp = ee.ImageCollection("MODIS/061/MOD17A3HGF").select("Gpp")
viirs_lst = ee.ImageCollection("NASA/VIIRS/002/VNP21A1N").select("LST_1KM")
viirs_ntl = ee.ImageCollection("NOAA/VIIRS/001/VNP46A2").select(
    "Gap_Filled_DNB_BRDF_Corrected_NTL"
)
ndvi_collection = ee.ImageCollection("MODIS/MOD09GA_006_NDVI").select("NDVI")
ndvi_v2 = ee.ImageCollection("MODIS/061/MOD09A1")


def compute_ndvi(image):
    ndvi = image.normalizedDifference(["sur_refl_b02", "sur_refl_b01"]).rename("NDVI")
    return ndvi.copyProperties(image, image.propertyNames())


ndvi_v2 = ndvi_v2.map(compute_ndvi)


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


def predict_country_level_mpi(country, selected_year, model_choice, alpha=None):
    regions = get_region_list(country)
    predictions = []
    weights = []

    for region in regions:
        region_geom = get_region_geometry(country, region)
        pop_stats = get_cached_population_stats(region_geom, selected_year)
        gpp_stats = get_cached_gpp_stats(region_geom, selected_year)
        lst_stats = get_cached_lst_stats(region_geom, selected_year)
        ntl_stats = get_cached_ntl_stats(region_geom, selected_year)
        ndvi_stats = get_cached_ndvi_stats(region_geom, selected_year)

        if not all([pop_stats, gpp_stats, lst_stats, ntl_stats, ndvi_stats]):
            continue  # skip region if any stat missing

        row = {
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

        df = pd.DataFrame([row])

        if model_choice == "DNN":
            pred = predict_dnn(df)
        elif model_choice == "ML":
            pred = predict_ml(df)
        elif model_choice in ["DNN+RF", "DNN+XGBoost"]:
            pred = predict_ensemble(df, model_choice, alpha)
        else:
            pred = None

        if pred is not None:
            predictions.append(float(pred[0]))
            weights.append(pop_stats["Total Population"])  # use total pop as weight

    if predictions:
        # Weighted average
        total_weight = sum(weights)
        weighted_avg = sum(p * w for p, w in zip(predictions, weights)) / total_weight
        return weighted_avg
    else:
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
def get_region_center(country, region):
    filtered = fao_gaul.filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", country),
            ee.Filter.eq("ADM1_NAME", region),
        )
    )
    coords = filtered.geometry().centroid().coordinates().getInfo()
    return coords if coords else [0, 0]


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


def show_helper_tab():
    st.title("Region Explorer & MPI Prediction")

    country = st.selectbox(
        "Select a Country", get_country_list(), key="country_pred_new"
    )
    region_list = get_region_list(country)

    if region_list:
        region = st.selectbox("Select a Region", region_list)
        selected_year = st.selectbox("Select Year", list(range(2012, 2025)))

        center_coords = get_region_center(country, region)
        region_geom = get_region_geometry(country, region)

        # Show map
        m = folium.Map(
            location=[center_coords[1], center_coords[0]],
            zoom_start=6,
            control_scale=True,
        )
        folium.GeoJson(
            region_geom,
            name="Region Boundary",
            style_function=lambda x: {
                "color": "blue",
                "weight": 2,
                "fillOpacity": 0.1,
            },
            tooltip=region,
        ).add_to(m)
        folium_static(m, width=700, height=500)

        # Show population stats
        st.subheader(f"Population Statistics for {region} ({selected_year})")
        stats = get_cached_population_stats(region_geom, selected_year)
        if stats:
            st.write(stats)
        else:
            st.warning("No population data available or not enough to extrapolate.")

        # Show GPP stats
        st.subheader(f"GPP Statistics for {region} ({selected_year})")
        gpp_stats = get_cached_gpp_stats(region_geom, selected_year)
        if gpp_stats:
            st.write(gpp_stats)
        else:
            st.warning("No GPP data available for this region/year.")

        # Show VIIRS LST stats
        st.subheader(f"LST Statistics for {region} ({selected_year})")
        lst_stats = get_cached_lst_stats(region_geom, selected_year)
        if lst_stats:
            st.write(lst_stats)
        else:
            st.warning("No LST data available for this region/year.")

        # Show VIIRS NTL stats
        st.subheader(f"NTL Statistics for {region} ({selected_year})")
        ntl_stats = get_cached_ntl_stats(region_geom, selected_year)
        if ntl_stats:
            st.write(ntl_stats)
        else:
            st.warning("No NTL data available for this region/year.")

        # Show MODIS NDVI stats
        st.subheader(f"NDVI Statistics for {region} ({selected_year})")
        ndvi_stats = get_cached_ndvi_stats(region_geom, selected_year)
        if ndvi_stats:
            st.write(ndvi_stats)
        else:
            st.warning("No NDVI data available for this region/year.")

        # --- MPI Prediction Section ---
        st.subheader("🔮 MPI Prediction")

        model_choice = st.selectbox(
            "Select a model for prediction:",
            ["DNN", "ML", "DNN+RF", "DNN+XGBoost"],
            key="testing_model_new",
        )
        if model_choice in ["DNN+RF", "DNN+XGBoost"]:
            alpha = st.slider(
                "Ensemble Weight (DNN Contribution)",
                0.0,
                1.0,
                0.15,
                key="testing_alpha_new",
            )

        # Prepare feature vector
        if all([stats, gpp_stats, lst_stats, ntl_stats, ndvi_stats]):
            row = {
                "Mean_Pop": stats["Mean Population"],
                "Total_Pop": stats["Total Population"],
                "Min_Pop": stats["Min Population"],
                "Max_Pop": stats["Max Population"],
                "Median_Pop": stats["Median Population"],
                "StdDev_Pop": stats["Std Dev Population"],
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
            df = pd.DataFrame([row])
            if st.button(
                f"Predict MPI for {region} ({selected_year})", key="predict_button_new"
            ):
                with st.spinner("Generating prediction..."):
                    if model_choice == "DNN":
                        predictions = predict_dnn(df)
                    elif model_choice == "ML":
                        predictions = predict_ml(df)
                    elif model_choice == "DNN+RF":
                        predictions = predict_ensemble(df, "DNN+RF", alpha)
                    elif model_choice == "DNN+XGBoost":
                        predictions = predict_ensemble(df, "DNN+XGBoost", alpha)
                    if predictions is not None:
                        st.success("✅ MPI Prediction Complete!")
                        st.metric("Predicted MPI", round(float(predictions[0]), 5))
                    else:
                        st.error("❌ Failed to predict MPI.")
            if st.button(f"Predict Country-Level MPI for {country} ({selected_year})"):
                with st.spinner("Aggregating predictions from regions..."):
                    mpi_country = predict_country_level_mpi(
                        country,
                        selected_year,
                        model_choice,
                        alpha if "DNN+" in model_choice else None,
                    )
                    if mpi_country is not None:
                        st.success("✅ Country-level MPI prediction complete!")
                        st.metric("Predicted Country MPI", round(mpi_country, 5))
                    else:
                        st.error(
                            "⚠️ Failed to compute MPI: insufficient data across regions."
                        )

        else:
            st.warning(
                "Cannot predict MPI: missing input data from one or more sources."
            )

    else:
        st.warning("No regions available for this country.")
