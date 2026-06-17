"""
Standalone ADM2-level MPI predictions for 5 countries, year 2024.
Runs DNN-only and XGB-only (trained_ml_model.pkl) models.
Mimics updated_predictions.py stat functions exactly.

Output (one pair of files per country):
  dnn_only_{country}.csv
  xgb_only_{country}.csv

Columns: Country, Governorate, District, ADM1_CODE, ADM2_CODE,
         Year, Predicted MPI, Predicted Severe Poverty %
"""

from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.models import load_model
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR
YEAR     = 2024
BUFFER_M = 500
MAX_WORKERS_OUTER = 2

COUNTRIES = ["Egypt", "Jordan", "Kyrgyzstan", "Morocco", "Tajikistan"]

MODEL_DNN_PATH    = BASE_DIR / "trained_dnn_model.h5"
SCALER_DNN_PATH   = BASE_DIR / "dnn_scaler.pkl"
MODEL_XGB_PATH    = BASE_DIR / "trained_ml_model.pkl"
SCALER_XGB_PATH   = BASE_DIR / "ml_scaler.pkl"

# ─── GEE init ─────────────────────────────────────────────────────────────────
ee.Initialize()

# ─── GEE datasets ─────────────────────────────────────────────────────────────
fao_gaul_lvl2 = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
worldpop      = ee.ImageCollection("WorldPop/GP/100m/pop").select("population")
modis_gpp     = ee.ImageCollection("MODIS/061/MOD17A3HGF").select("Gpp")
viirs_lst     = ee.ImageCollection("NASA/VIIRS/002/VNP21A1N").select("LST_1KM")
viirs_ntl     = ee.ImageCollection("NOAA/VIIRS/001/VNP46A2").select(
    "Gap_Filled_DNB_BRDF_Corrected_NTL"
)
ndvi_v2 = ee.ImageCollection("MODIS/061/MOD09A1").map(
    lambda img: img.normalizedDifference(["sur_refl_b02", "sur_refl_b01"])
    .rename("NDVI")
    .copyProperties(img, img.propertyNames())
)

# ─── Building mask ────────────────────────────────────────────────────────────
BUILDING_MASK = (
    ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018")
    .select("built_height")
    .gte(2.5)
    .focal_max(kernel=ee.Kernel.circle(radius=BUFFER_M, units="meters"))
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def compute_sev_pov(mpi: float) -> float:
    pov = 0.04133 + 34.58 * mpi + 263 * (mpi ** 2) - 180.8 * (mpi ** 3)
    return min(100.0, max(0.0, pov))


def _geom(geom_dict) -> ee.Geometry:
    return ee.Geometry(geom_dict)


# ─── Per-district stat functions (mirror updated_predictions.py exactly) ──────

def _pop_stats(geom_dict: dict) -> dict | None:
    geom = _geom(geom_dict)
    years = list(range(2012, 2021))
    stacked = ee.Image.cat([
        worldpop
        .filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y, 12, 31))
        .mean()
        .updateMask(BUILDING_MASK)
        .rename([f"pop_{y}"])
        for y in years
    ])
    all_stats = stacked.reduceRegion(
        reducer=ee.Reducer.mean()
        .combine(ee.Reducer.median(), None, True)
        .combine(ee.Reducer.stdDev(), None, True)
        .combine(ee.Reducer.sum(), None, True),
        geometry=geom,
        scale=100,
        bestEffort=True,
    ).getInfo()

    def extrapolate(vals, yrs, target):
        vals = np.array(vals, dtype=np.float64)
        yrs  = np.array(yrs)
        mask = ~np.isnan(vals)
        if mask.sum() < 2:
            return None
        growth = np.mean(np.diff(vals[mask]) / np.diff(yrs[mask]))
        return float(vals[mask][-1] + growth * (target - yrs[mask][-1]))

    props = ["mean", "median", "stdDev", "sum"]
    data: dict[str, list] = {p: [] for p in props}
    for y in years:
        mv = all_stats.get(f"pop_{y}_mean")
        if mv is not None:
            data["mean"].append(mv)
            data["median"].append(all_stats.get(f"pop_{y}_median"))
            data["stdDev"].append(all_stats.get(f"pop_{y}_stdDev"))
            data["sum"].append(all_stats.get(f"pop_{y}_sum"))
        else:
            for k in data:
                data[k].append(None)

    results = {}
    for k in data:
        v = extrapolate(data[k], years, YEAR)
        results[k] = round(v, 2) if v is not None else None

    if results["mean"] is None:
        return None
    return {
        "Median_Pop": results["median"],
        "StdDev_Pop": results["stdDev"],
        "Total_Pop":  results["sum"],
    }


def _gpp_stats(geom_dict: dict) -> dict | None:
    geom  = _geom(geom_dict)
    image = (
        modis_gpp
        .filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31")
        .mean()
        .updateMask(BUILDING_MASK)
    )
    stats = image.reduceRegion(
        reducer=ee.Reducer.minMax()
        .combine(ee.Reducer.stdDev(), "", True)
        .combine(ee.Reducer.sum(), "", True),
        geometry=geom,
        scale=500,
        bestEffort=True,
        maxPixels=1e13,
    ).getInfo()
    if stats.get("Gpp_sum") is None:
        return None
    area     = geom.area().getInfo()
    mean_gpp = stats["Gpp_sum"] / area if area else None
    if mean_gpp is None:
        return None
    return {
        "Mean_GPP":   mean_gpp,
        "StdDev_GPP": stats.get("Gpp_stdDev"),
    }


def _lst_stats(geom_dict: dict) -> dict | None:
    geom  = _geom(geom_dict)
    start = ee.Date.fromYMD(YEAR, 1, 1)
    end   = ee.Date.fromYMD(YEAR, 12, 31)

    viirs_img   = viirs_lst.filterDate(start, end).mean().updateMask(BUILDING_MASK)
    viirs_count = viirs_img.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=geom,
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
        .updateMask(BUILDING_MASK)
    )

    use_viirs = bool(ee.Number(viirs_count).gt(0).getInfo())
    img  = viirs_img if use_viirs else modis_img
    band = "LST_1KM" if use_viirs else "LST_Night_1km"

    stats = img.reduceRegion(
        reducer=ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), None, True),
        geometry=geom,
        scale=500,
        bestEffort=True,
        maxPixels=1e13,
    ).getInfo()

    mean_key = f"{band}_mean"
    std_key  = f"{band}_stdDev"
    if mean_key not in stats:
        return None
    return {
        "Mean_LST":   stats[mean_key],
        "StdDev_LST": stats.get(std_key),
    }


def _ntl_stats(geom_dict: dict) -> dict | None:
    geom  = _geom(geom_dict)
    image = (
        viirs_ntl
        .filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31")
        .mean()
        .updateMask(BUILDING_MASK)
    )
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), "", True)
        .combine(ee.Reducer.sum(), "", True),
        geometry=geom,
        scale=500,
        bestEffort=True,
        maxPixels=1e13,
    ).getInfo()
    b = "Gap_Filled_DNB_BRDF_Corrected_NTL"
    if f"{b}_mean" not in stats:
        return None
    return {
        "Mean_NTL":   stats[f"{b}_mean"],
        "StdDev_NTL": stats.get(f"{b}_stdDev"),
        "Sum_NTL":    stats.get(f"{b}_sum"),
    }


def _ndvi_stats(geom_dict: dict) -> dict | None:
    geom  = _geom(geom_dict)
    image = (
        ndvi_v2
        .filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31")
        .mean()
        .updateMask(BUILDING_MASK)
    )
    stats = image.reduceRegion(
        reducer=ee.Reducer.median()
        .combine(ee.Reducer.stdDev(), None, True),
        geometry=geom,
        scale=500,
        bestEffort=True,
        maxPixels=1e13,
    ).getInfo()
    if "NDVI_median" not in stats:
        return None
    return {
        "Median_NDVI": stats["NDVI_median"],
        "StdDev_NDVI": stats.get("NDVI_stdDev"),
    }


def fetch_features(geom_dict: dict) -> dict | None:
    """Fetch all 5 stat groups in parallel and assemble feature row."""
    with ThreadPoolExecutor(max_workers=5) as ex:
        f_pop  = ex.submit(_pop_stats,  geom_dict)
        f_gpp  = ex.submit(_gpp_stats,  geom_dict)
        f_lst  = ex.submit(_lst_stats,  geom_dict)
        f_ntl  = ex.submit(_ntl_stats,  geom_dict)
        f_ndvi = ex.submit(_ndvi_stats, geom_dict)
    pop  = f_pop.result()
    gpp  = f_gpp.result()
    lst  = f_lst.result()
    ntl  = f_ntl.result()
    ndvi = f_ndvi.result()

    if not all([pop, gpp, lst, ntl, ndvi]):
        return None

    return {
        "Median_Pop":  pop["Median_Pop"],
        "StdDev_Pop":  pop["StdDev_Pop"],
        "Mean_GPP":    gpp["Mean_GPP"],
        "StdDev_GPP":  gpp["StdDev_GPP"],
        "Mean_LST":    lst["Mean_LST"],
        "StdDev_LST":  lst["StdDev_LST"],
        "Mean_NTL":    ntl["Mean_NTL"],
        "StdDev_NTL":  ntl["StdDev_NTL"],
        "Sum_NTL":     ntl["Sum_NTL"],
        "Median_NDVI": ndvi["Median_NDVI"],
        "StdDev_NDVI": ndvi["StdDev_NDVI"],
        # ndvi_lst_ratio intentionally omitted -> preprocess_data fills with 0
    }


# ─── Model / scaler loading ───────────────────────────────────────────────────
def load_dnn():
    return load_model(
        str(MODEL_DNN_PATH),
        custom_objects={
            "mse":  MeanSquaredError(),
            "mae":  MeanAbsoluteError(),
            "rmse": tf.keras.metrics.RootMeanSquaredError(),
        },
    )


def load_xgb():
    return joblib.load(str(MODEL_XGB_PATH))


def preprocess(feature_row: dict, scaler) -> np.ndarray:
    df = pd.DataFrame([feature_row])
    for col in scaler.feature_names_in_:
        if col not in df.columns:
            df[col] = 0.0
    return scaler.transform(df[scaler.feature_names_in_])


# ─── District meta ────────────────────────────────────────────────────────────
def get_districts(country: str) -> list[dict]:
    """Return list of {adm2_code, adm2_name, adm1_name, adm1_code, geom}."""
    fc = fao_gaul_lvl2.filter(ee.Filter.eq("ADM0_NAME", country))
    codes  = fc.aggregate_array("ADM2_CODE").getInfo()
    names  = fc.aggregate_array("ADM2_NAME").getInfo()
    gov    = fc.aggregate_array("ADM1_NAME").getInfo()
    adm1c  = fc.aggregate_array("ADM1_CODE").getInfo()
    return [
        {"adm2_code": c, "adm2_name": n, "adm1_name": g, "adm1_code": a1}
        for c, n, g, a1 in zip(codes, names, gov, adm1c)
    ]


def get_geom(country: str, adm2_code: int) -> dict:
    fc = fao_gaul_lvl2.filter(ee.Filter.eq("ADM0_NAME", country))
    return fc.filter(ee.Filter.eq("ADM2_CODE", adm2_code)).geometry().getInfo()


# ─── Main ─────────────────────────────────────────────────────────────────────
def run_country(country: str, dnn_model, dnn_scaler, xgb_model, xgb_scaler):
    print(f"\nProcessing {country} ...")
    districts = get_districts(country)

    dnn_rows: list[dict] = []
    xgb_rows: list[dict] = []

    def process(d: dict):
        try:
            geom_dict = get_geom(country, d["adm2_code"])
            feats = fetch_features(geom_dict)
            if feats is None:
                return None
            return (d, feats)
        except Exception as exc:
            return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_OUTER) as ex:
        futures = {ex.submit(process, d): d for d in districts}
        for fut in tqdm(futures, total=len(districts), desc=country, unit="district"):
            result = fut.result()
            if result is None:
                continue
            d, feats = result

            # DNN prediction
            X_dnn = preprocess(feats, dnn_scaler)
            mpi_dnn = float(np.clip(dnn_model.predict(X_dnn, verbose=0).flatten()[0], 0, 1))
            dnn_rows.append({
                "Country":                  country,
                "Governorate":              d["adm1_name"],
                "District":                 d["adm2_name"],
                "ADM1_CODE":                d["adm1_code"],
                "ADM2_CODE":                d["adm2_code"],
                "Year":                     YEAR,
                "Predicted MPI":            round(mpi_dnn, 6),
                "Predicted Severe Poverty %": round(compute_sev_pov(mpi_dnn), 4),
            })

            # XGB prediction
            X_xgb = preprocess(feats, xgb_scaler)
            mpi_xgb = float(np.clip(xgb_model.predict(X_xgb).flatten()[0], 0, 1))
            xgb_rows.append({
                "Country":                  country,
                "Governorate":              d["adm1_name"],
                "District":                 d["adm2_name"],
                "ADM1_CODE":                d["adm1_code"],
                "ADM2_CODE":                d["adm2_code"],
                "Year":                     YEAR,
                "Predicted MPI":            round(mpi_xgb, 6),
                "Predicted Severe Poverty %": round(compute_sev_pov(mpi_xgb), 4),
            })

    safe_name = country.lower().replace(" ", "_")
    dnn_path = OUT_DIR / f"dnn_only_{safe_name}.csv"
    xgb_path = OUT_DIR / f"xgb_only_{safe_name}.csv"
    pd.DataFrame(dnn_rows).to_csv(dnn_path, index=False, encoding="utf-8")
    pd.DataFrame(xgb_rows).to_csv(xgb_path, index=False, encoding="utf-8")
    print(f"  Saved {dnn_path.name} ({len(dnn_rows)} rows)")
    print(f"  Saved {xgb_path.name} ({len(xgb_rows)} rows)")


def main():
    print("Loading models ...")
    dnn_model  = load_dnn()
    dnn_scaler = joblib.load(str(SCALER_DNN_PATH))
    xgb_model  = load_xgb()
    xgb_scaler = joblib.load(str(SCALER_XGB_PATH))

    for country in COUNTRIES:
        run_country(country, dnn_model, dnn_scaler, xgb_model, xgb_scaler)

    print("\nDone.")


if __name__ == "__main__":
    main()
