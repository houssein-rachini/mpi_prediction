"""
Build adm2_predictions_9countries.csv

Countries : Bosnia and Herzegovina, Egypt, Jordan, Kyrgyzstan, Montenegro,
            Morocco, Tajikistan, Tunisia, Turkey
Year      : 2024
Models    : locally trained DNN, XGB (ML model), DNN+XGB ensemble

Runs predictions twice per district — with and without ndvi_lst_ratio —
using the same GEE feature fetch. ndvi_lst_ratio is computed and stored
in the cache; the no-ratio run simply omits it (scaler fills it as 0.0).

Output columns:
  country, adm1_code, adm2_code, adm2_name, year, population_2024,
  predicted_MPI_with_ratio, pred_DNN_with_ratio, pred_XGB_with_ratio,
  predicted_MPI_no_ratio,   pred_DNN_no_ratio,   pred_XGB_no_ratio
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.models import load_model
import xgboost as xgb
from tqdm import tqdm

BASE_DIR   = Path(__file__).resolve().parent
OUT_FILE   = BASE_DIR / "adm2_predictions_9countries.csv"
CACHE_FILE = BASE_DIR / "adm2_features_9countries_cache.csv"

COUNTRIES = [
    "Bosnia and Herzegovina", "Egypt", "Jordan", "Kyrgyzstan",
    "Montenegro", "Morocco", "Tajikistan", "Tunisia", "Turkey",
]
YEAR       = 2024
BUFFER_M   = 500
MAX_WORKERS = 2

FEATURES_WITH_RATIO = [
    "Mean_GPP", "StdDev_GPP", "Median_Pop", "StdDev_Pop",
    "Mean_LST", "StdDev_LST", "Mean_NTL", "StdDev_NTL", "Sum_NTL",
    "Median_NDVI", "StdDev_NDVI", "ndvi_lst_ratio",
]
FEATURES_NO_RATIO = [f for f in FEATURES_WITH_RATIO if f != "ndvi_lst_ratio"]

POP_YEARS     = list(range(2012, 2021))
WORLDPOP      = None
MODIS_GPP     = None
VIIRS_LST     = None
VIIRS_NTL     = None
NDVI_V2       = None
FAO_LVL2      = None
BUILDING_MASK = None


# ─────────────────────── GEE init & collections ───────────────────────────────

def init_ee() -> None:
    try:
        ee.Initialize()
    except Exception as e:
        raise RuntimeError("EE init failed — run `earthengine authenticate` first.") from e


def setup_collections():
    global WORLDPOP, MODIS_GPP, VIIRS_LST, VIIRS_NTL, NDVI_V2, FAO_LVL2, BUILDING_MASK
    WORLDPOP   = ee.ImageCollection("WorldPop/GP/100m/pop").select("population")
    MODIS_GPP  = ee.ImageCollection("MODIS/061/MOD17A3HGF").select("Gpp")
    VIIRS_LST  = ee.ImageCollection("NASA/VIIRS/002/VNP21A1N").select("LST_1KM")
    VIIRS_NTL  = ee.ImageCollection("NOAA/VIIRS/001/VNP46A2").select("Gap_Filled_DNB_BRDF_Corrected_NTL")
    NDVI_V2    = ee.ImageCollection("MODIS/061/MOD09A1").map(
        lambda img: img.normalizedDifference(["sur_refl_b02", "sur_refl_b01"])
            .rename("NDVI")
            .copyProperties(img, img.propertyNames())
    )
    FAO_LVL2   = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
    BUILDING_MASK = (
        ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018")
        .select("built_height")
        .gte(2.5)
        .focal_max(kernel=ee.Kernel.circle(radius=BUFFER_M, units="meters"))
    )


# ─────────────────────── GEE stat helpers ─────────────────────────────────────

def _geom(geom_dict):
    return ee.Geometry(geom_dict)


def _pop_stats_2024(geom_dict) -> dict | None:
    geom = _geom(geom_dict)
    stacked = ee.Image.cat([
        WORLDPOP
        .filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y, 12, 31))
        .mean().updateMask(BUILDING_MASK).rename([f"pop_{y}"])
        for y in POP_YEARS
    ])
    stats = stacked.reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(ee.Reducer.sum(), None, True)
            .combine(ee.Reducer.min(), None, True)
            .combine(ee.Reducer.max(), None, True)
            .combine(ee.Reducer.median(), None, True)
            .combine(ee.Reducer.stdDev(), None, True),
        geometry=geom, scale=100, bestEffort=True,
    ).getInfo()

    props = ["mean", "sum", "min", "max", "median", "stdDev"]
    data = {p: [] for p in props}
    for y in POP_YEARS:
        if stats.get(f"pop_{y}_mean") is not None:
            for p in props:
                data[p].append(stats[f"pop_{y}_{p}"])
        else:
            for p in props:
                data[p].append(None)

    def extrapolate(values):
        arr = np.array(values, dtype=float)
        yrs = np.array(POP_YEARS)
        mask = np.isfinite(arr)
        if mask.sum() < 2:
            return None
        growth = np.mean(np.diff(arr[mask]) / np.diff(yrs[mask]))
        return float(arr[mask][-1] + growth * (YEAR - yrs[mask][-1]))

    result = {p: extrapolate(data[p]) for p in props}
    if result["mean"] is None:
        return None
    return {
        "Median_Pop":      result["median"],
        "StdDev_Pop":      result["stdDev"],
        "population_2024": result["sum"],
    }


def _gpp_stats(geom_dict) -> dict | None:
    geom = _geom(geom_dict)
    image = MODIS_GPP.filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31").mean().updateMask(BUILDING_MASK)
    stats = image.reduceRegion(
        reducer=ee.Reducer.minMax()
            .combine(ee.Reducer.stdDev(), "", True)
            .combine(ee.Reducer.sum(), "", True),
        geometry=geom, scale=500, bestEffort=True, maxPixels=1e13,
    ).getInfo()
    if stats.get("Gpp_sum") is None:
        return None
    area = geom.area().getInfo()
    mean_gpp = stats["Gpp_sum"] / area if area else None
    if mean_gpp is None:
        return None
    return {"Mean_GPP": mean_gpp, "StdDev_GPP": stats["Gpp_stdDev"]}


def _lst_stats(geom_dict) -> dict | None:
    geom = _geom(geom_dict)
    start, end = f"{YEAR}-01-01", f"{YEAR}-12-31"
    viirs_img = VIIRS_LST.filterDate(start, end).mean().updateMask(BUILDING_MASK)
    count = viirs_img.reduceRegion(
        reducer=ee.Reducer.count(), geometry=geom, scale=500, bestEffort=True,
    ).get("LST_1KM")
    use_viirs = ee.Number(count).gt(0).getInfo()
    if use_viirs:
        img  = viirs_img
        band = "LST_1KM"
    else:
        img = (
            ee.ImageCollection("MODIS/006/MOD11A2").select("LST_Night_1km")
            .filterDate(start, end).mean().multiply(0.02).updateMask(BUILDING_MASK)
        )
        band = "LST_Night_1km"
    stats = img.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), None, True),
        geometry=geom, scale=500, bestEffort=True, maxPixels=1e13,
    ).getInfo()
    if stats.get(f"{band}_mean") is None:
        return None
    return {"Mean_LST": stats[f"{band}_mean"], "StdDev_LST": stats[f"{band}_stdDev"]}


def _ntl_stats(geom_dict) -> dict | None:
    geom = _geom(geom_dict)
    image = VIIRS_NTL.filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31").mean().updateMask(BUILDING_MASK)
    band  = "Gap_Filled_DNB_BRDF_Corrected_NTL"
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean()
            .combine(ee.Reducer.stdDev(), "", True)
            .combine(ee.Reducer.sum(), "", True),
        geometry=geom, scale=500, bestEffort=True, maxPixels=1e13,
    ).getInfo()
    if stats.get(f"{band}_mean") is None:
        return None
    return {
        "Mean_NTL":   stats[f"{band}_mean"],
        "StdDev_NTL": stats[f"{band}_stdDev"],
        "Sum_NTL":    stats[f"{band}_sum"],
    }


def _ndvi_stats(geom_dict) -> dict | None:
    geom  = _geom(geom_dict)
    image = NDVI_V2.filterDate(f"{YEAR}-01-01", f"{YEAR}-12-31").mean().updateMask(BUILDING_MASK)
    stats = image.reduceRegion(
        reducer=ee.Reducer.median().combine(ee.Reducer.stdDev(), "", True),
        geometry=geom, scale=500, bestEffort=True, maxPixels=1e13,
    ).getInfo()
    if stats.get("NDVI_median") is None:
        return None
    return {"Median_NDVI": stats["NDVI_median"], "StdDev_NDVI": stats["NDVI_stdDev"]}


def fetch_district_features(geom_dict) -> dict | None:
    with ThreadPoolExecutor(max_workers=5) as ex:
        f_pop  = ex.submit(_pop_stats_2024, geom_dict)
        f_gpp  = ex.submit(_gpp_stats,      geom_dict)
        f_lst  = ex.submit(_lst_stats,       geom_dict)
        f_ntl  = ex.submit(_ntl_stats,       geom_dict)
        f_ndvi = ex.submit(_ndvi_stats,      geom_dict)
    pop, gpp, lst, ntl, ndvi = (
        f_pop.result(), f_gpp.result(), f_lst.result(),
        f_ntl.result(), f_ndvi.result(),
    )
    if not all([pop, gpp, lst, ntl, ndvi]):
        return None
    row = {**pop, **gpp, **lst, **ntl, **ndvi}
    mean_lst    = row.get("Mean_LST")
    median_ndvi = row.get("Median_NDVI")
    row["ndvi_lst_ratio"] = (
        (median_ndvi / mean_lst)
        if (mean_lst and mean_lst != 0 and median_ndvi is not None)
        else 0.0
    )
    return row


# ─────────────────────── ADM2 metadata fetch ──────────────────────────────────

def fetch_adm2_districts(countries: list[str]) -> pd.DataFrame:
    print("Fetching ADM2 district list from FAO GAUL ...")
    fc   = FAO_LVL2.filter(ee.Filter.inList("ADM0_NAME", countries))
    data = fc.select(["ADM0_NAME", "ADM1_CODE", "ADM2_CODE", "ADM2_NAME"]).getInfo()
    rows = []
    for feat in data["features"]:
        p = feat["properties"]
        rows.append({
            "country":   p["ADM0_NAME"],
            "adm1_code": int(p["ADM1_CODE"]),
            "adm2_code": int(p["ADM2_CODE"]),
            "adm2_name": p["ADM2_NAME"],
        })
    df = pd.DataFrame(rows).drop_duplicates("adm2_code").reset_index(drop=True)
    print(f"  -> {len(df)} districts | {df['country'].value_counts().to_dict()}")
    return df


def fetch_adm2_geometry(adm2_code: int) -> dict | None:
    feat = FAO_LVL2.filter(ee.Filter.eq("ADM2_CODE", adm2_code)).first()
    geom = ee.Feature(feat).geometry().getInfo()
    if geom and geom.get("type") == "GeometryCollection":
        polys = [g for g in geom.get("geometries", []) if g.get("type") in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        geom = (
            {"type": "MultiPolygon", "coordinates": [p["coordinates"] for p in polys]}
            if len(polys) > 1 else polys[0]
        )
    return geom


# ─────────────────────── Model loading ────────────────────────────────────────

def load_models():
    custom = {
        "mse":  MeanSquaredError(),
        "mae":  MeanAbsoluteError(),
        "rmse": tf.keras.metrics.RootMeanSquaredError(),
    }
    dnn_model     = load_model(BASE_DIR / "trained_dnn_model.h5",              custom_objects=custom)
    dnn_scaler    = joblib.load(BASE_DIR / "dnn_scaler.pkl")
    ml_model      = joblib.load(BASE_DIR / "trained_ml_model.pkl")
    ml_scaler     = joblib.load(BASE_DIR / "ml_scaler.pkl")
    ens_dnn_model = load_model(BASE_DIR / "trained_ensemble_xgb_dnn_model.h5", custom_objects=custom)
    ens_xgb_model = xgb.XGBRegressor()
    ens_xgb_model.load_model(BASE_DIR / "trained_ensemble_xgb_model.json")
    ens_scaler    = joblib.load(BASE_DIR / "ensemble_scaler.pkl")
    return {
        "dnn_model": dnn_model, "dnn_scaler": dnn_scaler,
        "ml_model":  ml_model,  "ml_scaler":  ml_scaler,
        "ens_dnn":   ens_dnn_model, "ens_xgb": ens_xgb_model, "ens_scaler": ens_scaler,
    }


def _scale_and_fill(df_input, scaler):
    feat_names = scaler.feature_names_in_
    for col in feat_names:
        if col not in df_input.columns:
            df_input[col] = 0.0
    return scaler.transform(df_input[feat_names])


def predict_all(feature_row: dict, models: dict) -> tuple[float, float, float]:
    df = pd.DataFrame([feature_row])
    X_dnn    = _scale_and_fill(df.copy(), models["dnn_scaler"])
    pred_dnn = float(np.clip(models["dnn_model"].predict(X_dnn, verbose=0).flatten()[0], 0, 1))
    X_ml     = _scale_and_fill(df.copy(), models["ml_scaler"])
    pred_xgb = float(np.clip(models["ml_model"].predict(X_ml)[0], 0, 1))
    X_ens    = _scale_and_fill(df.copy(), models["ens_scaler"])
    p_dnn_e  = models["ens_dnn"].predict(X_ens, verbose=0).flatten()[0]
    p_xgb_e  = models["ens_xgb"].predict(X_ens)[0]
    pred_ens = float(np.clip(0.4 * p_dnn_e + 0.6 * p_xgb_e, 0, 1))
    return pred_ens, pred_dnn, pred_xgb


# ─────────────────────── Main ─────────────────────────────────────────────────

def process_district(row: pd.Series) -> dict | None:
    adm2_code = int(row["adm2_code"])
    try:
        geom = fetch_adm2_geometry(adm2_code)
        if geom is None:
            return None
        features = fetch_district_features(geom)
        if features is None:
            return None
        return {"adm2_code": adm2_code, "features": features}
    except Exception as e:
        print(f"  [SKIP] adm2_code={adm2_code} ({row.get('adm2_name')}): {e}")
        return None


def main():
    print("Initialising Earth Engine ...")
    init_ee()
    print("  -> EE ready.")

    print("Setting up GEE collections ...")
    setup_collections()
    print("  -> Collections ready.")

    districts = fetch_adm2_districts(COUNTRIES)

    if CACHE_FILE.exists():
        cached     = pd.read_csv(CACHE_FILE, encoding="utf-8")
        done_codes = set(cached["adm2_code"].astype(int).tolist())
        remaining  = districts[~districts["adm2_code"].isin(done_codes)]
        print(f"Resuming: {len(done_codes)} cached, {len(remaining)} remaining.")
        cache_rows = cached.to_dict("records")
    else:
        remaining  = districts
        cache_rows = []
        print("No cache found — fetching all districts from scratch.")

    total     = len(remaining)
    rows_list = remaining.to_dict("records")
    skipped   = [0]

    def _process_row(row):
        result = process_district(row)
        if result is None:
            skipped[0] += 1
            return None
        return {
            "country":   row["country"],
            "adm1_code": row["adm1_code"],
            "adm2_code": row["adm2_code"],
            "adm2_name": row["adm2_name"],
            **result["features"],
        }

    print(f"Fetching GEE features for {total} districts (MAX_WORKERS={MAX_WORKERS}) ...")
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fs = {ex.submit(_process_row, row): row for row in rows_list}
        with tqdm(total=total, unit="district", dynamic_ncols=True) as pbar:
            for future in as_completed(fs):
                row = fs[future]
                pbar.set_postfix(
                    country=row["country"],
                    district=row["adm2_name"][:18],
                    skipped=skipped[0],
                )
                futures.append(future.result())
                pbar.update(1)

    new_rows = [r for r in futures if r is not None]
    cache_rows.extend(new_rows)
    print(f"GEE fetch done: {len(new_rows)} succeeded, {skipped[0]} skipped.")

    pd.DataFrame(cache_rows).to_csv(CACHE_FILE, index=False, encoding="utf-8")
    print(f"Feature cache saved -> {CACHE_FILE.name} ({len(cache_rows)} districts).")

    print("Loading locally trained models ...")
    models = load_models()
    print("  -> Models loaded.")

    results = []
    for feat_row in tqdm(cache_rows, desc="Predicting", unit="district", dynamic_ncols=True):
        # with ratio — needs all 12 features
        feat_with = {k: feat_row[k] for k in FEATURES_WITH_RATIO if k in feat_row}
        # no ratio — 11 features; scaler fills ndvi_lst_ratio=0.0 automatically
        feat_no   = {k: feat_row[k] for k in FEATURES_NO_RATIO if k in feat_row}

        if len(feat_with) < len(FEATURES_WITH_RATIO) or len(feat_no) < len(FEATURES_NO_RATIO):
            missing = [k for k in FEATURES_WITH_RATIO if k not in feat_row]
            print(f"  [SKIP predict] {feat_row.get('adm2_name')} — missing: {missing}")
            continue

        ens_w, dnn_w, xgb_w = predict_all(feat_with, models)
        ens_n, dnn_n, xgb_n = predict_all(feat_no,   models)

        results.append({
            "country":                  feat_row["country"],
            "adm1_code":                feat_row["adm1_code"],
            "adm2_code":                feat_row["adm2_code"],
            "adm2_name":                feat_row["adm2_name"],
            "year":                     YEAR,
            "population_2024":          feat_row.get("population_2024"),
            "predicted_MPI_with_ratio": ens_w,
            "pred_DNN_with_ratio":      dnn_w,
            "pred_XGB_with_ratio":      xgb_w,
            "predicted_MPI_no_ratio":   ens_n,
            "pred_DNN_no_ratio":        dnn_n,
            "pred_XGB_no_ratio":        xgb_n,
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"Done. Saved {OUT_FILE.name} ({len(df_out)} rows).")
    print(f"  -> {df_out['country'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
