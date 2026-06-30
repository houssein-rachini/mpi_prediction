"""
Export all 20 training features for Turkey TR-grouped regions to a local CSV.

Strategy: uses reduceRegions so ALL regions are reduced in a single GEE call per
feature-group per year (~3 calls/year instead of one-per-region). Anomaly z-scores
and population extrapolation are computed in Python afterwards so the output matches
updated_predictions.py exactly.

Per-feature scales are preserved to match the app:
    pop, ghsl  -> scale 100
    ntl, lst_night, ndvi, gpp, anomaly bands -> scale 500
    lst_day    -> scale 1000

Caveat vs app: the app falls back from VIIRS to MODIS night LST per-region when VIIRS
has no pixels. reduceRegions cannot do per-region fallback, so VIIRS night LST is used
for all regions. For Turkish provinces VIIRS night coverage is complete, so values match.

Usage:
    python export_tr_features.py

Output: tr_features_<start>_<end>.csv in the current directory.
"""
import ee
import json
import os
import time
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
YEARS = list(range(2012, 2025))          # 2012 – 2024 inclusive
BUFFER_RADIUS_M = 500                    # must match app default (500m buffer)
TR_ASSET_ID = "projects/ee-housseinrachini213/assets/TR_regions_admin1_groups"
OUTPUT_CSV = f"tr_features_{YEARS[0]}_{YEARS[-1]}.csv"
ID_PROP = "region_code"
TILE_SCALE = 4                           # raise if "User memory limit exceeded"
BASELINE_YEARS = list(range(2012, 2020)) # 2012–2019 baseline for anomalies

# The 20 features the model is trained on. Any row missing one is dropped (never zeroed).
MODEL_FEATURES = [
    "Mean_NTL", "Mean_LST", "Median_NTL", "Mean_LST_Day", "NTL_anom",
    "StdDev_NTL", "StdDev_Pop", "ndvi_lst_ratio", "Mean_Pop", "Median_Pop",
    "Mean_GPP", "Sum_NTL", "NDVI_anom", "LSTN_anom", "LST_Day_anom",
    "NTL_anom_lag1", "Mean_BUILT_S", "Median_BUILT_S", "StdDev_BUILT_S",
    "StdDev_BUILT_V",
]
# ───────────────────────────────────────────────────────────────────────────────

# Observed epochs only (capped at 2020) to match the training data and prediction path.
GHSL_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]

# Module-level EE objects (set after init)
viirs_ntl = viirs_lst = modis_gpp = modis_lst_day = ndvi_v2 = worldpop = None
building_mask = tr_fc = None


def _nearest_ghsl_epoch(year):
    return min(GHSL_EPOCHS, key=lambda e: abs(e - year))


def init_ee():
    creds_path = os.path.expanduser("~/.config/earthengine/credentials")
    with open(creds_path) as f:
        data = json.load(f)
    project = data.get("project", "ee-housseinrachini21")
    ee.Initialize(project=project)


def _setup():
    global viirs_ntl, viirs_lst, modis_gpp, modis_lst_day, ndvi_v2, worldpop
    global building_mask, tr_fc

    viirs_ntl = ee.ImageCollection("NOAA/VIIRS/001/VNP46A2").select(
        "Gap_Filled_DNB_BRDF_Corrected_NTL"
    )
    viirs_lst     = ee.ImageCollection("NASA/VIIRS/002/VNP21A1N").select("LST_1KM")
    modis_gpp     = ee.ImageCollection("MODIS/061/MOD17A3HGF").select("Gpp")
    modis_lst_day = ee.ImageCollection("MODIS/061/MOD11A2").select("LST_Day_1km")
    worldpop      = ee.ImageCollection("WorldPop/GP/100m/pop").select("population")

    def _ndvi(image):
        nd = image.normalizedDifference(["sur_refl_b02", "sur_refl_b01"]).rename("NDVI")
        return nd.copyProperties(image, image.propertyNames())

    ndvi_v2 = ee.ImageCollection("MODIS/061/MOD09A1").map(_ndvi)

    building_mask = (
        ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018")
        .select("built_height")
        .gte(2.5)
        .focal_max(kernel=ee.Kernel.circle(radius=BUFFER_RADIUS_M, units="meters"))
    )

    tr_fc = ee.FeatureCollection(TR_ASSET_ID)


# ── Per-year band builders (one masked multi-band image per scale group) ────────

def _yearly_mean(coll, year, name):
    return coll.filterDate(f"{year}-01-01", f"{year}-12-31").mean().rename([name])


def _img_scale500(year):
    """NTL (4 stats), LST night (mean), NDVI (median), GPP (sum + area), anomaly bands."""
    lag_year = year - 1
    anom_years = sorted(set(BASELINE_YEARS + ([lag_year] if lag_year >= 2012 else []) + [year]))

    bands = [
        _yearly_mean(viirs_ntl, year, "NTL"),
        _yearly_mean(viirs_lst, year, "LST"),
        _yearly_mean(ndvi_v2,   year, "NDVI"),
        _yearly_mean(modis_gpp, year, "GPP"),
    ]
    # GPP masked-area band: mask * pixelArea, summed later -> denominator for mean GPP
    gpp_mean = modis_gpp.filterDate(f"{year}-01-01", f"{year}-12-31").mean()
    bands.append(gpp_mean.mask().multiply(ee.Image.pixelArea()).rename(["GPParea"]))

    # Anomaly per-year bands (means for NTL/LSTN/LSTD, NDVI handled via median reducer)
    for y in anom_years:
        bands.append(_yearly_mean(viirs_ntl, y, f"NTLa_{y}"))
        bands.append(_yearly_mean(viirs_lst, y, f"LSTN_{y}"))
        bands.append(
            modis_lst_day.filterDate(f"{y}-01-01", f"{y}-12-31").mean()
            .multiply(0.02).rename([f"LSTD_{y}"])
        )
        bands.append(_yearly_mean(ndvi_v2, y, f"NDVIa_{y}"))

    return ee.Image.cat(bands).updateMask(building_mask), anom_years


def _img_scale100(year):
    """Population (building-masked) + GHSL built_s/built_v (NOT masked).

    Training masks population but reduces raw GHSL over the whole region, so we
    apply the building mask only to the population bands and leave GHSL unmasked
    (per-band masks; reduceRegions reduces each band over its own valid pixels).
    """
    epoch = _nearest_ghsl_epoch(year)
    # GHSL: no building mask (matches gee_export_ghsl_building.py)
    built_s = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{epoch}").select("built_surface").rename(["BUILT_S"])
    built_v = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_V/{epoch}").select("built_volume_total").rename(["BUILT_V"])

    # Population: building-masked (matches the population export)
    if year <= 2020:
        pop_bands = [_yearly_mean(worldpop, year, "pop")]
    else:
        pop_bands = [_yearly_mean(worldpop, y, f"pop_{y}") for y in range(2012, 2021)]
    pop_img = ee.Image.cat(pop_bands).updateMask(building_mask)

    return ee.Image.cat([pop_img, built_s, built_v])


def _img_scale1000(year):
    """Daytime LST (mean)."""
    img = (
        modis_lst_day.filterDate(f"{year}-01-01", f"{year}-12-31").mean()
        .multiply(0.02).rename(["LSTday"])
    )
    return img.updateMask(building_mask)


# ── reduceRegions helpers ───────────────────────────────────────────────────────

def _R500():
    return (
        ee.Reducer.mean()
        .combine(ee.Reducer.median(), None, True)
        .combine(ee.Reducer.stdDev(), None, True)
        .combine(ee.Reducer.sum(),    None, True)
    )


def _R100():
    return (
        ee.Reducer.mean()
        .combine(ee.Reducer.median(), None, True)
        .combine(ee.Reducer.stdDev(), None, True)
    )


def _R1000():
    return ee.Reducer.mean().combine(ee.Reducer.stdDev(), None, True)


def _reduce(image, reducer, scale):
    """reduceRegions over all TR regions, return {region_code: {prop: val}}."""
    fc = image.reduceRegions(
        collection=tr_fc,
        reducer=reducer,
        scale=scale,
        tileScale=TILE_SCALE,
    ).getInfo()
    out = {}
    for feat in fc["features"]:
        props = feat["properties"]
        out[props[ID_PROP]] = props
    return out


def _get(props, band, stat):
    """Read '<band>_<stat>' with fallback to plain '<band>'."""
    if props is None:
        return None
    return props.get(f"{band}_{stat}", props.get(band))


# ── Anomaly z-score (matches updated_predictions.py) ────────────────────────────

def _zscore(props_by_year, prefix, stat, target_year, anom_years):
    bl = [y for y in BASELINE_YEARS if y in anom_years]
    vals = []
    for y in bl:
        v = _get(props_by_year.get(y), f"{prefix}_{y}", stat)
        if v is not None:
            vals.append(v)
    if len(vals) < 2:
        return None
    mu  = np.mean(vals)
    sig = np.std(vals, ddof=1)
    if sig == 0:
        return 0.0
    val = _get(props_by_year.get(target_year), f"{prefix}_{target_year}", stat)
    return float((val - mu) / sig) if val is not None else None


# ── Main per-year assembly ──────────────────────────────────────────────────────

def process_year(year):
    img500, anom_years = _img_scale500(year)
    s500 = _reduce(img500, _R500(), 500)
    s100 = _reduce(_img_scale100(year), _R100(), 100)
    s1000 = _reduce(_img_scale1000(year), _R1000(), 1000)

    rows = []
    codes = sorted(s500.keys())
    for code in codes:
        p500 = s500.get(code)
        p100 = s100.get(code)
        p1k  = s1000.get(code)

        # All anomaly year-bands live in this region's p500 (single stacked reduce).
        props_by_year = {y: p500 for y in anom_years}

        mean_ntl   = _get(p500, "NTL", "mean")
        mean_lst   = _get(p500, "LST", "mean")
        median_ndvi = _get(p500, "NDVI", "median")
        gpp_sum    = _get(p500, "GPP", "sum")
        gpp_area   = _get(p500, "GPParea", "sum")

        # required-feature guard (same as app: pop/gpp/lst/ntl/ndvi must exist)
        if None in (mean_ntl, mean_lst, median_ndvi) or not gpp_sum or not gpp_area:
            continue

        # population
        if year <= 2020:
            mean_pop   = _get(p100, "pop", "mean")
            median_pop = _get(p100, "pop", "median")
            stddev_pop = _get(p100, "pop", "stdDev")
        else:
            mean_pop   = _extrap_pop_stat(p100, "mean",   year)
            median_pop = _extrap_pop_stat(p100, "median", year)
            stddev_pop = _extrap_pop_stat(p100, "stdDev", year)
        if mean_pop is None:
            continue

        row = {
            "region_code": code,
            "year": year,
            "Mean_NTL":      mean_ntl,
            "Mean_LST":      mean_lst,
            "Median_NTL":    _get(p500, "NTL", "median"),
            "Mean_LST_Day":  (p1k.get("LSTday_mean", p1k.get("mean")) if p1k else None),
            "NTL_anom":      _zscore(props_by_year, "NTLa", "mean", year, anom_years),
            "StdDev_NTL":    _get(p500, "NTL", "stdDev"),
            "StdDev_Pop":    stddev_pop,
            "ndvi_lst_ratio": (median_ndvi / mean_lst) if mean_lst else None,
            "Mean_Pop":      mean_pop,
            "Median_Pop":    median_pop,
            "Mean_GPP":      gpp_sum / gpp_area,
            "Sum_NTL":       _get(p500, "NTL", "sum"),
            "NDVI_anom":     _zscore(props_by_year, "NDVIa", "median", year, anom_years),
            "LSTN_anom":     _zscore(props_by_year, "LSTN", "mean", year, anom_years),
            "LST_Day_anom":  _zscore(props_by_year, "LSTD", "mean", year, anom_years),
            "NTL_anom_lag1": (
                _zscore(props_by_year, "NTLa", "mean", year - 1, anom_years)
                if (year - 1) >= 2012 else None
            ),
            "Mean_BUILT_S":   _get(p100, "BUILT_S", "mean"),
            "Median_BUILT_S": _get(p100, "BUILT_S", "median"),
            "StdDev_BUILT_S": _get(p100, "BUILT_S", "stdDev"),
            "StdDev_BUILT_V": _get(p100, "BUILT_V", "stdDev"),
        }
        missing = [f for f in MODEL_FEATURES
                   if row.get(f) is None or (isinstance(row.get(f), float) and row[f] != row[f])]
        if missing:
            print(f"    SKIP {code} {year}: missing {missing}")
            continue
        rows.append(row)
    return rows


def _extrap_pop_stat(p100, stat, target_year):
    base_years = list(range(2012, 2021))
    vals = np.array([_get(p100, f"pop_{y}", stat) for y in base_years], dtype=float)
    mask = ~np.isnan(vals)
    if mask.sum() < 2:
        return None
    yrs = np.array(base_years)
    growth = np.mean(np.diff(vals[mask]) / np.diff(yrs[mask]))
    return float(vals[mask][-1] + growth * (target_year - yrs[mask][-1]))


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("Initialising Earth Engine...")
    init_ee()
    _setup()

    codes = tr_fc.aggregate_array(ID_PROP).getInfo()
    print(f"TR regions ({len(codes)}): {codes}")
    print(f"Years: {YEARS[0]}–{YEARS[-1]}  |  ~{len(YEARS) * 3} reduceRegions calls\n")

    all_rows = []
    for year in YEARS:
        t0 = time.time()
        rows = process_year(year)
        all_rows.extend(rows)
        print(f"  {year}: {len(rows):>2} regions  ({time.time() - t0:4.1f}s)")

    df = pd.DataFrame(all_rows).sort_values(["region_code", "year"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone — {len(df)} rows saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
