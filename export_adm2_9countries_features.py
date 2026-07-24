"""
Export ALL 22 model features for the 9 target countries at GAUL ADM2 level.

This is a direct port of export_tr_features.py (the reference implementation that
matches updated_predictions.py exactly) from Turkey/ADM1-groups to the 9-country
ADM2 set. Every reducer, scale, mask, GHSL epoch rule, anomaly z-score and
population-extrapolation rule is preserved verbatim.

Why a port rather than reusing gee_export_adm2_9countries.py: that script exports
only 9 of the 22 features (no GHSL, no anomalies, no LST-day, no Median_NTL /
Mean_Pop) and still points at the deprecated MODIS/006 night collection.

Scale differences vs the Turkey script:
  Turkey = 26 regions; here = ~1,800 ADM2 districts. A single reduceRegions
  .getInfo() over all districts with ~60 anomaly bands exceeds GEE's payload
  limit, so regions are processed in chunks (--chunk, default 80) per country-year.

Per-feature scales (unchanged):
    pop, ghsl -> 100 | ntl, lst_night, ndvi, gpp, anomalies -> 500 | lst_day -> 1000

Usage:
    # validate first on one country/year (fast):
    python export_adm2_9countries_features.py --countries Montenegro --years 2020

    # full run:
    python export_adm2_9countries_features.py

Output: adm2_features_9countries_<start>_<end>.csv
"""
import argparse
import json
import os
import time

import ee
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
COUNTRIES = [
    "Bosnia and Herzegovina", "Egypt", "Jordan", "Kyrgyzstan",
    "Montenegro", "Morocco", "Tajikistan", "Tunisia", "Turkey",
]
GAUL_L2 = "FAO/GAUL_SIMPLIFIED_500m/2015/level2"
BUFFER_RADIUS_M = 500                     # must match app default (500m buffer)
TILE_SCALE = 4                            # raise if "User memory limit exceeded"
BASELINE_YEARS = list(range(2012, 2020))  # 2012–2019 baseline for anomalies
ID_PROP = "ADM2_CODE"

MODEL_FEATURES = [
    "Mean_NTL", "Mean_LST", "Median_NTL", "Mean_LST_Day", "NTL_anom",
    "StdDev_NTL", "StdDev_Pop", "ndvi_lst_ratio", "Mean_Pop", "Median_Pop",
    "Mean_GPP", "Sum_NTL", "NDVI_anom", "LSTN_anom", "LST_Day_anom",
    "NTL_anom_lag1", "Mean_BUILT_S", "Median_BUILT_S", "StdDev_BUILT_S",
    "StdDev_BUILT_V", "NTL_per_capita", "CV_Pop",
]

# Observed epochs only (capped at 2020) to match training and the prediction path.
GHSL_EPOCHS = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]

viirs_ntl = viirs_lst = modis_gpp = modis_lst_day = ndvi_v2 = worldpop = None
modis_lst_night = building_mask = None


def _nearest_ghsl_epoch(year):
    return min(GHSL_EPOCHS, key=lambda e: abs(e - year))


def init_ee():
    creds_path = os.path.expanduser("~/.config/earthengine/credentials")
    project = "ee-housseinrachini21"
    try:
        with open(creds_path) as f:
            project = json.load(f).get("project", project)
    except Exception:
        pass
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Initialize()


def _setup():
    global viirs_ntl, viirs_lst, modis_gpp, modis_lst_day, ndvi_v2, worldpop
    global modis_lst_night, building_mask

    viirs_ntl = ee.ImageCollection("NOAA/VIIRS/001/VNP46A2").select(
        "Gap_Filled_DNB_BRDF_Corrected_NTL"
    )
    viirs_lst = ee.ImageCollection("NASA/VIIRS/002/VNP21A1N").select("LST_1KM")
    modis_lst_night = ee.ImageCollection("MODIS/006/MOD11A2").select("LST_Night_1km")
    modis_gpp = ee.ImageCollection("MODIS/061/MOD17A3HGF").select("Gpp")
    modis_lst_day = ee.ImageCollection("MODIS/061/MOD11A1").select("LST_Day_1km")
    worldpop = ee.ImageCollection("WorldPop/GP/100m/pop").select("population")

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


# ── Per-year band builders (identical to export_tr_features.py) ────────────────

def _yearly_mean(coll, year, name):
    return coll.filterDate(f"{year}-01-01", f"{year}-12-31").mean().rename([name])


def _img_scale500(year):
    lag_year = year - 1
    anom_years = sorted(set(BASELINE_YEARS + ([lag_year] if lag_year >= 2012 else []) + [year]))

    def _modis_night(y, name):
        coll = modis_lst_night.filterDate(f"{y}-01-01", f"{y}-12-31")
        return ee.Image(ee.Algorithms.If(
            coll.size().gt(0),
            coll.mean().multiply(0.02).rename([name]),
            ee.Image.constant(0).rename([name]).updateMask(ee.Image.constant(0)),
        ))

    bands = [
        _yearly_mean(viirs_ntl, year, "NTL"),
        _yearly_mean(viirs_lst, year, "LST"),
        _modis_night(year, "LSTm"),
        _yearly_mean(ndvi_v2, year, "NDVI"),
        _yearly_mean(modis_gpp, year, "GPP"),
    ]
    gpp_mean = modis_gpp.filterDate(f"{year}-01-01", f"{year}-12-31").mean()
    bands.append(gpp_mean.mask().multiply(ee.Image.pixelArea()).rename(["GPParea"]))

    for y in anom_years:
        bands.append(_yearly_mean(viirs_ntl, y, f"NTLa_{y}"))
        bands.append(_yearly_mean(viirs_lst, y, f"LSTN_{y}"))
        bands.append(_modis_night(y, f"LSTNm_{y}"))
        bands.append(
            modis_lst_day.filterDate(f"{y}-01-01", f"{y}-12-31").mean()
            .multiply(0.02).rename([f"LSTD_{y}"])
        )
        bands.append(_yearly_mean(ndvi_v2, y, f"NDVIa_{y}"))

    return ee.Image.cat(bands).updateMask(building_mask), anom_years


def _img_scale100(year):
    epoch = _nearest_ghsl_epoch(year)
    built_s = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{epoch}").select("built_surface").rename(["BUILT_S"])
    built_v = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_V/{epoch}").select("built_volume_total").rename(["BUILT_V"])

    if year <= 2020:
        pop_bands = [_yearly_mean(worldpop, year, "pop")]
    else:
        pop_bands = [_yearly_mean(worldpop, y, f"pop_{y}") for y in range(2012, 2021)]
    pop_img = ee.Image.cat(pop_bands).updateMask(building_mask)

    return ee.Image.cat([pop_img, built_s, built_v])


def _img_scale1000(year):
    img = (
        modis_lst_day.filterDate(f"{year}-01-01", f"{year}-12-31").mean()
        .multiply(0.02).rename(["LSTday"])
    )
    return img.updateMask(building_mask)


# ── reduceRegions helpers ──────────────────────────────────────────────────────

def _R500():
    return (ee.Reducer.mean()
            .combine(ee.Reducer.median(), None, True)
            .combine(ee.Reducer.stdDev(), None, True)
            .combine(ee.Reducer.sum(), None, True))


def _R100():
    return (ee.Reducer.mean()
            .combine(ee.Reducer.median(), None, True)
            .combine(ee.Reducer.stdDev(), None, True)
            .combine(ee.Reducer.sum(), None, True))


def _R1000():
    return ee.Reducer.mean().combine(ee.Reducer.stdDev(), None, True)


def _reduce(image, fc, reducer, scale):
    """reduceRegions over the given FeatureCollection chunk."""
    res = image.reduceRegions(
        collection=fc, reducer=reducer, scale=scale, tileScale=TILE_SCALE
    ).getInfo()
    out = {}
    for feat in res["features"]:
        props = feat["properties"]
        out[props[ID_PROP]] = props
    return out


def _get(props, band, stat):
    if props is None:
        return None
    return props.get(f"{band}_{stat}", props.get(band))


# ── Anomaly z-scores (identical) ───────────────────────────────────────────────

def _zscore(props_by_year, prefix, stat, target_year, anom_years):
    bl = [y for y in BASELINE_YEARS if y in anom_years]
    vals = []
    for y in bl:
        v = _get(props_by_year.get(y), f"{prefix}_{y}", stat)
        if v is not None:
            vals.append(v)
    if len(vals) < 2:
        return None
    mu = np.mean(vals)
    sig = np.std(vals, ddof=1)
    if sig == 0:
        return 0.0
    val = _get(props_by_year.get(target_year), f"{prefix}_{target_year}", stat)
    return float((val - mu) / sig) if val is not None else None


def _zscore_fb(props_by_year, prefix, prefix_fb, stat, target_year, anom_years):
    bl = [y for y in BASELINE_YEARS if y in anom_years]

    def _v(y):
        p = props_by_year.get(y)
        v = _get(p, f"{prefix}_{y}", stat)
        return v if v is not None else _get(p, f"{prefix_fb}_{y}", stat)

    vals = [x for x in (_v(y) for y in bl) if x is not None]
    if len(vals) < 2:
        return None
    mu, sig = np.mean(vals), np.std(vals, ddof=1)
    if sig == 0:
        return 0.0
    val = _v(target_year)
    return float((val - mu) / sig) if val is not None else None


def _extrap_pop_stat(p100, stat, target_year):
    base_years = list(range(2012, 2021))
    vals = np.array([_get(p100, f"pop_{y}", stat) for y in base_years], dtype=float)
    mask = ~np.isnan(vals)
    if mask.sum() < 2:
        return None
    yrs = np.array(base_years)
    growth = np.mean(np.diff(vals[mask]) / np.diff(yrs[mask]))
    return float(vals[mask][-1] + growth * (target_year - yrs[mask][-1]))


# ── Per country-year assembly ──────────────────────────────────────────────────

def process_country_year(country, year, chunk_size, meta):
    """Return (rows, n_skipped) for one country-year, chunking the districts."""
    fc_all = ee.FeatureCollection(GAUL_L2).filter(ee.Filter.eq("ADM0_NAME", country))
    codes = meta[country]["codes"]

    img500, anom_years = _img_scale500(year)
    img100 = _img_scale100(year)
    img1k = _img_scale1000(year)

    rows, skipped = [], 0
    for i in range(0, len(codes), chunk_size):
        batch = codes[i:i + chunk_size]
        fc = fc_all.filter(ee.Filter.inList(ID_PROP, batch))

        s500 = _reduce(img500, fc, _R500(), 500)
        s100 = _reduce(img100, fc, _R100(), 100)
        s1000 = _reduce(img1k, fc, _R1000(), 1000)

        for code in batch:
            p500, p100, p1k = s500.get(code), s100.get(code), s1000.get(code)
            props_by_year = {y: p500 for y in anom_years}

            mean_ntl = _get(p500, "NTL", "mean")
            mean_lst = _get(p500, "LST", "mean")
            if mean_lst is None:
                mean_lst = _get(p500, "LSTm", "mean")
            median_ndvi = _get(p500, "NDVI", "median")
            gpp_sum = _get(p500, "GPP", "sum")
            gpp_area = _get(p500, "GPParea", "sum")

            if None in (mean_ntl, mean_lst, median_ndvi) or not gpp_sum or not gpp_area:
                skipped += 1
                continue

            if year <= 2020:
                mean_pop = _get(p100, "pop", "mean")
                median_pop = _get(p100, "pop", "median")
                stddev_pop = _get(p100, "pop", "stdDev")
                total_pop = _get(p100, "pop", "sum")
            else:
                mean_pop = _extrap_pop_stat(p100, "mean", year)
                median_pop = _extrap_pop_stat(p100, "median", year)
                stddev_pop = _extrap_pop_stat(p100, "stdDev", year)
                total_pop = _extrap_pop_stat(p100, "sum", year)
            if mean_pop is None:
                skipped += 1
                continue

            sum_ntl = _get(p500, "NTL", "sum")

            # Mean_LST_Day: scale-1000 reduce (matches training). Small districts whose
            # built-up area misses the 1km sampling grid come back null -> fall back to
            # the LSTD_{year} band already in the 500m stack. Same MOD11A1 annual mean,
            # same mask, only a finer sampling grid (MOD11A1 is natively ~1km, so this
            # oversamples the same pixels rather than adding information).
            mean_lst_day = (p1k.get("LSTday_mean", p1k.get("mean")) if p1k else None)
            lst_day_fallback = False
            if mean_lst_day is None:
                mean_lst_day = _get(p500, f"LSTD_{year}", "mean")
                lst_day_fallback = mean_lst_day is not None

            info = meta[country]["names"].get(code, {})
            adm1, adm2 = info.get("adm1"), info.get("adm2")
            # GAUL leaves ADM2_NAME as a placeholder in some countries (e.g. Montenegro),
            # where ADM1_NAME already carries the district name. Fall back so the
            # District column matches the existing prediction schema.
            if not adm2 or str(adm2).lower().startswith("administrative unit not"):
                adm2 = adm1
            row = {
                "Country": country,
                "Governorate": adm1,
                "District": adm2,
                "adm2_code": code,
                "year": year,
                "Mean_NTL": mean_ntl,
                "Mean_LST": mean_lst,
                "Median_NTL": _get(p500, "NTL", "median"),
                "Mean_LST_Day": mean_lst_day,
                "lst_day_500m_fallback": lst_day_fallback,
                "NTL_anom": _zscore(props_by_year, "NTLa", "mean", year, anom_years),
                "StdDev_NTL": _get(p500, "NTL", "stdDev"),
                "StdDev_Pop": stddev_pop,
                "ndvi_lst_ratio": (median_ndvi / mean_lst) if mean_lst else None,
                "Mean_Pop": mean_pop,
                "Median_Pop": median_pop,
                "Mean_GPP": gpp_sum / gpp_area,
                "Sum_NTL": sum_ntl,
                "Total_Pop": total_pop,
                "NTL_per_capita": (sum_ntl / total_pop) if total_pop else None,
                "CV_Pop": (stddev_pop / mean_pop) if mean_pop else None,
                "NDVI_anom": _zscore(props_by_year, "NDVIa", "median", year, anom_years),
                "LSTN_anom": _zscore_fb(props_by_year, "LSTN", "LSTNm", "mean", year, anom_years),
                "LST_Day_anom": _zscore(props_by_year, "LSTD", "mean", year, anom_years),
                "NTL_anom_lag1": (
                    _zscore(props_by_year, "NTLa", "mean", year - 1, anom_years)
                    if (year - 1) >= 2012 else None
                ),
                "Mean_BUILT_S": _get(p100, "BUILT_S", "mean"),
                "Median_BUILT_S": _get(p100, "BUILT_S", "median"),
                "StdDev_BUILT_S": _get(p100, "BUILT_S", "stdDev"),
                "StdDev_BUILT_V": _get(p100, "BUILT_V", "stdDev"),
            }
            missing = [f for f in MODEL_FEATURES
                       if row.get(f) is None or (isinstance(row.get(f), float) and row[f] != row[f])]
            if missing:
                skipped += 1
                continue
            rows.append(row)
    return rows, skipped


def load_meta(countries):
    """Fetch ADM2 codes + names per country once (no geometry)."""
    meta = {}
    base = ee.FeatureCollection(GAUL_L2)
    for c in countries:
        fc = base.filter(ee.Filter.eq("ADM0_NAME", c))
        info = fc.select(["ADM2_CODE", "ADM1_NAME", "ADM2_NAME"], None, False).getInfo()
        codes, names = [], {}
        for f in info["features"]:
            p = f["properties"]
            code = p.get("ADM2_CODE")
            if code is None:
                continue
            codes.append(code)
            names[code] = {"adm1": p.get("ADM1_NAME"), "adm2": p.get("ADM2_NAME")}
        meta[c] = {"codes": sorted(codes), "names": names}
        print(f"  {c:26} {len(codes):>4} ADM2 districts")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="*", default=COUNTRIES)
    # 2013 is the earliest usable year: NTL_anom_lag1 needs year-1 >= 2012, and
    # VNP46A2 has no 2011 data (collection starts 2012-01-19), so every 2012 row
    # would fail the 22-feature completeness check and be dropped.
    ap.add_argument("--years", nargs="*", type=int, default=list(range(2013, 2025)))
    ap.add_argument("--chunk", type=int, default=80, help="districts per reduceRegions call")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("Initialising Earth Engine...")
    init_ee()
    _setup()

    print(f"\nResolving ADM2 districts for {len(args.countries)} countries...")
    meta = load_meta(args.countries)
    total = sum(len(meta[c]["codes"]) for c in args.countries)
    print(f"\ntotal districts: {total:,} | years: {args.years} | chunk: {args.chunk}")
    print(f"~{sum(-(-len(meta[c]['codes']) // args.chunk) for c in args.countries) * len(args.years) * 3} reduceRegions calls\n")

    all_rows, all_skipped = [], 0
    for year in args.years:
        for c in args.countries:
            t0 = time.time()
            rows, skipped = process_country_year(c, year, args.chunk, meta)
            all_rows.extend(rows)
            all_skipped += skipped
            print(f"  {year} {c:26} {len(rows):>4} rows  ({skipped} skipped)  {time.time()-t0:5.1f}s")

    out = args.out or f"adm2_features_9countries_{min(args.years)}_{max(args.years)}.csv"
    df = pd.DataFrame(all_rows).sort_values(["Country", "adm2_code", "year"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"\nDone — {len(df):,} rows ({all_skipped} skipped) -> {out}")
    have = [f for f in MODEL_FEATURES if f in df.columns]
    print(f"model features present: {len(have)}/22")


if __name__ == "__main__":
    main()
