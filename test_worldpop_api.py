"""
Compare extrapolated WorldPop (2021-2023) vs WorldPop API actual values.

Fetches ADM1 geometries from GAUL via EE, queries the WorldPop stats API
for 2021-2023, and compares against the linearly-extrapolated values in
all_82_population_3000m_original_ref_gaul_actual.csv.

Sample: 8 ADM1 regions across diverse countries.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
POP_CSV  = (BASE_DIR / "gee_all_vars_3000m_original_ref_gaul"
            / "all_82_population_3000m_original_ref_gaul_actual.csv")
OUT_CSV  = BASE_DIR / "pop_comparison_2021_2023.csv"

# Small, diverse sample for testing
SAMPLE_REGIONS = [
    ("India",      "Maharashtra"),
    ("Nigeria",    "Lagos"),
    ("Brazil",     "Minas Gerais"),
    ("Mexico",     "Jalisco"),
    ("Ethiopia",   "Oromia"),
    ("Bangladesh", "Dhaka"),
    ("Kenya",      "Nairobi"),
    ("Pakistan",   "Punjab"),
]

TARGET_YEARS = [2021, 2022, 2023]
WP_API = "https://api.worldpop.org/v1/services/stats"


def init_ee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def get_geojson(gaul: ee.FeatureCollection, country: str, region: str) -> dict:
    feat = gaul.filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", country),
            ee.Filter.eq("ADM1_NAME", region),
        )
    ).first()
    return feat.geometry().getInfo()


def simplify_geojson(geojson: dict, tolerance: float = 0.05) -> dict:
    """Reduce vertex count by keeping only every Nth coordinate."""
    def _simplify_ring(ring, tol):
        if len(ring) <= 4:
            return ring
        # Douglas-Peucker not available without shapely — use stride sampling
        stride = max(1, int(len(ring) * tol))
        simplified = ring[::stride]
        if simplified[0] != simplified[-1]:
            simplified.append(simplified[0])
        return simplified

    def _simplify_coords(coords, depth=0):
        if depth == 0:          # polygon rings
            return [_simplify_ring(ring, tolerance) for ring in coords]
        return coords

    g = dict(geojson)
    gtype = g.get("type", "")
    if gtype == "Polygon":
        g["coordinates"] = _simplify_coords(g["coordinates"], depth=0)
    elif gtype == "MultiPolygon":
        # keep only the largest polygon to minimise payload
        polys = g["coordinates"]
        largest = max(polys, key=lambda p: len(p[0]))
        g = {"type": "Polygon", "coordinates": _simplify_coords(largest, depth=0)}
    return g


def query_api(geojson: dict, year: int) -> dict | None:
    simplified = simplify_geojson(geojson)
    payload = json.dumps(simplified)
    print(f"(geometry vertices: {len(json.loads(payload).get('coordinates', [[]])[0])})", end=" ")
    try:
        # POST to avoid URL-length limits on large polygons
        r = requests.post(
            WP_API,
            data={
                "dataset":  "wpgppop",
                "year":     year,
                "geojson":  payload,
                "runasync": "false",
            },
            timeout=90,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"    API error: {exc}")
        return None


def main():
    init_ee()
    gaul = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")

    df = pd.read_csv(POP_CSV, encoding="utf-8")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    records = []

    for country, region in SAMPLE_REGIONS:
        print(f"\n{'-'*50}")
        print(f"  {country} / {region}")

        print("  Fetching geometry from EE ...")
        geojson = get_geojson(gaul, country, region)

        extrap = (
            df[(df["Country"] == country) & (df["Region"] == region) &
               (df["Year"].isin(TARGET_YEARS))]
            .set_index("Year")[["Total Population", "Median Population"]]
        )

        for year in TARGET_YEARS:
            print(f"  Querying API for {year} ...", end=" ", flush=True)
            raw = query_api(geojson, year)

            # Print raw response once so we can see the schema
            if year == TARGET_YEARS[0] and country == SAMPLE_REGIONS[0][0]:
                print(f"\n  [Raw API response sample]\n  {json.dumps(raw, indent=2)[:600]}\n")

            api_total = None
            if raw:
                # navigate response — key may be 'data' or top-level
                data = raw.get("data") or raw
                if isinstance(data, dict):
                    api_total = (
                        data.get("total_population")
                        or data.get("totalPopulation")
                        or data.get("pop")
                        or data.get("total")
                    )
                elif isinstance(data, list) and data:
                    api_total = data[0].get("total_population") or data[0].get("pop")

            extrap_total = (
                float(extrap.loc[year, "Total Population"])
                if year in extrap.index else None
            )

            diff     = (api_total - extrap_total) if (api_total and extrap_total) else None
            pct_diff = (diff / extrap_total * 100)  if (diff is not None and extrap_total) else None

            print(f"extrap={extrap_total:,.0f}  api={api_total}  diff%={pct_diff}")

            records.append({
                "country":    country,
                "region":     region,
                "year":       year,
                "extrapolated_total": extrap_total,
                "api_total":          api_total,
                "abs_diff":           diff,
                "pct_diff":           round(pct_diff, 2) if pct_diff is not None else None,
            })

            time.sleep(0.5)

    out = pd.DataFrame(records)
    out.to_csv(OUT_CSV, index=False)
    print("=" * 60)
    print(f"Saved {OUT_CSV.name}")
    print("\nSummary (mean abs % diff by year):")
    print(out.groupby("year")["pct_diff"].agg(["mean", "min", "max"]).round(2).to_string())
    print("\nFull results:")
    print(out[["country", "region", "year", "extrapolated_total", "api_total", "pct_diff"]].to_string(index=False))


if __name__ == "__main__":
    main()
